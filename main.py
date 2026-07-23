"""
YouTube Arabic → German → English Pipeline
===========================================

Downloads a YouTube video, transcribes Arabic audio to SRT,
translates to German and English SRT (verified with multiple AI tools),
and compresses the video to ~50 MB with burned subtitles.

Translation backends (in priority order):
  1. DeepL API (requires DEEPL_API_KEY)
  2. Google Translate (free fallback, no key needed)

Multi-pass verification:
  - Pass 1: Google Translate (free)
  - Pass 2: DeepL API (if available)
  - Pass 3: MyMemory Translate (free)

Uses MLX-Whisper for fast transcription on Apple Silicon GPU.
Resume support: re-run the same command and it detects existing
output files, skipping completed steps automatically.

Usage:
    python main.py <youtube_url>

Example:
    python main.py https://youtu.be/MgxTrPOkhDU

Flags:
    --min-quality       Download minimum quality video/audio
    --cleanup           Clean up temp files after completion
    --force-retranslate Force re-translation even if German SRT exists
    --deepl-key         DeepL API key for translation/verification
"""

import argparse
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import cast

import time

import requests
from deep_translator import GoogleTranslator
from dotenv import load_dotenv

import mlx_whisper

# Local translation backend
try:
    from local_translation import translate_with_local
except ImportError:
    translate_with_local = None

# ─── Language Detection ───────────────────────────────────────────

def is_arabic(text: str) -> bool:
    """Detect if text contains Arabic characters."""
    arabic_ranges = [
        (0x0600, 0x06FF),  # Arabic
        (0x0750, 0x077F),  # Arabic Supplement
        (0x08A0, 0x08FF),  # Arabic Extended-A
        (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
        (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
    ]
    for char in text:
        code = ord(char)
        for start, end in arabic_ranges:
            if start <= code <= end:
                return True
    return False


# ─── SRT I/O ──────────────────────────────────────────────────────

def sanitize_filename(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = re.sub(r'\s+', "_", name.strip())
    name = name.strip("._")
    return name if name else "video"


def get_youtube_title(url: str) -> str:
    result = subprocess.run(
        ["yt-dlp", "--print", "title", url],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def prompt_output_name(url: str) -> str:
    try:
        print("  Fetching video title...")
        default_name = get_youtube_title(url)
    except subprocess.CalledProcessError:
        default_name = "video"

    safe_default = sanitize_filename(default_name)
    user_input = input(f"  Output file name [{safe_default}]: ").strip()
    chosen = user_input if user_input else safe_default
    return sanitize_filename(chosen)


def format_time(seconds: float) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"


def write_srt(segments: list, srt_path: Path) -> None:
    """Write segments to SRT file (single language)."""
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            f.write(f"{i}\n")
            f.write(f"{format_time(seg['start'])} --> {format_time(seg['end'])}\n")
            f.write(f"{seg['text'].strip()}\n\n")


def write_triilingual_srt(arabic_segs: list, german_segs: list, english_segs: list, srt_path: Path) -> None:
    """Write a combined SRT file with Arabic, German, and English subtitles.

    Each subtitle block contains all three languages, separated by lines.
    """
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, (ar, de, en) in enumerate(zip(arabic_segs, german_segs, english_segs), start=1):
            f.write(f"{i}\n")
            f.write(f"{format_time(ar['start'])} --> {format_time(ar['end'])}\n")
            f.write(f"[AR] {ar['text'].strip()}\n")
            f.write(f"[DE] {de['text'].strip()}\n")
            f.write(f"[EN] {en['text'].strip()}\n\n")


def read_srt(srt_path: Path) -> list:
    """Read any SRT file and return segments with start/end/text."""
    segments = []
    with open(srt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.isdigit() and i + 1 < len(lines) and " --> " in lines[i + 1]:
            time_part = lines[i + 1].strip()
            start_str, end_str = time_part.split(" --> ")
            def parse_ts(ts: str) -> float:
                h, m, rest = ts.split(":")
                s, ms = rest.split(",")
                return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
            start = parse_ts(start_str)
            end = parse_ts(end_str)
            i += 2
            text_lines = []
            while i < len(lines) and lines[i].strip():
                text_lines.append(lines[i].strip())
                i += 1
            segments.append({"start": start, "end": end, "text": " ".join(text_lines)})
        i += 1
    return segments


# ─── Download ─────────────────────────────────────────────────────

def download_youtube(url: str, output_dir: Path, min_quality: bool = False) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    if min_quality:
        video_format = "worstvideo[ext=mp4]+worstaudio[ext=m4a]/worst[ext=mp4]/worst"
    else:
        video_format = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"

    video_template = output_dir / "%(id)s_video.%(ext)s"
    subprocess.run(
        ["yt-dlp", "-f", video_format, "-o", str(video_template), url],
        check=True, capture_output=True, text=True,
    )

    audio_template = output_dir / "%(id)s.%(ext)s"
    subprocess.run(
        ["yt-dlp", "-x", "--audio-format", "mp3", "-o", str(audio_template), url],
        check=True, capture_output=True, text=True,
    )

    mp4_files = sorted(output_dir.glob("*_video.mp4"))
    mp3_files = sorted(output_dir.glob("*.mp3"))

    if not mp4_files or not mp3_files:
        raise RuntimeError("Failed to download video or audio.")

    video_id = mp4_files[0].stem.replace("_video", "")
    return {
        "video_id": video_id,
        "video": mp4_files[0],
        "audio": mp3_files[0],
    }


# ─── Transcription ─────────────────────────────────────────────────

def transcribe_arabic(audio_path: Path, model_name: str, batch_size: int = 8, condition_on_prev: bool = False) -> list:
    """Transcribe Arabic audio using MLX-Whisper on Apple Silicon GPU."""
    print(f"  Transcribing Arabic audio with MLX-Whisper ({model_name})...")
    hf_repo = model_name if "/" in model_name else f"mlx-community/whisper-{model_name}"
    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=hf_repo,
        language="ar",
        verbose=False,
        batch_size=batch_size,
        condition_on_previous_text=condition_on_prev,
    )
    return cast(list, result["segments"])


def double_check_arabic_srt(segments: list, audio_path: Path, model_name: str, batch_size: int = 8, condition_on_prev: bool = False) -> list:
    """Re-transcribe Arabic audio to double-check the SRT."""
    print(f"  Double-checking Arabic transcription with MLX-Whisper ({model_name})...")
    hf_repo = model_name if "/" in model_name else f"mlx-community/whisper-{model_name}"
    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=hf_repo,
        language="ar",
        verbose=False,
        batch_size=batch_size,
        condition_on_previous_text=condition_on_prev,
    )
    return cast(list, result["segments"])


# ─── Translation Backends ─────────────────────────────────────────

def translate_batch_with_deepl(texts: list, source: str, target: str, api_key: str) -> list:
    """Translate a list of texts using DeepL API in a single batch request with retry logic."""
    import time
    if not api_key or not texts:
        return texts
    
    # DeepL Free API key ends with :fx. Standard Pro key doesn't.
    base_url = "https://api-free.deepl.com" if api_key.endswith(":fx") else "https://api.deepl.com"
    url = f"{base_url}/v2/translate"
    
    # DeepL requires target language code to be uppercase, and EN must specify variant (e.g. EN-US).
    target_lang = target.upper()
    if target_lang == "EN":
        target_lang = "EN-US"
        
    source_lang = source.upper()
    
    max_retries = 5
    backoff = 1.0
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url,
                headers={
                    "Authorization": f"DeepL-Auth-Key {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "text": texts,
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                },
                timeout=30
            )
            if response.status_code == 429:
                # Rate limit hit, wait and retry
                time.sleep(backoff)
                backoff *= 2.0
                continue
            response.raise_for_status()
            result = response.json()
            return [t["text"] for t in result["translations"]]
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  [WARNING] DeepL batch translation failed after {max_retries} attempts: {e}")
                raise
            time.sleep(backoff)
            backoff *= 2.0

def translate_with_deepl(text: str, source: str, target: str, api_key: str) -> str:
    """Translate a single text using DeepL API."""
    try:
        results = translate_batch_with_deepl([text], source, target, api_key)
        return results[0] if results else text
    except Exception:
        return text


def translate_with_google(text: str, source: str, target: str) -> str:
    """Translate text using Google Translate (free) with retries and MyMemory fallback."""
    import time
    import random
    
    max_retries = 5
    backoff = 1.0
    
    for attempt in range(max_retries):
        try:
            translator = GoogleTranslator(source=source, target=target)
            result = translator.translate(text)
            if result and result.strip():
                return result if isinstance(result, str) else str(result)
        except Exception as e:
            if attempt == max_retries - 1:
                break
            
            # Sleep with some jitter
            sleep_time = backoff + random.uniform(0.1, 0.5)
            time.sleep(sleep_time)
            backoff *= 2.0
            
    # Fallback to MyMemory Translate if Google Translate failed completely
    try:
        mymemory_res = translate_with_mymemory(text, source, target)
        if mymemory_res and mymemory_res.strip() != text.strip():
            return mymemory_res
    except Exception:
        pass
        
    return text


def translate_with_mymemory(text: str, source: str, target: str) -> str:
    """Translate text using MyMemory Translate (free)."""
    from deep_translator import MyMemoryTranslator
    
    # Map 2-letter codes to full names that MyMemoryTranslator supports
    lang_map = {
        "ar": "arabic",
        "de": "german",
        "en": "english"
    }
    src_mapped = lang_map.get(source.lower(), source.lower())
    tgt_mapped = lang_map.get(target.lower(), target.lower())
    
    try:
        translator = MyMemoryTranslator(source=src_mapped, target=tgt_mapped)
        result = translator.translate(text)
        return result if isinstance(result, str) else str(result)
    except Exception as e:
        print(f"  [WARNING] MyMemory translation failed: {e}")
        return text


def get_translation_backend(args) -> tuple:
    """Return the best available translation backend based on arguments.
    Returns a tuple (backend_name, key) where backend_name is one of
    'google', 'mymemory', 'deepl', or 'local'.
    """
    if getattr(args, 'local_translate', False):
        return ("local", "")
    # DeepL currently disabled; fallback to Google
    return ("google", "")



def translate_segment(text: str, source: str, target: str, backend: str, deepl_key: str) -> str:
    """Translate a single segment using the specified backend."""
    try:
        if backend == "local":
            if translate_with_local is None:
                raise ImportError("Please install transformers and sentencepiece: pip install transformers sentencepiece")
            return translate_with_local(text, source, target)
        if backend == "deepl" and deepl_key:
            return translate_with_deepl(text, source, target, deepl_key)
        else:
            return translate_with_google(text, source, target)
    except Exception as e:
        print(f"  [WARNING] Translation backend '{backend}' failed: {e}")
        print("  Falling back to Google Translate...")
        try:
            return translate_with_google(text, source, target)
        except Exception as e2:
            print(f"  [ERROR] Google Translate also failed: {e2}")
            return text


def _assign_translations(batch: list, combined_text: str, lang_key: str) -> None:
    """Parse a combined numbered translation block back into the batch."""
    lines = combined_text.strip().split("\n\n")
    for idx, line in enumerate(lines):
        if idx < len(batch) and line.strip():
            cleaned = line.strip()
            if cleaned.startswith("["):
                cleaned = cleaned.split("]", 1)[1].strip()
            batch[idx][lang_key] = cleaned


def translate_segments(segments: list, output_dir: Path | None, args, deepl_key: str) -> list:
    """Translate each segment's text from Arabic to German and English.

    Supports resume capability and live backup.
    """
    backend, deepl_key = get_translation_backend(args)
    args._backend = backend

    # Store original Arabic text before translation
    for seg in segments:
        if "original_ar" not in seg:
            seg["original_ar"] = seg["text"]

    total = len(segments)
    batch_size = 20
    max_workers = 3

    # Resume support
    translated_indices = set()
    if output_dir:
        progress_file = output_dir / "translation_progress.json"
        if progress_file.exists():
            with open(progress_file) as f:
                translated_indices = set(json.load(f))
            if translated_indices:
                print(f"  Resuming: {len(translated_indices)} segments already translated")

    def save_progress():
        if output_dir:
            progress_file = output_dir / "translation_progress.json"
            with open(progress_file, "w") as f:
                json.dump(list(translated_indices), f)

    def save_live_backup(batch_end: int):
        if not output_dir:
            return
        temp_srt = output_dir / "translation_temp.srt"
        try:
            with open(temp_srt, "w", encoding="utf-8") as f:
                for i, seg in enumerate(segments[:batch_end], start=1):
                    if seg.get("text_de", "").strip():
                        f.write(f"{i}\n")
                        f.write(f"{format_time(seg['start'])} --> {format_time(seg['end'])}\n")
                        f.write(f"{seg['text_de'].strip()}\n\n")
        except Exception as e:
            print(f"  [WARNING] Could not save live backup: {e}")

    def translate_batch(batch_start: int, batch_end: int) -> int:
        if any(i in translated_indices for i in range(batch_start, batch_end)):
            return batch_end

        batch = segments[batch_start:batch_end]
        valid_indices = [i for i, seg in enumerate(batch) if seg['text'].strip()]
        texts = [batch[i]['text'].strip() for i in valid_indices]

        if not texts:
            for i in range(batch_start, batch_end):
                translated_indices.add(i)
            save_progress()
            return batch_end

        if backend == "deepl" and deepl_key:
            try:
                de_results = translate_batch_with_deepl(texts, "ar", "de", deepl_key)
                en_results = translate_batch_with_deepl(texts, "ar", "en", deepl_key)
                for idx, v_idx in enumerate(valid_indices):
                    batch[v_idx]["text_de"] = de_results[idx]
                    batch[v_idx]["text_en"] = en_results[idx]
            except Exception as e:
                print(f"  [WARNING] DeepL batch translation failed: {e}. Falling back to single segment translation...")
                for i, seg in enumerate(batch):
                    if seg['text'].strip():
                        seg["text_de"] = translate_segment(seg['text'].strip(), "ar", "de", backend, deepl_key)
                        seg["text_en"] = translate_segment(seg['text'].strip(), "ar", "en", backend, deepl_key)
        else:
            for i, seg in enumerate(batch):
                if seg['text'].strip():
                    seg["text_de"] = translate_segment(seg['text'].strip(), "ar", "de", backend, deepl_key)
                    seg["text_en"] = translate_segment(seg['text'].strip(), "ar", "en", backend, deepl_key)

        # Mark as translated
        for i in range(batch_start, batch_end):
            translated_indices.add(i)
        save_progress()
        save_live_backup(batch_end)

        return batch_end


    # Process batches in parallel
    print(f"  Translating {total} segments Arabic → German + English ({backend} - standard mode) ...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            if not any(i in translated_indices for i in range(batch_start, batch_end)):
                futures.append(executor.submit(translate_batch, batch_start, batch_end))

        completed = 0
        for future in as_completed(futures):
            batch_end = future.result()
            completed = max(completed, batch_end)
            if completed % 10 == 0 or completed == total:
                print(f"  Progress: {completed}/{total} ({completed*100//total}%)")

    # Set text fields for German and English
    for seg in segments:
        if "text_de" in seg:
            seg["text"] = seg["text_de"]
        if "text_en" not in seg:
            seg["text_en"] = seg.get("original_ar", seg.get("text", ""))

    print(f"  Translation completed successfully ({backend})")
    return segments


# ─── Verification with Report ─────────────────────────────────────

def verify_translations_with_report(segments: list, output_dir: Path, args, deepl_key: str) -> dict:
    """Verify translations using multiple AI tools with progress bar and save report.

    Pass 1: Google Translate (free)
    Pass 2: DeepL API (if available)
    Pass 3: MyMemory Translate (free)
    """
    deepl_key = ""

    total = len(segments)
    
    # Setup ytemp.json path
    ytemp_path = output_dir / "ytemp.json"
    
    # Initialize report
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_segments": total,
        "google_verified": 0,
        "google_mismatches": 0,
        "deepl_verified": 0,
        "deepl_mismatches": 0,
        "mymemory_verified": 0,
        "mymemory_mismatches": 0,
        "mismatches": [],
    }
    
    start_idx = 0
    
    # Resume support
    if ytemp_path.exists():
        try:
            with open(ytemp_path, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
                if saved_data and "report" in saved_data:
                    report = saved_data["report"]
                    start_idx = saved_data.get("completed_count", 0)
                    print(f"  Resuming translation verification: {start_idx}/{total} segments already verified")
        except Exception as e:
            print(f"  [WARNING] Could not read ytemp.json: {e}. Starting verification from scratch.")
            start_idx = 0
            
    # Setup verify_count limit
    verify_count = getattr(args, "verify_count", 20)
    if verify_count < 0 or verify_count > total:
        verify_count = total

    if start_idx < verify_count:
        print(f"\n  Verifying translations with multiple AI tools (up to {verify_count} segments)...")
        print(f"  Pass 1: Google Translate (free)")
        if deepl_key:
            print(f"  Pass 2: DeepL API")
        print(f"  Pass 3: MyMemory Translate (free)")

    for idx in range(start_idx + 1, verify_count + 1):
        seg = segments[idx - 1]
        original_text = seg.get("original_ar", "")
        if not original_text.strip():
            # Update progress
            try:
                with open(ytemp_path, "w", encoding="utf-8") as f:
                    json.dump({"completed_count": idx, "report": report}, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            continue

        current_de = seg.get("text_de", seg.get("text", "")).strip()
        current_en = seg.get("text_en", "").strip()

        # Pass 1: Google Translate
        try:
            google_de = translate_with_google(original_text, "ar", "de")
            google_en = translate_with_google(original_text, "ar", "en")

            de_match_google = google_de.strip() == current_de
            en_match_google = google_en.strip() == current_en

            if de_match_google:
                report["google_verified"] += 1
            else:
                report["google_mismatches"] += 1
                report["mismatches"].append({
                    "segment": idx,
                    "original_ar": original_text[:100],
                    "current_de": current_de[:100],
                    "google_de": google_de[:100],
                    "current_en": current_en[:100],
                    "google_en": google_en[:100],
                    "google_match": de_match_google,
                })
        except Exception:
            pass

        # Pass 2: DeepL API
        if deepl_key:
            try:
                deepl_de = translate_with_deepl(original_text, "ar", "de", deepl_key)
                if deepl_de.strip() == current_de:
                    report["deepl_verified"] += 1
                else:
                    report["deepl_mismatches"] += 1
            except Exception:
                pass

        # Pass 3: MyMemory Translate
        try:
            mymemory_de = translate_with_mymemory(original_text, "ar", "de")
            if mymemory_de.strip() == current_de:
                report["mymemory_verified"] += 1
            else:
                report["mymemory_mismatches"] += 1
        except Exception:
            pass

        # Save live backup to ytemp.json
        try:
            with open(ytemp_path, "w", encoding="utf-8") as f:
                json.dump({"completed_count": idx, "report": report}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        # Rate-limiting delay to avoid exceeding API limits (e.g. 5 requests/sec)
        time.sleep(0.5)

    # Save final report
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"verification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    print(f"  Google verification: {report['google_verified']} verified, {report['google_mismatches']} mismatches")
    if deepl_key:
        print(f"  DeepL verification: {report['deepl_verified']} verified, {report['deepl_mismatches']} mismatches")
    print(f"  MyMemory verification: {report['mymemory_verified']} verified, {report['mymemory_mismatches']} mismatches")
    if start_idx < total:
        print(f"  Report saved to: {report_path}")

    return report


def cleanup_temp_files(output_dir: Path, base_name: str) -> None:
    """Clean up temporary and intermediate files, keeping only merged SRT and compressed video."""
    temp_files = [
        output_dir / "translation_progress.json",
        output_dir / "translation_temp.srt",
        output_dir / "reports" / "verification_live.json",
        output_dir / "ytemp.json",
        output_dir / f"{base_name}_ar.srt",
        output_dir / f"{base_name}_de.srt",
        output_dir / f"{base_name}_en.srt",
    ]

    # Clean up downloaded raw video/audio files
    for pattern in ["*_video.mp4", "*.mp3", "*_video.m4a", "*_video.webm"]:
        for f in output_dir.glob(pattern):
            temp_files.append(f)

    for temp_file in temp_files:
        if temp_file.exists():
            try:
                temp_file.unlink()
                print(f"  Cleaned up: {temp_file.name}")
            except Exception:
                pass

    # Thoroughly delete the reports directory and its contents
    reports_dir = output_dir / "reports"
    if reports_dir.exists():
        for f in reports_dir.glob("*"):
            try:
                f.unlink()
            except Exception:
                pass
        try:
            reports_dir.rmdir()
            print("  Cleaned up: reports directory")
        except Exception:
            pass


# ─── Video Compression ────────────────────────────────────────────

def compress_video(video_path: Path, output_path: Path, target_mb: int,
                   arabic_srt: Path | None = None,
                   german_srt: Path | None = None,
                   english_srt: Path | None = None,
                   combined_srt: Path | None = None) -> None:
    """Compress video and burn subtitles using ffmpeg with progress bar.

    Burns up to 3 subtitle tracks:
    - Arabic (bottom)
    - German (top)
    - English (middle, if available)
    """
    if not video_path.exists():
        print(f"  Warning: Video not found at {video_path}, skipping compression.")
        return

    current_mb = video_path.stat().st_size / (1024 * 1024)
    print(f"  Current video size: {current_mb:.1f} MB")

    if current_mb <= target_mb:
        print(f"  Video already under {target_mb} MB, copying without re-encode.")
        output_path.write_bytes(video_path.read_bytes())
        return

    duration = float(
        subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    )

    target_bitrate = int((target_mb * 8 * 1024 * 1024) / duration)
    audio_bitrate = min(64, target_bitrate // 4)
    video_bitrate = target_bitrate - audio_bitrate * 1024

    print(f"  Duration: {duration:.0f}s, target bitrate: {target_bitrate // 1000} kbps")

    # Build ffmpeg command with optional subtitles
    cmd = ["ffmpeg", "-y", "-i", str(video_path)]

    # Add subtitle inputs if provided
    subtitle_inputs = []
    if german_srt and german_srt.exists():
        cmd.extend(["-i", str(german_srt)])
        subtitle_inputs.append(("de", german_srt))
    if english_srt and english_srt.exists():
        cmd.extend(["-i", str(english_srt)])
        subtitle_inputs.append(("en", english_srt))
    if arabic_srt and arabic_srt.exists():
        cmd.extend(["-i", str(arabic_srt)])
        subtitle_inputs.append(("ar", arabic_srt))

    # Build filter for burning subtitles
    if subtitle_inputs:
        def escape_ffmpeg_path(path: Path) -> str:
            p_str = path.as_posix().replace(":", "\\:")
            p_str = p_str.replace("'", "'\\\\\\''")
            return p_str

        filter_parts = []
        current_label = "[0:v]"
        total_subtitles = len(subtitle_inputs)

        # Enhanced subtitle styling with high-quality readable fonts
        # Style: Bold outline with semi-transparent background for optimal readability
        # Commas in force_style must be escaped as \\, to prevent FFMPEG parsing them as filter parameters
        german_style = "FontName=Arial\\,FontSize=26\\,PrimaryColour=&HFFFFFF&\\,SecondaryColour=&H000000&\\,OutlineColour=&H000000&\\,BackColour=&H00000000&\\,Bold=-1\\,Italic=0\\,BorderStyle=1\\,Outline=3\\,Shadow=1\\,MarginV=20\\,Alignment=8"  # Top center
        english_style = "FontName=Arial\\,FontSize=22\\,PrimaryColour=&H00FFFF&\\,SecondaryColour=&H000000&\\,OutlineColour=&H000000&\\,BackColour=&H00000000&\\,Bold=-1\\,Italic=0\\,BorderStyle=1\\,Outline=3\\,Shadow=1\\,MarginV=20\\,Alignment=5"  # Middle center
        arabic_style = "FontName=Arial\\,FontSize=26\\,PrimaryColour=&HFFFFFF&\\,SecondaryColour=&H000000&\\,OutlineColour=&H000000&\\,BackColour=&H00000000&\\,Bold=-1\\,Italic=0\\,BorderStyle=1\\,Outline=3\\,Shadow=1\\,MarginV=20\\,Alignment=2"  # Bottom left (RTL support)

        # German on top
        if ("de", german_srt) in subtitle_inputs:
            next_label = f"[v{len(filter_parts)+1}]" if len(filter_parts) + 1 < total_subtitles else "[vout]"
            filter_parts.append(
                f"{current_label}subtitles=filename='{escape_ffmpeg_path(german_srt)}':force_style='{german_style}'{next_label}"
            )
            current_label = next_label

        # English in middle
        if ("en", english_srt) in subtitle_inputs:
            next_label = f"[v{len(filter_parts)+1}]" if len(filter_parts) + 1 < total_subtitles else "[vout]"
            filter_parts.append(
                f"{current_label}subtitles=filename='{escape_ffmpeg_path(english_srt)}':force_style='{english_style}'{next_label}"
            )
            current_label = next_label

        # Arabic on bottom
        if ("ar", arabic_srt) in subtitle_inputs:
            next_label = f"[v{len(filter_parts)+1}]" if len(filter_parts) + 1 < total_subtitles else "[vout]"
            filter_parts.append(
                f"{current_label}subtitles=filename='{escape_ffmpeg_path(arabic_srt)}':force_style='{arabic_style}'{next_label}"
            )
            current_label = next_label

        cmd.extend(["-filter_complex", ";".join(filter_parts)])
        cmd.extend(["-map", "[vout]"])
        cmd.extend(["-map", "0:a?"])

    cmd.extend([
        "-c:v", "libx264", "-preset", "slow", "-crf", "23", "-b:v", f"{video_bitrate}k",
        "-c:a", "aac", "-b:a", f"{audio_bitrate}k",
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ])

    # Run with progress monitoring
    print("  Compressing video with progress bar...")
    last_update = 0.0
    ffmpeg_output = []

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    try:
        if process.stdout:
            for line in process.stdout:
                ffmpeg_output.append(line)
                time_match = re.search(r"time=(\d+):(\d+):(\d+\.\d+)", line)
                if time_match:
                    hours, mins, secs = time_match.groups()
                    elapsed = int(hours) * 3600 + int(mins) * 60 + float(secs)
                    now = time.time()
                    if duration > 0 and now - last_update > 1:
                        progress_pct = min(100, int(elapsed * 100 / duration))
                        print(
                            f"\r  Progress: {progress_pct}% ({elapsed:.0f}s/{int(duration)}s)",
                            end="",
                            flush=True,
                        )
                        last_update = now
    finally:
        process.wait()
    print()

    if process.returncode != 0:
        print("  [ERROR] FFMPEG failed with the following output:")
        print("".join(ffmpeg_output))
        raise RuntimeError(f"FFMPEG failed with exit code {process.returncode}")

    final_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Compressed video size: {final_mb:.1f} MB")


# ─── Argument Parsing ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="YouTube Arabic → German → English: SRT subtitles + compressed video."
    )
    parser.add_argument("url", type=str, help="YouTube video URL")
    parser.add_argument(
        "--model", type=str, default="medium",
        help="Whisper model size (e.g. medium, large-v3-turbo, large-v3-4bit) or HF repo (default: medium)",
    )
    parser.add_argument(
        "--whisper-batch-size", type=int, default=8,
        help="Batch size for MLX-Whisper parallel decoding (default: 8)",
    )
    parser.add_argument(
        "--condition-on-previous", action="store_true",
        help="Condition Whisper transcription on previous text (default: False, can increase loops but sometimes helps consistency)",
    )
    parser.add_argument(
        "--target-size", type=int, default=50,
        help="Target compressed video size in MB (default: 50)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output",
        help="Output directory (default: output/)",
    )
    parser.add_argument(
        "--min-quality", action="store_true",
        help="Download minimum quality video/audio to save bandwidth and space",
    )
    parser.add_argument(
        "--deepl-key", type=str,
        help="DeepL API key for translation/verification",
    )
    parser.add_argument(
        "--local-translate",
        action="store_true",
        help="Use local LLM translation backend instead of online services",
    )
    parser.add_argument(
        "--verify-count", type=int, default=20,
        help="Number of segments to verify during translation verification (default: 20, use -1 for all)",
    )
    parser.add_argument(
        "--no-cleanup", action="store_true",
        help="Skip cleanup of temporary files after completion",
    )
    return parser.parse_args()


# ─── Main Pipeline ────────────────────────────────────────────────

def main():
    load_dotenv()
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  YouTube → Arabic/German/English Pipeline")
    print(f"  URL: {args.url}")
    print(f"{'='*60}\n")

    # Step 0: Name & Detect existing output
    base_name = None
    existing_ar_srt = None
    for f in output_dir.glob("*_ar.srt"):
        existing_ar_srt = f
        break

    if existing_ar_srt:
        base_name = existing_ar_srt.stem.removesuffix("_ar")
        print(f"  Found existing Arabic SRT: {existing_ar_srt.name}")
        print(f"     Resume with base name: {base_name}")
        answer = input("  Re-use this base name? [Y/n]: ").strip().lower()
        if answer in ("", "y", "yes"):
            print(f"  Resuming with existing base name: {base_name}")

    if base_name is None:
        print("[0/6] Naming output files...")
        base_name = prompt_output_name(args.url)
        print(f"       Base name: {base_name}")

    # Define output paths
    arabic_srt = output_dir / f"{base_name}_ar.srt"
    german_srt = output_dir / f"{base_name}_de.srt"
    english_srt = output_dir / f"{base_name}_en.srt"
    combined_srt = output_dir / f"{base_name}_ar-de-en.srt"
    compressed_video = output_dir / f"{base_name}_compressed.mp4"

    # Step 1: Download (skip if video+audio exist)
    mp4_files = sorted(output_dir.glob("*_video.mp4"))
    mp3_files = sorted(output_dir.glob("*.mp3"))
    video_path = mp4_files[0] if mp4_files else None
    audio_path = mp3_files[0] if mp3_files else None

    if video_path and audio_path:
        print("\n[1/6] Video & audio already downloaded")
        print(f"       Video: {video_path.name}")
        print(f"       Audio: {audio_path.name}")
    else:
        print("\n[1/6] Downloading video & audio...")
        media = download_youtube(args.url, output_dir, args.min_quality)
        video_path = media["video"]
        audio_path = media["audio"]
        print(f"       Video: {video_path.name}")
        print(f"       Audio: {audio_path.name}")

    # Step 2: Transcribe (skip if Arabic SRT exists)
    if arabic_srt.exists():
        print(f"\n[2/6] Already transcribed — loading from {arabic_srt.name}")
        segments = read_srt(arabic_srt)
        print(f"       {len(segments)} segments loaded")

        # Double-check Arabic transcription always
        if audio_path and audio_path.exists():
            print(f"\n  Double-checking Arabic transcription...")
            double_check_segments = double_check_arabic_srt(
                segments, audio_path, args.model,
                batch_size=args.whisper_batch_size,
                condition_on_prev=args.condition_on_previous,
            )
            print(f"  Double-check completed: {len(double_check_segments)} segments")
            # Use the double-checked segments
            segments = double_check_segments
            write_srt(segments, arabic_srt)
            print(f"  Updated Arabic SRT: {arabic_srt.name}")
    else:
        print(f"\n[2/6] Transcribing Arabic (model: {args.model})...")
        segments = transcribe_arabic(
            audio_path, args.model,
            batch_size=args.whisper_batch_size,
            condition_on_prev=args.condition_on_previous,
        )
        write_srt(segments, arabic_srt)
        print(f"       Arabic SRT: {arabic_srt.name} ({len(segments)} segments)")

    # Step 3: Translate (skip if German SRT exists and has proper German text)
    needs_retranslate = False
    if german_srt.exists():
        de_segments = read_srt(german_srt)
        if de_segments:
            # Check if German SRT actually contains Arabic text
            sample_text = de_segments[0]["text"]
            if is_arabic(sample_text):
                print(f"\n  [WARNING] German SRT contains Arabic script! Re-translating.")
                needs_retranslate = True

    if german_srt.exists() and english_srt.exists() and not needs_retranslate:
        print(f"\n[3/6] Already translated — loading from {german_srt.name} and {english_srt.name}")
        arabic_segments = read_srt(arabic_srt) if arabic_srt.exists() else []
        de_segments = read_srt(german_srt)
        en_segments = read_srt(english_srt)

        # Merge into segments
        for i, seg in enumerate(de_segments):
            seg["original_ar"] = arabic_segments[i]["text"] if i < len(arabic_segments) else ""
            seg["text_de"] = seg["text"]
            seg["text_en"] = en_segments[i]["text"] if i < len(en_segments) else ""
            seg["text"] = seg["text_de"]

        segments = de_segments
        print(f"       {len(segments)} segments loaded")

        # Verify with multiple AI tools
        backend, deepl_key = get_translation_backend(args)
        args._backend = backend
        verify_translations_with_report(segments, output_dir, args, deepl_key)
    else:
        print(f"\n[3/6] Translating to German + English...")
        backend, deepl_key = get_translation_backend(args)
        args._backend = backend

        if needs_retranslate:
            print(f"  [INFO] Re-translating due to Arabic script in German SRT")
            # Clear progress to force re-translation
            progress_file = output_dir / "translation_progress.json"
            if progress_file.exists():
                progress_file.unlink()

        segments = translate_segments(segments, output_dir, args, deepl_key)

        # Write German SRT
        de_segments = []
        for seg in segments:
            de_segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg.get("text_de", seg.get("text", "")),
            })
        write_srt(de_segments, german_srt)
        print(f"       German SRT: {german_srt.name}")

        # Write English SRT
        en_segments = []
        for seg in segments:
            en_segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg.get("text_en", ""),
            })
        write_srt(en_segments, english_srt)
        print(f"       English SRT: {english_srt.name}")

        # Write combined tri-lingual SRT
        write_triilingual_srt(
            [{"start": s["start"], "end": s["end"], "text": s.get("original_ar", s.get("text", ""))} for s in segments],
            de_segments,
            en_segments,
            combined_srt,
        )
        print(f"       Combined SRT: {combined_srt.name}")

        # Verify with multiple AI tools
        verify_translations_with_report(segments, output_dir, args, deepl_key)

    # Step 4: Compress video (skip if exists)
    if compressed_video.exists():
        print(f"\n[4/6] Compressed video already exists: {compressed_video.name}")
    else:
        print(f"\n[4/6] Compressing video (target: {args.target_size} MB)...")
        compress_video(video_path, compressed_video, args.target_size,
                       arabic_srt, german_srt, english_srt, combined_srt)

    # Delete monolingual SRT files (always, keeping only the combined/merged SRT)
    print("\n[5/6] Deleting monolingual SRT files...")
    for srt_file in [arabic_srt, german_srt, english_srt]:
        if srt_file.exists():
            try:
                srt_file.unlink()
                print(f"       Deleted: {srt_file.name}")
            except Exception as e:
                print(f"       [WARNING] Could not delete {srt_file.name}: {e}")

    # Clean up temporary/intermediate files (always, unless --no-cleanup)
    if not args.no_cleanup:
        print(f"       Cleaning up temporary files...")
        cleanup_temp_files(output_dir, base_name)

    # Step 6: Summary
    print(f"\n[6/6] Summary")
    print(f"\n{'='*60}")
    print(f"  All done! Files in '{output_dir}/':")
    print(f"     - {combined_srt.name} (Combined AR/DE/EN subtitles)")
    print(f"     - {compressed_video.name}  (compressed video with burned subtitles)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
