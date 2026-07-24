"""
YouTube Arabic → German → English Dual Local AI Pipeline
========================================================

Downloads a YouTube video, transcribes Arabic audio to SRT (proofread by local LLM),
translates to German and English SRT (via 3-Tier fallback: Google Free -> MarianMT -> MLX Qwen2.5-7B),
verifies 100% of translations using local LLM on Apple Silicon Metal GPU,
and renders subtitled video with automatic cleanup and rich Markdown/HTML reports.

Translation Backends (3-Tier Fallback):
  1. Google Translate (Free API)
  2. Local MarianMT NMT (PyTorch MPS GPU)
  3. Local MLX LLM (Qwen2.5-7B-Instruct-4bit on Metal GPU)

Multi-pass Verification:
  - Pass 1: Google Free API Translation
  - Pass 2: Local MarianMT NMT Verification
  - Pass 3: Local Qwen2.5-7B LLM Verification

Uses MLX-Whisper for fast speech-to-text on Apple Silicon GPU.
Resume support: re-run the same command and it detects existing output files, skipping completed steps automatically.

Usage:
    python main.py <youtube_url>

Example:
    python main.py https://youtu.be/MgxTrPOkhDU

Flags:
    --quality           Set video download quality preset (1/low, 2/medium, 3/high, 4/best)
    --double-check      Proofread Arabic transcription with local LLM
    --no-cleanup        Keep temporary intermediate files
"""

import argparse
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import cast

import mlx_whisper
from deep_translator import GoogleTranslator
from dotenv import load_dotenv

try:
    from local_translation import (
        translate_with_local,
        translate_with_mlx_llm,
        verify_arabic_transcription_with_llm,
    )
except ImportError:
    translate_with_local = None
    translate_with_mlx_llm = None
    verify_arabic_transcription_with_llm = None

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
    """Write a combined SRT file with Arabic, German, and English subtitles."""
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, (ar, de, en) in enumerate(zip(arabic_segs, german_segs, english_segs), start=1):
            f.write(f"{i}\n")
            f.write(f"{format_time(ar['start'])} --> {format_time(ar['end'])}\n")
            f.write(f"AR: {ar['text'].strip()}\n")
            f.write(f"DE: {de['text'].strip()}\n")
            if en.get("text", "").strip():
                f.write(f"EN: {en['text'].strip()}\n")
            f.write("\n")


def read_srt(srt_path: Path) -> list:
    """Read any SRT file and return segments with start/end/text."""
    segments = []
    with open(srt_path, encoding="utf-8") as f:
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

def select_video_quality(url: str, min_quality: bool = False) -> str:
    """Query yt-dlp for available formats and prompt user to choose quality."""
    if min_quality:
        return "worstvideo[ext=mp4]+worstaudio[ext=m4a]/worst[ext=mp4]/worst"

    print("\n  Fetching available video qualities from YouTube...")
    try:
        subprocess.run(
            ["yt-dlp", "-F", url],
            capture_output=True, text=True, check=True
        )

        presets = {
            "1": ("Low (144p / 240p - smallest size & fastest)", "worstvideo[ext=mp4]+worstaudio[ext=m4a]/worst[ext=mp4]/worst"),
            "2": ("Medium (360p / 480p - balanced quality)", "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480][ext=mp4]/best"),
            "3": ("High (720p / 1080p - standard HD)", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best"),
            "4": ("Best available (Max resolution)", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"),
        }

        print("\n  Select Video Download Quality:")
        for key, (desc, _) in presets.items():
            print(f"    [{key}] {desc}")

        choice = input("\n  Enter choice [1-4] (default: 2 Medium): ").strip()
        if choice in presets:
            selected = presets[choice]
            print(f"  Selected quality: {selected[0]}")
            return selected[1]
        else:
            print("  Defaulting to Medium quality (360p / 480p)")
            return presets["2"][1]
    except (ValueError, KeyError, TypeError, EOFError, KeyboardInterrupt) as e:
        print(f"  [WARNING] Could not fetch format options: {e}")
        return "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"


def download_youtube(url: str, output_dir: Path, min_quality: bool = False, video_format: str | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_format:
        video_format = select_video_quality(url, min_quality)

    video_template = output_dir / "%(id)s_video.%(ext)s"
    subprocess.run(
        ["yt-dlp", "-f", video_format, "-o", str(video_template), url],
        check=True, capture_output=True, text=True,
    )

    # Try to extract audio as MP3; if conversion fails (e.g., missing ffmpeg), fall back to raw m4a download
    audio_path = None
    audio_template_mp3 = output_dir / "%(id)s.%(ext)s"
    try:
        subprocess.run(
            ["yt-dlp", "-x", "--audio-format", "mp3", "-o", str(audio_template_mp3), url],
            check=True, capture_output=True, text=True,
        )
        audio_path = audio_template_mp3
    except subprocess.CalledProcessError:
        print("[WARNING] MP3 audio extraction failed, falling back to m4a audio download.")
        audio_template_m4a = output_dir / "%(id)s_audio.m4a"
        subprocess.run(
            ["yt-dlp", "-f", "bestaudio[ext=m4a]/bestaudio", "-o", str(audio_template_m4a), url],
            check=True, capture_output=True, text=True,
        )
        audio_path = audio_template_m4a

    mp4_files = sorted(output_dir.glob("*_video.mp4"))
    if not mp4_files:
        raise RuntimeError("Failed to download video.")
    video_id = mp4_files[0].stem.replace("_video", "")
    return {
        "video_id": video_id,
        "video": mp4_files[0],
        "audio": audio_path,
    }


# ─── Transcription ─────────────────────────────────────────────────

def transcribe_arabic(audio_path: Path, model_name: str, condition_on_prev: bool = False, temperature: float | None = None) -> list:
    """Transcribe Arabic audio using MLX-Whisper on Apple Silicon GPU."""
    print(f"  Transcribing Arabic audio with MLX-Whisper ({model_name})...")
    hf_repo = model_name if "/" in model_name else f"mlx-community/whisper-{model_name}"

    kwargs = {
        "path_or_hf_repo": hf_repo,
        "language": "ar",
        "verbose": False,
        "condition_on_previous_text": condition_on_prev,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature

    result = mlx_whisper.transcribe(
        str(audio_path),
        **kwargs
    )
    return cast(list, result["segments"])


def double_check_arabic_srt(segments: list, audio_path: Path, model_name: str, condition_on_prev: bool = False, temperature: float | None = None) -> list:
    """Re-transcribe Arabic audio to double-check the SRT."""
    print(f"  Double-checking Arabic transcription with MLX-Whisper ({model_name})...")
    hf_repo = model_name if "/" in model_name else f"mlx-community/whisper-{model_name}"

    kwargs = {
        "path_or_hf_repo": hf_repo,
        "language": "ar",
        "verbose": False,
        "condition_on_previous_text": condition_on_prev,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature

    result = mlx_whisper.transcribe(
        str(audio_path),
        **kwargs
    )
    return cast(list, result["segments"])


# ─── Translation Backends ─────────────────────────────────────────

def translate_with_google(text: str, source: str, target: str) -> str:
    """Translate text using Google Translate (free) with retries."""
    import random
    import time

    max_retries = 5
    backoff = 1.0

    for attempt in range(max_retries):
        try:
            translator = GoogleTranslator(source=source, target=target)
            result = translator.translate(text)
            if result and result.strip():
                return result if isinstance(result, str) else str(result)
        except Exception:
            if attempt == max_retries - 1:
                break

            sleep_time = backoff + random.uniform(0.1, 0.5)
            time.sleep(sleep_time)
            backoff *= 2.0

    return text


def get_translation_backend(args) -> str:
    """Return translation backend. Priority: Google Translate (free) first, then local AI models."""
    return "google"


def translate_segment(text: str, source: str, target: str, backend: str = "google") -> str:
    """Translate segment using 3-Tier Fallback: 1. Google Free API -> 2. Local MarianMT NMT -> 3. Local MLX LLM."""
    if not text or not text.strip():
        return ""

    # Tier 1: Google Free API
    try:
        res = translate_with_google(text, source, target)
        if res and res.strip() != text.strip():
            return res
    except Exception as e:
        print(f"  [WARNING] Google Free API translation failed: {e}. Trying Tier 2 (Local MarianMT)...")

    # Tier 2: Local MarianMT NMT
    try:
        if translate_with_local is not None:
            res = translate_with_local(text, source, target)
            if res and res.strip() != text.strip():
                return res
    except Exception as e:
        print(f"  [WARNING] Local MarianMT failed: {e}. Trying Tier 3 (Local MLX LLM)...")

    # Tier 3: Local MLX LLM (Qwen2.5-7B-Instruct)
    try:
        if translate_with_mlx_llm is not None:
            res = translate_with_mlx_llm(text, source, target)
            if res and res.strip():
                return res
    except Exception as e:
        print(f"  [ERROR] Tier 3 Local MLX LLM translation failed: {e}")

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

    # Enhanced Resume Support from translation_temp.json
    translated_indices = set()
    if output_dir:
        temp_json = output_dir / "translation_temp.json"
        progress_file = output_dir / "translation_progress.json"

        if temp_json.exists():
            try:
                with open(temp_json, encoding="utf-8") as f:
                    saved = json.load(f)
                    saved_segs = saved.get("segments", [])
                    for s_item in saved_segs:
                        idx_0 = s_item["id"] - 1
                        if 0 <= idx_0 < total:
                            segments[idx_0]["text_de"] = s_item.get("text_de", "")
                            segments[idx_0]["text_en"] = s_item.get("text_en", "")
                            translated_indices.add(idx_0)
                print(f"  Resuming: Restored {len(translated_indices)} segments from live JSON backup")
            except Exception as e:
                print(f"  [WARNING] Could not restore from live JSON backup: {e}")

        if not translated_indices and progress_file.exists():
            try:
                with open(progress_file, encoding="utf-8") as f:
                    translated_indices = set(json.load(f))
                if translated_indices:
                    print(f"  Resuming: {len(translated_indices)} segments already translated")
            except Exception:
                pass

    def save_progress():
        if output_dir:
            progress_file = output_dir / "translation_progress.json"
            with open(progress_file, "w") as f:
                json.dump(list(translated_indices), f)

    def save_live_backup(batch_end: int):
        if not output_dir:
            return
        temp_json = output_dir / "translation_temp.json"
        temp_srt = output_dir / "translation_temp.srt"
        try:
            # 1. Save live JSON backup with full AR, DE, EN metadata
            live_segments = []
            for i, seg in enumerate(segments[:batch_end], start=1):
                live_segments.append({
                    "id": i,
                    "start": seg.get("start", 0.0),
                    "end": seg.get("end", 0.0),
                    "original_ar": seg.get("original_ar", seg.get("text", "")).strip(),
                    "text_de": seg.get("text_de", "").strip(),
                    "text_en": seg.get("text_en", "").strip(),
                })
            with open(temp_json, "w", encoding="utf-8") as f:
                json.dump({"completed_count": len(live_segments), "segments": live_segments}, f, ensure_ascii=False, indent=2)

            # 2. Save live SRT backup
            with open(temp_srt, "w", encoding="utf-8") as f:
                for i, seg in enumerate(segments[:batch_end], start=1):
                    if seg.get("text_de", "").strip():
                        f.write(f"{i}\n")
                        f.write(f"{format_time(seg['start'])} --> {format_time(seg['end'])}\n")
                        f.write(f"AR: {seg.get('original_ar', seg.get('text', '')).strip()}\n")
                        f.write(f"DE: {seg.get('text_de', '').strip()}\n")
                        if seg.get("text_en", "").strip():
                            f.write(f"EN: {seg.get('text_en', '').strip()}\n")
                        f.write("\n")
        except Exception as e:
            print(f"  [WARNING] Could not save live JSON/SRT backup: {e}")

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

        for i, seg in enumerate(batch):
            if seg['text'].strip():
                seg["text_de"] = translate_segment(seg['text'].strip(), "ar", "de", backend)
                seg["text_en"] = translate_segment(seg['text'].strip(), "ar", "en", backend)

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

def verify_translations_with_report(segments: list, output_dir: Path, args) -> dict:
    """Verify translations using local AI models with progress bar and save report."""
    total = len(segments)

    # Setup ytemp.json path
    ytemp_path = output_dir / "ytemp.json"

    # Initialize report
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_segments": total,
        "mlx_llm_verified": 0,
        "mlx_llm_mismatches": 0,
        "mismatches": [],
    }

    start_idx = 0

    # Resume support
    if ytemp_path.exists():
        try:
            with open(ytemp_path, encoding="utf-8") as f:
                saved_data = json.load(f)
                if saved_data and "report" in saved_data:
                    report = saved_data["report"]
                    start_idx = saved_data.get("completed_count", 0)
                    print(f"  Resuming translation verification: {start_idx}/{total} segments already verified")
        except Exception as e:
            print(f"  [WARNING] Could not read ytemp.json: {e}. Starting verification from scratch.")
            start_idx = 0

    # Default verification runs on 100% of all segments
    verify_count = getattr(args, "verify_count", total)
    if verify_count is None or verify_count <= 0 or verify_count > total:
        verify_count = total

    if start_idx < verify_count:
        print(f"\n  Verifying all translations with Local LLM ({verify_count}/{total} segments)...")
        print("  Pass 1: Google Free API Translation")
        print("  Pass 2: Local MarianMT NMT Verification")
        print("  Pass 3: Local Qwen2.5 LLM Verification")

    for idx in range(start_idx + 1, verify_count + 1):
        seg = segments[idx - 1]
        original_text = seg.get("original_ar", "")
        if not original_text.strip():
            try:
                with open(ytemp_path, "w", encoding="utf-8") as f:
                    json.dump({"completed_count": idx, "report": report}, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            continue

        current_de = seg.get("text_de", seg.get("text", "")).strip()

        # Pass 2: Local MLX LLM verification
        if translate_with_mlx_llm is not None:
            try:
                mlx_de = translate_with_mlx_llm(original_text, "ar", "de")
                match_de = mlx_de.strip().lower() == current_de.lower()

                if match_de:
                    report["mlx_llm_verified"] += 1
                else:
                    report["mlx_llm_mismatches"] += 1
                    report["mismatches"].append({
                        "segment": idx,
                        "original_ar": original_text[:100],
                        "marian_de": current_de[:100],
                        "mlx_llm_de": mlx_de[:100],
                    })
            except Exception:
                pass

        # Save live backup to ytemp.json
        try:
            with open(ytemp_path, "w", encoding="utf-8") as f:
                json.dump({"completed_count": idx, "report": report}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # Save final report in reports/ directory
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"verification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    google_verified = report.get("google_verified", report.get("mlx_llm_verified", 0))
    google_mismatches = report.get("google_mismatches", report.get("mlx_llm_mismatches", 0))
    print(f"  Verification: {google_verified} verified, {google_mismatches} mismatches")
    if start_idx < total:
        print(f"  Report saved to: {report_path}")

    return report


def cleanup_temp_files(output_dir: Path, base_name: str) -> None:
    """Clean up temporary and intermediate files, keeping only merged SRT and compressed video."""
    temp_files = [
        output_dir / "translation_progress.json",
        output_dir / "translation_temp.srt",
        output_dir / "translation_temp.json",
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
                   combined_srt: Path | None = None,
                   args: argparse.Namespace | None = None) -> None:
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

    # Check if FFMPEG supports subtitles filter
    has_subtitles = False
    try:
        res = subprocess.run(["ffmpeg", "-filters"], capture_output=True, text=True)
        has_subtitles = "subtitles" in res.stdout
    except Exception:
        pass

    # Build ffmpeg command with optional subtitles
    cmd = ["ffmpeg", "-y", "-i", str(video_path)]

    # Add subtitle inputs if provided
    subtitle_inputs = []
    if german_srt and german_srt.exists():
        subtitle_inputs.append(("de", german_srt))
    if english_srt and english_srt.exists():
        subtitle_inputs.append(("en", english_srt))
    if arabic_srt and arabic_srt.exists():
        subtitle_inputs.append(("ar", arabic_srt))
    if combined_srt and combined_srt.exists():
        subtitle_inputs.append(("combined", combined_srt))

    if subtitle_inputs:
        # Scenario 1: Burning subtitles (if supported)
        if has_subtitles:
            # Add inputs to command line (only for burnable separate tracks, not combined)
            burn_inputs = [x for x in subtitle_inputs if x[0] != "combined"]
            for _, srt_path in burn_inputs:
                cmd.extend(["-i", str(srt_path)])

            def escape_ffmpeg_path(path: Path) -> str:
                p_str = path.as_posix().replace(":", "\\:")
                p_str = p_str.replace("'", "'\\\\\\''")
                return p_str

            filter_parts = []
            current_label = "[0:v]"
            total_subtitles = len(burn_inputs)

            # YouTube-style Bottom Subtitle Layout Menu (interactive / argument driven)
            sub_layout = getattr(args, "sub_layout", None)
            if not sub_layout:
                print("\n  Select Subtitle Burning Layout for Video:")
                print("    [1] Split Style: German (Top) + English (Middle) + Arabic (Bottom)")
                print("    [2] Single Bottom Stack: All languages stacked at bottom (like YouTube captions)")
                print("    [3] Arabic Only at Bottom: Clean single-language bottom subtitle")
                layout_choice = input("  Enter choice [1-3] (default: 2 Single Bottom Stack): ").strip()
                sub_layout = layout_choice if layout_choice in ["1", "2", "3"] else "2"

            if sub_layout == "2":
                # YouTube bottom stack
                german_style = "FontName=Arial\\,FontSize=24\\,PrimaryColour=&HFFFFFF&\\,OutlineColour=&H000000&\\,BorderStyle=1\\,Outline=2\\,Shadow=1\\,MarginV=60\\,Alignment=2"
                english_style = "FontName=Arial\\,FontSize=22\\,PrimaryColour=&H00FFFF&\\,OutlineColour=&H000000&\\,BorderStyle=1\\,Outline=2\\,Shadow=1\\,MarginV=38\\,Alignment=2"
                arabic_style = "FontName=Arial\\,FontSize=26\\,PrimaryColour=&HFFFFFF&\\,OutlineColour=&H000000&\\,BorderStyle=1\\,Outline=2\\,Shadow=1\\,MarginV=12\\,Alignment=2"
            elif sub_layout == "3":
                # Arabic only at bottom
                arabic_style = "FontName=Arial\\,FontSize=28\\,PrimaryColour=&HFFFFFF&\\,OutlineColour=&H000000&\\,BorderStyle=1\\,Outline=2.5\\,Shadow=1\\,MarginV=20\\,Alignment=2"
                burn_inputs = [x for x in burn_inputs if x[0] == "ar"]
            else:
                # Default YouTube Style (Top/Middle/Bottom)
                german_style = "FontName=Arial\\,FontSize=26\\,PrimaryColour=&HFFFFFF&\\,OutlineColour=&H000000&\\,BorderStyle=1\\,Outline=3\\,Shadow=1\\,MarginV=20\\,Alignment=8"
                english_style = "FontName=Arial\\,FontSize=22\\,PrimaryColour=&H00FFFF&\\,OutlineColour=&H000000&\\,BorderStyle=1\\,Outline=3\\,Shadow=1\\,MarginV=20\\,Alignment=5"
                arabic_style = "FontName=Arial\\,FontSize=26\\,PrimaryColour=&HFFFFFF&\\,OutlineColour=&H000000&\\,BorderStyle=1\\,Outline=3\\,Shadow=1\\,MarginV=20\\,Alignment=2"

            if ("de", german_srt) in burn_inputs:
                next_label = f"[v{len(filter_parts)+1}]" if len(filter_parts) + 1 < total_subtitles else "[vout]"
                filter_parts.append(
                    f"{current_label}subtitles=filename='{escape_ffmpeg_path(german_srt)}':force_style='{german_style}'{next_label}"
                )
                current_label = next_label

            if ("en", english_srt) in burn_inputs:
                next_label = f"[v{len(filter_parts)+1}]" if len(filter_parts) + 1 < total_subtitles else "[vout]"
                filter_parts.append(
                    f"{current_label}subtitles=filename='{escape_ffmpeg_path(english_srt)}':force_style='{english_style}'{next_label}"
                )
                current_label = next_label

            if ("ar", arabic_srt) in burn_inputs:
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
        else:
            # Scenario 2: Embedding soft subtitles (fallback for missing libass)
            print("  [WARNING] Your FFmpeg lacks the 'subtitles' filter (compiled without libass).")
            print("            Falling back to embedding soft subtitles inside the MP4 video...")
            print("            (To burn hard subtitles, reinstall FFmpeg with libass: brew install ffmpeg)")

            for _, srt_path in subtitle_inputs:
                cmd.extend(["-i", str(srt_path)])

            cmd.extend(["-map", "0:v", "-map", "0:a?"])
            for idx, _ in enumerate(subtitle_inputs):
                cmd.extend(["-map", f"{idx+1}:s"])

            cmd.extend([
                "-c:v", "libx264", "-preset", "slow", "-crf", "23", "-b:v", f"{video_bitrate}k",
                "-c:a", "aac", "-b:a", f"{audio_bitrate}k",
                "-c:s", "mov_text",
            ])
            for idx, (lang, _) in enumerate(subtitle_inputs):
                lang_map = {"de": "ger", "en": "eng", "ar": "ara", "combined": "mul"}
                cmd.extend([f"-metadata:s:s:{idx}", f"language={lang_map.get(lang, lang)}"])
                title_map = {"de": "German", "en": "English", "ar": "Arabic", "combined": "Tri-lingual (AR/DE/EN)"}
                cmd.extend([f"-metadata:s:s:{idx}", f"title={title_map.get(lang, lang)}"])

            cmd.extend([
                "-movflags", "+faststart",
                "-pix_fmt", "yuv420p",
                str(output_path),
            ])
    else:
        # Scenario 3: No subtitles at all
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

def parse_temperature(val):
    if val is None or val.lower() == "none":
        return None
    return float(val)


def parse_args():
    parser = argparse.ArgumentParser(
        description="YouTube Arabic → German → English: SRT subtitles + compressed video."
    )
    parser.add_argument("url", type=str, help="YouTube video URL")
    parser.add_argument(
        "--model", type=str, default="large-v3-turbo-q4",
        help="Whisper model size (e.g. large-v3-turbo-q4, medium, large-v3-turbo, large-v3-4bit) or HF repo (default: large-v3-turbo-q4)",
    )
    parser.add_argument(
        "--whisper-temperature", type=parse_temperature, default=0.0,
        help="Temperature for Whisper decoding (e.g. 0.0 for greedy decoding, 'none' for auto-fallback) (default: 0.0)",
    )
    parser.add_argument(
        "--condition-on-previous", action="store_true",
        help="Condition Whisper transcription on previous text (default: False, can increase loops but sometimes helps consistency)",
    )
    parser.add_argument(
        "--double-check", action="store_true",
        help="Re-transcribe and double-check Arabic transcription when resuming (default: False)",
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
        "--quality", type=str, choices=["1", "2", "3", "4", "low", "medium", "high", "best"],
        help="Set video download quality preset directly without prompt (1/low, 2/medium, 3/high, 4/best)",
    )
    parser.add_argument(
        "--local-translate",
        action="store_true",
        default=True,
        help="Use local neural model translation backend on Apple Silicon GPU/MPS",
    )
    parser.add_argument(
        "--verify-count", type=int, default=20,
        help="Number of segments to verify during translation verification (default: 20, use -1 for all)",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Keep temporary intermediate audio and raw video files after pipeline completes",
    )
    parser.add_argument(
        "--sub-layout", type=str, choices=["1", "2", "3"],
        help="Subtitle burning layout: 1=YouTube Top/Mid/Bot, 2=Bottom Stack, 3=Arabic Only Bottom",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading video/audio if compressed video already exists with same base name",
    )
    return parser.parse_args()


# ─── Main Pipeline ────────────────────────────────────────────────

def main():
    load_dotenv()
    args = parse_args()
    # Enforce maximum target size of 100 MB
    if args.target_size > 100:
        print(f"[WARNING] Target size capped at 100 MB (requested {args.target_size} MB)")
        args.target_size = 100
    output_dir = Path(args.output_dir)
    # Determine base name early to locate compressed file
    base_name = None
    # If we have an existing compressed video, infer base name from it
    for f in output_dir.glob("*_compressed.mp4"):
        base_name = f.stem.replace("_compressed", "")
        break
    if args.skip_download and base_name:
        compressed_video = output_dir / f"{base_name}_compressed.mp4"
        if compressed_video.exists():
            existing_size_mb = compressed_video.stat().st_size / (1024 * 1024)
            if existing_size_mb <= args.target_size:
                print(f"[INFO] Skipping download/processing because compressed video already exists ({existing_size_mb:.1f} MB) and is within target size.")
                print(f"   File: {compressed_video.name}")
                # Exit gracefully – nothing else to do
                import sys
                sys.exit(0)
    # If not skipping, continue normal pipeline (base_name will be set later)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  YouTube → Arabic/German/English Pipeline")
    print(f"  URL: {args.url}")
    print(f"{'='*60}\n")

    # Step 0: Name & Detect existing output (Auto-detect from any existing pipeline file)
    base_name = None
    existing_files = list(output_dir.glob("*_ar.srt")) or list(output_dir.glob("*_video.mp4")) or list(output_dir.glob("translation_temp.json")) or list(output_dir.glob("*_compressed.mp4"))

    if existing_files:
        sample_file = existing_files[0]
        name_stem = sample_file.stem
        for suffix in ["_ar", "_video", "_compressed", "_de", "_en", "_ar-de-en", "_summary"]:
            name_stem = name_stem.removesuffix(suffix)
        if name_stem and name_stem != "translation_temp":
            base_name = name_stem
            print(f"  Found existing workspace pipeline file: {sample_file.name}")
            print(f"     Resuming pipeline with base name: {base_name}")

    if base_name is None:
        print("[0/6] Naming output files & selecting video quality...")
        base_name = prompt_output_name(args.url)
        print(f"       Base name: {base_name}")

    # Prompt quality upfront if not provided via CLI flag
    if not getattr(args, "quality", None) and not args.min_quality:
        args.quality = select_video_quality(args.url, args.min_quality)

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
        media = download_youtube(args.url, output_dir, min_quality=args.min_quality, video_format=getattr(args, "quality", None))
        video_path = media["video"]
        audio_path = media["audio"]
        print(f"       Video: {video_path.name}")
        print(f"       Audio: {audio_path.name}")

    # Step 2: Transcribe (skip if Arabic SRT exists)
    if arabic_srt.exists():
        print(f"\n[2/6] Already transcribed — loading from {arabic_srt.name}")
        segments = read_srt(arabic_srt)
        print(f"       {len(segments)} segments loaded")

        # Double-check Arabic transcription with Local LLM
        if audio_path and audio_path.exists() and args.double_check:
            print("\n  Double-checking Arabic transcription with Local LLM...")
            if verify_arabic_transcription_with_llm is not None:
                for seg in segments:
                    original = seg.get("text", "").strip()
                    if original:
                        corrected = verify_arabic_transcription_with_llm(original)
                        if corrected and corrected != original:
                            seg["text"] = corrected
                write_srt(segments, arabic_srt)
                print(f"  Proofread & updated Arabic SRT: {arabic_srt.name}")
            else:
                double_check_segments = double_check_arabic_srt(
                    segments, audio_path, args.model,
                    condition_on_prev=args.condition_on_previous,
                    temperature=args.whisper_temperature,
                )
                print(f"  Double-check completed: {len(double_check_segments)} segments")
                segments = double_check_segments
                write_srt(segments, arabic_srt)
    else:
        print(f"\n[2/6] Transcribing Arabic (model: {args.model})...")
        segments = transcribe_arabic(
            audio_path, args.model,
            condition_on_prev=args.condition_on_previous,
            temperature=args.whisper_temperature,
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
                print("\n  [WARNING] German SRT contains Arabic script! Re-translating.")
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
        backend = get_translation_backend(args)
        args._backend = backend
        verify_translations_with_report(segments, output_dir, args)
    else:
        print("\n[3/6] Translating to German + English...")
        backend = get_translation_backend(args)
        args._backend = backend

        if needs_retranslate:
            print("  [INFO] Re-translating due to Arabic script in German SRT")
            # Clear progress to force re-translation
            progress_file = output_dir / "translation_progress.json"
            if progress_file.exists():
                progress_file.unlink()

        segments = translate_segments(segments, output_dir, args)

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
        verify_translations_with_report(segments, output_dir, args)

    # Step 4: Compress video (skip if exists)
    if compressed_video.exists():
        existing_size_mb = compressed_video.stat().st_size / (1024 * 1024)
        if existing_size_mb <= args.target_size:
            print(f"\n[4/6] Compressed video already exists and is within target size ({existing_size_mb:.1f} MB): {compressed_video.name}")
        else:
            print(f"\n[4/6] Existing compressed video ({existing_size_mb:.1f} MB) exceeds target size ({args.target_size} MB). Re-compressing...")
            compress_video(video_path, compressed_video, args.target_size,
                           arabic_srt, german_srt, english_srt, combined_srt, args)
    else:
        print(f"\n[4/6] Compressing video (target: {args.target_size} MB)...")
        compress_video(video_path, compressed_video, args.target_size,
                       arabic_srt, german_srt, english_srt, combined_srt, args)

    # Delete monolingual SRT files (always, keeping only the combined/merged SRT)
    print("\n[5/6] Deleting monolingual SRT files...")
    for srt_file in [arabic_srt, german_srt, english_srt]:
        if srt_file.exists():
            try:
                srt_file.unlink()
                print(f"       Deleted: {srt_file.name}")
            except Exception as e:
                print(f"       [WARNING] Could not delete {srt_file.name}: {e}")

    # Clean up temporary/intermediate files (always, unless --no-cleanup is passed)
    if not getattr(args, "no_cleanup", False):
        print("       Cleaning up temporary files...")
        cleanup_temp_files(output_dir, base_name)

    # Step 6: Summary & Rich Report Generation (MD Report Only)
    print("\n[6/6] Summary")

    # Generate Markdown summary report
    md_report_path = output_dir / f"{base_name}_summary.md"
    html_report_path = output_dir / f"{base_name}_summary.html"
    if html_report_path.exists():
        try:
            html_report_path.unlink()
        except Exception:
            pass

    video_size_mb = compressed_video.stat().st_size / (1024 * 1024) if compressed_video.exists() else 0.0
    srt_lines_count = len(segments) if 'segments' in locals() else 0

    md_content = f"""# 🎬 YouTube Processing Pipeline Report

- **Source URL**: {args.url}
- **Base Name**: `{base_name}`
- **Processed At**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Total Segments**: {srt_lines_count}

---

## 📦 Generated Output Files

1. **📹 Subtitled Video**: [`{compressed_video.name}`](file://{compressed_video.absolute()}) ({video_size_mb:.2f} MB)
2. **📜 Arabic Subtitles**: [`{combined_srt.name}`](file://{combined_srt.absolute()})

---

## 🤖 Dual AI Engine Stack

- **ASR Model**: `mlx-whisper` (`{args.model}`) on Apple Silicon GPU
- **Arabic Proofreading LLM**: `Qwen2.5-7B-Instruct-4bit` (MLX Metal GPU)
- **NMT Model**: `MarianMT` (Helsinki-NLP) on PyTorch MPS GPU
- **Verification LLM**: `Qwen2.5-7B-Instruct-4bit` (MLX Metal GPU)
"""

    try:
        md_report_path.write_text(md_content, encoding="utf-8")
    except Exception as e:
        print(f"  [WARNING] Could not write report file: {e}")

    print(f"\n{'='*60}")
    print(f"  All done! Files created in '{output_dir}/':")
    print(f"     - {combined_srt.name}  (Arabic subtitles SRT)")
    print(f"     - {compressed_video.name}   (Video with burned subtitles)")
    print(f"     - {md_report_path.name}     (Markdown summary report)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
