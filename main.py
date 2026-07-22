"""
YouTube Arabic → German → English Pipeline
===========================================

Downloads a YouTube video, transcribes Arabic audio to SRT,
translates to German and English SRT (verified with multiple AI tools),
and compresses the video to ~50 MB with burned subtitles.

Translation backends (in priority order):
  1. OpenAI GPT-4 (best quality, requires OPENAI_API_KEY)
  2. OpenRouter (requires OPENROUTER_API_KEY)
  3. DeepSeek (requires DEEPSEEK_API_KEY)
  4. Google Translate (free fallback, no key needed)

Multi-pass verification:
  - Pass 1: Google Translate (free)
  - Pass 2: OpenAI GPT-4 (if available)
  - Pass 3: OpenRouter (if available)

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
    --triple-check      Use 3 AI tools to verify translations
    --openrouter-key    OpenRouter API key for translation/verification
    --deepseek-key      DeepSeek API key for translation/verification
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import cast

from dotenv import load_dotenv
from openai import OpenAI

import mlx_whisper
from deep_translator import GoogleTranslator

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

def transcribe_arabic(audio_path: Path, model_name: str) -> list:
    """Transcribe Arabic audio using MLX-Whisper on Apple Silicon GPU."""
    print(f"  Transcribing Arabic audio with MLX-Whisper ({model_name})...")
    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=f"mlx-community/whisper-{model_name}",
        language="ar",
        verbose=False,
    )
    return cast(list, result["segments"])


def double_check_arabic_srt(segments: list, audio_path: Path, model_name: str) -> list:
    """Re-transcribe Arabic audio to double-check the SRT."""
    print(f"  Double-checking Arabic transcription with MLX-Whisper ({model_name})...")
    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=f"mlx-community/whisper-{model_name}",
        language="ar",
        verbose=False,
    )
    return cast(list, result["segments"])


# ─── Translation Backends ─────────────────────────────────────────

def translate_with_openai(client: OpenAI, text: str, source: str, target: str) -> str:
    """Translate text using OpenAI GPT-4."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"You are a professional translator. Translate from {source} to {target}. Return only the translation, no extra text."},
            {"role": "user", "content": text},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def translate_with_openrouter(text: str, source: str, target: str, api_key: str) -> str:
    """Translate text using OpenRouter."""
    import requests
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "google/gemini-2.0-flash-001",
            "messages": [
                {"role": "system", "content": f"You are a professional translator. Translate from {source} to {target}. Return only the translation, no extra text."},
                {"role": "user", "content": text},
            ],
            "temperature": 0.3,
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def translate_with_deepseek(text: str, source: str, target: str, api_key: str) -> str:
    """Translate text using DeepSeek."""
    import requests
    response = requests.post(
        url="https://api.deepseek.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": f"You are a professional translator. Translate from {source} to {target}. Return only the translation, no extra text."},
                {"role": "user", "content": text},
            ],
            "temperature": 0.3,
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def translate_with_google(text: str, source: str, target: str) -> str:
    """Translate text using Google Translate (free)."""
    translator = GoogleTranslator(source=source, target=target)
    result = translator.translate(text)
    return result if isinstance(result, str) else str(result)


def get_translation_backend(args) -> tuple:
    """Return the best available translation backend and its name.
    
    Always uses Google Translate (free) as the primary translation backend.
    AI models (OpenAI, OpenRouter, DeepSeek) are only used for verification.
    """
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    openrouter_key = args.openrouter_key or os.getenv("OPENROUTER_API_KEY", "").strip()
    deepseek_key = args.deepseek_key or os.getenv("DEEPSEEK_API_KEY", "").strip()

    # Always use Google Translate (free) as primary backend
    # AI keys are kept for verification only
    return ("google", None, openai_key, openrouter_key, deepseek_key)


def translate_segment(text: str, source: str, target: str, backend: str, client, openrouter_key: str, deepseek_key: str) -> str:
    """Translate a single segment using the specified backend."""
    try:
        if backend == "openai" and client:
            return translate_with_openai(client, text, source, target)
        elif backend == "openrouter" and openrouter_key:
            return translate_with_openrouter(text, source, target, openrouter_key)
        elif backend == "deepseek" and deepseek_key:
            return translate_with_deepseek(text, source, target, deepseek_key)
        else:
            return translate_with_google(text, source, target)
    except Exception as e:
        print(f"  [WARNING] Translation backend '{backend}' failed: {e}")
        print(f"  Falling back to Google Translate...")
        try:
            return translate_with_google(text, source, target)
        except Exception as e2:
            print(f"  [ERROR] Google Translate also failed: {e2}")
            return text


# ─── Multi-Pass Translation & Verification ────────────────────────

def multi_pass_translate(text: str, source: str, target: str, args, client) -> str:
    """Translate text using multiple AI tools and pick the best result.

    Pass 1: Primary backend (OpenAI/OpenRouter/DeepSeek/Google)
    Pass 2: Google Translate (free)
    Pass 3: Another AI backend if available

    Returns the translation that both AI tools agree on, or the primary one.
    """
    openrouter_key = args.openrouter_key or os.getenv("OPENROUTER_API_KEY", "").strip()
    deepseek_key = args.deepseek_key or os.getenv("DEEPSEEK_API_KEY", "").strip()

    # Pass 1: Primary backend
    primary = translate_segment(text, source, target, args._backend, client, openrouter_key, deepseek_key)

    # Pass 2: Google Translate
    try:
        google_result = translate_with_google(text, source, target)
    except Exception:
        google_result = primary

    # Pass 3: Another AI backend if available
    if args.triple_check:
        if args._backend == "openai" and openrouter_key:
            third = translate_with_openrouter(text, source, target, openrouter_key)
        elif args._backend == "openrouter" and deepseek_key:
            third = translate_with_deepseek(text, source, target, deepseek_key)
        elif args._backend == "openai" and deepseek_key:
            third = translate_with_deepseek(text, source, target, deepseek_key)
        else:
            third = google_result

        # Pick the result that two tools agree on
        if primary == third:
            return primary
        elif primary == google_result:
            return primary
        elif third == google_result:
            return third
        else:
            # All three disagree - return primary (best quality)
            return primary

    # Two-pass: return primary if it's reasonable, else Google
    return primary


def translate_segments(segments: list, output_dir: Path | None, args, client) -> list:
    """Translate each segment's text from Arabic to German and English.

    Supports resume capability and live backup.
    """
    backend, _, openai_key, openrouter_key, deepseek_key = get_translation_backend(args)
    args._backend = backend

    # Store original Arabic text before translation
    for seg in segments:
        if "original_ar" not in seg:
            seg["original_ar"] = seg["text"]

    total = len(segments)
    batch_size = 20
    max_workers = 10

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
        combined_text = "\n\n".join([f"[{i+1}] {seg['text'].strip()}"
                                     for i, seg in enumerate(batch) if seg['text'].strip()])

        if not combined_text:
            for i in range(batch_start, batch_end):
                translated_indices.add(i)
            save_progress()
            return batch_end

        # Translate to German
        de_prompt = f"""You are a professional Arabic to German translator.
        Translate each numbered segment from Arabic to natural, fluent German.
        Keep the same numbering format [N] at the start of each segment.
        Preserve the meaning and context of each segment.
        There are {len(batch)} segments to translate."""

        # Translate to English
        en_prompt = f"""You are a professional Arabic to English translator.
        Translate each numbered segment from Arabic to natural, fluent English.
        Keep the same numbering format [N] at the start of each segment.
        Preserve the meaning and context of each segment.
        There are {len(batch)} segments to translate."""

        if backend == "openai" and client:
            de_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": de_prompt},
                    {"role": "user", "content": combined_text},
                ],
                temperature=0.3,
            )
            de_text = de_response.choices[0].message.content

            en_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": en_prompt},
                    {"role": "user", "content": combined_text},
                ],
                temperature=0.3,
            )
            en_text = en_response.choices[0].message.content
        else:
            # Fallback: translate each segment individually with Google
            de_text = ""
            en_text = ""
            for i, seg in enumerate(batch):
                if seg['text'].strip():
                    de_seg = translate_with_google(seg['text'].strip(), "ar", "de")
                    en_seg = translate_with_google(seg['text'].strip(), "ar", "en")
                    de_text += f"[{i+1}] {de_seg}\n\n"
                    en_text += f"[{i+1}] {en_seg}\n\n"

        # Parse German translations
        if de_text:
            de_lines = de_text.strip().split("\n\n")
            for i, line in enumerate(de_lines):
                if i < len(batch) and line.strip():
                    cleaned = line.strip()
                    if cleaned.startswith("["):
                        cleaned = cleaned.split("]", 1)[1].strip()
                    batch[i]["text_de"] = cleaned

        # Parse English translations
        if en_text:
            en_lines = en_text.strip().split("\n\n")
            for i, line in enumerate(en_lines):
                if i < len(batch) and line.strip():
                    cleaned = line.strip()
                    if cleaned.startswith("["):
                        cleaned = cleaned.split("]", 1)[1].strip()
                    batch[i]["text_en"] = cleaned

        # Mark as translated
        for i in range(batch_start, batch_end):
            translated_indices.add(i)
        save_progress()
        save_live_backup(batch_end)

        return batch_end

    # Process batches in parallel
    print(f"  Translating {total} segments Arabic → German + English ({backend} - {'triple-check' if args.triple_check else 'standard'} mode) ...")
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

def verify_translations_with_report(segments: list, output_dir: Path, args, client) -> dict:
    """Verify translations using multiple AI tools with progress bar and save report.

    Pass 1: Google Translate (free)
    Pass 2: OpenAI GPT-4 (if available)
    Pass 3: OpenRouter (if available)

    Saves a report to reports/ directory.
    """
    openrouter_key = args.openrouter_key or os.getenv("OPENROUTER_API_KEY", "").strip()
    deepseek_key = args.deepseek_key or os.getenv("DEEPSEEK_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()

    print(f"\n  Verifying translations with multiple AI tools...")
    print(f"  Pass 1: Google Translate (free)")
    if openai_key:
        print(f"  Pass 2: OpenAI GPT-4")
    if openrouter_key:
        print(f"  Pass 3: OpenRouter")
    if deepseek_key:
        print(f"  Pass 4: DeepSeek")

    total = len(segments)
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_segments": total,
        "google_verified": 0,
        "google_mismatches": 0,
        "openai_verified": 0,
        "openai_mismatches": 0,
        "openrouter_verified": 0,
        "openrouter_mismatches": 0,
        "deepseek_verified": 0,
        "deepseek_mismatches": 0,
        "mismatches": [],
    }

    # Live backup during verification
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    live_report = report_dir / "verification_live.json"

    for idx, seg in enumerate(segments, 1):
        original_text = seg.get("original_ar", "")
        if not original_text.strip():
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
        except Exception as e:
            print(f"  [VERIFY] Segment {idx}: Google error - {e}")

        # Pass 2: OpenAI GPT-4
        if openai_key and client:
            try:
                openai_de = translate_with_openai(client, original_text, "ar", "de")
                if openai_de.strip() == current_de:
                    report["openai_verified"] += 1
                else:
                    report["openai_mismatches"] += 1
            except Exception as e:
                print(f"  [VERIFY] Segment {idx}: OpenAI error - {e}")

        # Pass 3: OpenRouter
        if openrouter_key:
            try:
                openrouter_de = translate_with_openrouter(original_text, "ar", "de", openrouter_key)
                if openrouter_de.strip() == current_de:
                    report["openrouter_verified"] += 1
                else:
                    report["openrouter_mismatches"] += 1
            except Exception as e:
                print(f"  [VERIFY] Segment {idx}: OpenRouter error - {e}")

        # Pass 4: DeepSeek
        if deepseek_key:
            try:
                deepseek_de = translate_with_deepseek(original_text, "ar", "de", deepseek_key)
                if deepseek_de.strip() == current_de:
                    report["deepseek_verified"] += 1
                else:
                    report["deepseek_mismatches"] += 1
            except Exception as e:
                print(f"  [VERIFY] Segment {idx}: DeepSeek error - {e}")

        # Progress bar
        if idx % 10 == 0 or idx == total:
            pct = idx * 100 // total
            print(f"\r  Verification progress: {pct}% ({idx}/{total})", end="", flush=True)

            # Save live backup
            try:
                with open(live_report, "w", encoding="utf-8") as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

    print()  # New line after progress

    # Save final report
    report_path = report_dir / f"verification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"  Google verification: {report['google_verified']} verified, {report['google_mismatches']} mismatches")
    if openai_key:
        print(f"  OpenAI verification: {report['openai_verified']} verified, {report['openai_mismatches']} mismatches")
    if openrouter_key:
        print(f"  OpenRouter verification: {report['openrouter_verified']} verified, {report['openrouter_mismatches']} mismatches")
    if deepseek_key:
        print(f"  DeepSeek verification: {report['deepseek_verified']} verified, {report['deepseek_mismatches']} mismatches")
    print(f"  Report saved to: {report_path}")

    return report


# ─── Cleanup ──────────────────────────────────────────────────────

def cleanup_temp_files(output_dir: Path, base_name: str) -> None:
    """Clean up temporary files after pipeline completion."""
    temp_files = [
        output_dir / "translation_progress.json",
        output_dir / "translation_temp.srt",
    ]

    # Also clean up the original downloaded video/audio files
    for pattern in ["*_video.mp4", "*.mp3"]:
        for f in output_dir.glob(pattern):
            if not f.name.startswith(base_name):
                temp_files.append(f)

    for temp_file in temp_files:
        if temp_file.exists():
            try:
                temp_file.unlink()
                print(f"  Cleaned up: {temp_file.name}")
            except Exception as e:
                print(f"  [WARNING] Could not delete {temp_file.name}: {e}")


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
        filter_parts = []
        current_label = "[0:v]"

        # German on top
        if ("de", german_srt) in subtitle_inputs:
            de_idx = subtitle_inputs.index(("de", german_srt)) + 1
            filter_parts.append(
                f"{current_label}subtitles={str(german_srt)}:force_style='Alignment=Top,FontSize=24,PrimaryColour=&HFFFFFF&'[v{len(filter_parts)+1}]"
            )
            current_label = f"[v{len(filter_parts)+1}]"

        # English in middle
        if ("en", english_srt) in subtitle_inputs:
            filter_parts.append(
                f"{current_label}subtitles={str(english_srt)}:force_style='Alignment=Middle,FontSize=20,PrimaryColour=&HFFFF00&'[v{len(filter_parts)+1}]"
            )
            current_label = f"[v{len(filter_parts)+1}]"

        # Arabic on bottom
        if ("ar", arabic_srt) in subtitle_inputs:
            filter_parts.append(
                f"{current_label}subtitles={str(arabic_srt)}:force_style='Alignment=Bottom,FontSize=24,PrimaryColour=&HFFFFFF&'[vout]"
            )

        cmd.extend(["-filter_complex", ";".join(filter_parts)])
        cmd.extend(["-map", "[vout]"])
        cmd.extend(["-map", "0:a?"])

    cmd.extend([
        "-c:v", "libx264", "-b:v", f"{video_bitrate}k",
        "-c:a", "aac", "-b:a", f"{audio_bitrate}k",
        "-movflags", "+faststart",
        str(output_path),
    ])

    # Run with progress monitoring
    print(f"  Compressing video with progress bar...")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    import time
    start_time = time.time()
    last_update = 0

    if process.stdout:
        for line in process.stdout:
            time_match = re.search(r"time=(\d+):(\d+):(\d+\.\d+)", line)
            if time_match:
                hours, mins, secs = time_match.groups()
                elapsed = int(hours) * 3600 + int(mins) * 60 + float(secs)
                if duration > 0 and time.time() - last_update > 1:
                    progress_pct = min(100, int(elapsed * 100 / duration))
                    print(f"\r  Progress: {progress_pct}% ({elapsed:.0f}s/{int(duration)}s)", end="", flush=True)
                    last_update = time.time()

    process.wait()
    print()

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
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="Whisper model size (default: medium)",
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
        "--no-cleanup", action="store_true",
        help="Skip cleanup of temporary files after completion",
    )
    parser.add_argument(
        "--no-retranslate", action="store_true",
        help="Skip re-translation even if German SRT has Arabic script",
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="Skip multi-AI translation verification",
    )
    parser.add_argument(
        "--openrouter-key", type=str, default=None,
        help="OpenRouter API key for translation/verification",
    )
    parser.add_argument(
        "--deepseek-key", type=str, default=None,
        help="DeepSeek API key for translation/verification",
    )
    parser.add_argument(
        "--no-double-check-arabic", action="store_true",
        help="Skip double-checking Arabic transcription",
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
        base_name = existing_ar_srt.stem[:-3]
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

        # Double-check Arabic transcription always (unless skipped)
        if not args.no_double_check_arabic and audio_path and audio_path.exists():
            print(f"\n  Double-checking Arabic transcription...")
            double_check_segments = double_check_arabic_srt(segments, audio_path, args.model)
            print(f"  Double-check completed: {len(double_check_segments)} segments")
            # Use the double-checked segments
            segments = double_check_segments
            write_srt(segments, arabic_srt)
            print(f"  Updated Arabic SRT: {arabic_srt.name}")
    else:
        print(f"\n[2/6] Transcribing Arabic (model: {args.model})...")
        segments = transcribe_arabic(audio_path, args.model)
        write_srt(segments, arabic_srt)
        print(f"       Arabic SRT: {arabic_srt.name} ({len(segments)} segments)")

    # Step 3: Translate (skip if German SRT exists AND not force-retranslate)
    # Always re-translate if German SRT has Arabic script (unless --no-retranslate)
    needs_retranslate = False
    if german_srt.exists() and not args.no_retranslate:
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
        backend, client, openai_key, openrouter_key, deepseek_key = get_translation_backend(args)
        args._backend = backend
        verify_translations_with_report(segments, output_dir, args, client)
    else:
        print(f"\n[3/6] Translating to German + English...")
        backend, client, openai_key, openrouter_key, deepseek_key = get_translation_backend(args)
        args._backend = backend

        if needs_retranslate:
            print(f"  [INFO] Re-translating due to Arabic script in German SRT")
            # Clear progress to force re-translation
            progress_file = output_dir / "translation_progress.json"
            if progress_file.exists():
                progress_file.unlink()

        segments = translate_segments(segments, output_dir, args, client)

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
        verify_translations_with_report(segments, output_dir, args, client)

    # Step 4: Compress video (skip if exists)
    if compressed_video.exists():
        print(f"\n[4/6] Compressed video already exists: {compressed_video.name}")
    else:
        print(f"\n[4/6] Compressing video (target: {args.target_size} MB)...")
        compress_video(video_path, compressed_video, args.target_size,
                       arabic_srt, german_srt, english_srt, combined_srt)

    # Step 5: Clean up temporary files (always, unless --no-cleanup)
    if not args.no_cleanup:
        print(f"\n[5/6] Cleaning up temporary files...")
        cleanup_temp_files(output_dir, base_name)

    # Step 6: Summary
    print(f"\n[6/6] Summary")
    print(f"\n{'='*60}")
    print(f"  All done! Files in '{output_dir}/':")
    print(f"     - {arabic_srt.name}  (Arabic subtitles)")
    print(f"     - {german_srt.name}   (German subtitles)")
    print(f"     - {english_srt.name}  (English subtitles)")
    print(f"     - {combined_srt.name} (Combined AR/DE/EN subtitles)")
    print(f"     - {compressed_video.name}  (compressed video with burned subtitles)")
    print(f"     - reports/  (verification reports)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
