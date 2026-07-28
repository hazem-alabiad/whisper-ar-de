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

import os
import argparse
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import cast

# Restrict global CPU threads & memory allocation to protect Mac hardware
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import mlx_whisper
from deep_translator import GoogleTranslator
from dotenv import load_dotenv

try:
    from whisper_tools.translation import (
        translate_segments,
        unload_local_models,
        verify_translations_with_report,
        verify_transcription_with_llm,
        ensure_models_downloaded,
    )
except ImportError:
    from .translation import (
        translate_segments,
        unload_local_models,
        verify_translations_with_report,
        verify_transcription_with_llm,
        ensure_models_downloaded,
    )

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


def write_multilingual_srt(source_segs: list, target_segs_dict: dict[str, list], source_lang: str, target_langs: list[str], srt_path: Path) -> None:
    """Write a combined SRT file with source and multiple target subtitles."""
    with open(srt_path, "w", encoding="utf-8") as f:
        total = len(source_segs)
        for idx in range(total):
            f.write(f"{idx+1}\n")
            src_seg = source_segs[idx]
            f.write(f"{format_time(src_seg['start'])} --> {format_time(src_seg['end'])}\n")
            f.write(f"{source_lang.upper()}: {src_seg['text'].strip()}\n")
            for target_lang in target_langs:
                segs = target_segs_dict.get(target_lang, [])
                if idx < len(segs):
                    text = segs[idx]['text'].strip()
                    if text:
                        f.write(f"{target_lang.upper()}: {text}\n")
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


def download_youtube(url: str, output_dir: Path, base_name: str, min_quality: bool = False, video_format: str | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_format:
        video_format = select_video_quality(url, min_quality)

    video_path = output_dir / f"{base_name}_video.mp4"
    audio_path = output_dir / f"{base_name}.mp3"

    subprocess.run(
        ["yt-dlp", "-f", video_format, "-o", str(video_path), url],
        check=True, capture_output=True, text=True,
    )

    try:
        subprocess.run(
            ["yt-dlp", "-x", "--audio-format", "mp3", "-o", str(audio_path), url],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError:
        print("[WARNING] MP3 audio extraction failed, falling back to m4a audio download.")
        audio_path = output_dir / f"{base_name}_audio.m4a"
        subprocess.run(
            ["yt-dlp", "-f", "bestaudio[ext=m4a]/bestaudio", "-o", str(audio_path), url],
            check=True, capture_output=True, text=True,
        )

    return {
        "video_id": base_name,
        "video": video_path,
        "audio": audio_path,
    }


# ─── Transcription ─────────────────────────────────────────────────

def transcribe_audio(audio_path: Path, model_name: str, source_lang: str = "ar", condition_on_prev: bool = False, temperature: float | None = None) -> list:
    """Transcribe audio using MLX-Whisper on Apple Silicon GPU."""
    print(f"  Transcribing audio with MLX-Whisper ({model_name}) in '{source_lang}'...")
    hf_repo = model_name if "/" in model_name else f"mlx-community/whisper-{model_name}"

    kwargs = {
        "path_or_hf_repo": hf_repo,
        "language": source_lang,
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


def double_check_srt_audio(segments: list, audio_path: Path, model_name: str, source_lang: str = "ar", condition_on_prev: bool = False, temperature: float | None = None) -> list:
    """Re-transcribe audio to double-check the SRT."""
    print(f"  Double-checking transcription with MLX-Whisper ({model_name}) in '{source_lang}'...")
    hf_repo = model_name if "/" in model_name else f"mlx-community/whisper-{model_name}"

    kwargs = {
        "path_or_hf_repo": hf_repo,
        "language": source_lang,
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
                   srt_files: dict[str, Path],
                   source_lang: str,
                   target_langs: list[str],
                   args: argparse.Namespace | None = None) -> None:
    """Compress video and burn subtitles using ffmpeg with progress bar.

    Burns up to 3 subtitle tracks dynamically.
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
    # Add source language first
    if source_lang in srt_files and srt_files[source_lang].exists():
        subtitle_inputs.append((source_lang, srt_files[source_lang]))
    # Add target languages
    for target in target_langs:
        if target in srt_files and srt_files[target].exists():
            subtitle_inputs.append((target, srt_files[target]))
    # Add combined if present
    if "combined" in srt_files and srt_files["combined"].exists():
        subtitle_inputs.append(("combined", srt_files["combined"]))

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

            # Subtitle Layout Menu (interactive / argument driven)
            sub_layout = getattr(args, "sub_layout", None)
            if not sub_layout:
                print("\n  Select Subtitle Burning Layout for Video:")
                print(f"    [1] Split Style: {source_lang.upper()} (Bottom) + targets (Top/Middle)")
                print("    [2] Single Bottom Stack: All languages stacked at bottom (like YouTube captions)")
                print(f"    [3] {source_lang.upper()} Only at Bottom: Clean single-language bottom subtitle")
                layout_choice = input("  Enter choice [1-3] (default: 2 Single Bottom Stack): ").strip()
                sub_layout = layout_choice if layout_choice in ["1", "2", "3"] else "2"

            if sub_layout == "3":
                burn_inputs = [x for x in burn_inputs if x[0] == source_lang]
                total_subtitles = len(burn_inputs)

            for idx, (lang, srt_path) in enumerate(burn_inputs):
                if sub_layout == "2":
                    # YouTube bottom stack
                    if idx == 0:  # Source language at bottom
                        style = "FontName=Arial\\,FontSize=26\\,PrimaryColour=&HFFFFFF&\\,OutlineColour=&H000000&\\,BackColour=&H80000000&\\,BorderStyle=4\\,Outline=2\\,Shadow=1\\,MarginV=14\\,Alignment=2"
                    elif idx == 1:  # First target in middle
                        style = "FontName=Arial\\,FontSize=22\\,PrimaryColour=&H00FFFF&\\,OutlineColour=&H000000&\\,BackColour=&H80000000&\\,BorderStyle=4\\,Outline=2\\,Shadow=1\\,MarginV=42\\,Alignment=2"
                    else:  # Second target at top
                        style = "FontName=Arial\\,FontSize=24\\,PrimaryColour=&HFFFFFF&\\,OutlineColour=&H000000&\\,BackColour=&H80000000&\\,BorderStyle=4\\,Outline=2\\,Shadow=1\\,MarginV=68\\,Alignment=2"
                elif sub_layout == "3":
                    # Source only at bottom
                    style = "FontName=Arial\\,FontSize=28\\,PrimaryColour=&HFFFFFF&\\,OutlineColour=&H000000&\\,BorderStyle=1\\,Outline=2.5\\,Shadow=1\\,MarginV=20\\,Alignment=2"
                else:
                    # Split top/middle/bottom style
                    if idx == 0:  # Bottom
                        style = "FontName=Arial\\,FontSize=26\\,PrimaryColour=&HFFFFFF&\\,OutlineColour=&H000000&\\,BorderStyle=1\\,Outline=3\\,Shadow=1\\,MarginV=20\\,Alignment=2"
                    elif idx == 1:  # Top
                        style = "FontName=Arial\\,FontSize=26\\,PrimaryColour=&HFFFFFF&\\,OutlineColour=&H000000&\\,BorderStyle=1\\,Outline=3\\,Shadow=1\\,MarginV=20\\,Alignment=8"
                    else:  # Middle
                        style = "FontName=Arial\\,FontSize=22\\,PrimaryColour=&H00FFFF&\\,OutlineColour=&H000000&\\,BorderStyle=1\\,Outline=3\\,Shadow=1\\,MarginV=20\\,Alignment=5"

                next_label = f"[v{len(filter_parts)+1}]" if len(filter_parts) + 1 < total_subtitles else "[vout]"
                filter_parts.append(
                    f"{current_label}subtitles=filename='{escape_ffmpeg_path(srt_path)}':force_style='{style}'{next_label}"
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
                cmd.extend([f"-metadata:s:s:{idx}", f"language={lang_map.get(lang, lang[:3])}"])
                title_map = {"de": "German", "en": "English", "ar": "Arabic", "combined": "Combined Subtitles"}
                lang_title = title_map.get(lang, f"{lang.upper()} Subtitles")
                cmd.extend([f"-metadata:s:s:{idx}", f"title={lang_title}"])

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
        description="YouTube & Book Arabic → German / English Dual Translation Pipeline."
    )
    parser.add_argument("url", type=str, nargs="?", default=None, help="YouTube video URL or path to book file")
    parser.add_argument(
        "--book", type=str, default=None,
        help="Path to book file (.txt, .pdf, .docx, .md) to translate interactively",
    )
    parser.add_argument(
        "--source-lang", type=str, default="ar",
        help="Source language of the input (default: ar)",
    )
    parser.add_argument(
        "--target-lang", type=str, default="de,en",
        help="Comma-separated target languages for translation (default: de,en)",
    )
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
        "--no-local-translate",
        action="store_false",
        dest="local_translate",
        help="Disable local neural models and fall back to Google Translate",
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
    parser.add_argument(
        "--formats", type=str, default="all",
        help="Comma-separated book export formats: pdf, epub, docx, md, or all (default: all)",
    )
    return parser.parse_args()


# ─── Main Pipeline ────────────────────────────────────────────────

def run_interactive_launcher(args):
    """Enter an interactive CLI configuration and mode selector menu when run without arguments."""
    import sys
    print("\n" + "=" * 60)
    print("  🤖 WHISPER-TOOLS — BUILT WITH AGENTIC AI & LLM STACK")
    print("=" * 60)
    print("  ✨ Engineered 100% via Antigravity AI Pair Programming")
    print("  🧠 Apple Silicon MLX GPU + Multi-Lingual Dual LLM Pipeline")
    print("=" * 60)
    
    while True:
        print("\nPlease select an action:")
        print("  [1] Translate a YouTube Video (subtitle burning & compression)")
        print("  [2] Translate a Local Book (.txt, .pdf, .docx, .md)")
        print("  [3] Review CLI Flags, Arguments & Options (Help)")
        print("  [4] Interactively Select & Pre-download Models to Disk")
        print("  [5] Exit")
        
        choice = input("\n  Enter choice [1-5]: ").strip()
        
        if choice == "1":
            print("\n--- 🎬 YouTube Video Translation Configuration ---")
            url = input("  Enter YouTube URL: ").strip()
            while not url:
                url = input("  URL cannot be empty. Enter YouTube URL: ").strip()
            args.url = url
            
            src = input(f"  Enter source language code (default: {getattr(args, 'source_lang', 'ar')}): ").strip().lower()
            if src:
                args.source_lang = src
                
            targets = input(f"  Enter comma-separated target language codes (default: {getattr(args, 'target_lang', 'de,en')}): ").strip().lower()
            if targets:
                args.target_lang = targets
                
            quality = input("  Set video download quality preset [1=low, 2=medium, 3=high, 4=best] (default: ask upfront): ").strip()
            if quality in ("1", "2", "3", "4", "low", "medium", "high", "best"):
                args.quality = quality

            print("\n  Select Whisper ASR Model Size:")
            print("    [1] base (150MB - Fastest, lowest memory usage)")
            print("    [2] small (460MB - Balanced speed/accuracy)")
            print("    [3] medium (1.5GB - High accuracy)")
            print("    [4] large-v3-turbo-q4 (1.6GB - Best quality, default)")
            model_choice = input("  Enter model choice [1-4] (default: 4): ").strip()
            if model_choice == "1":
                args.model = "base"
            elif model_choice == "2":
                args.model = "small"
            elif model_choice == "3":
                args.model = "medium"
            else:
                args.model = "large-v3-turbo-q4"
                
            local_ans = input("\n  Use local translation models? (y/n, default: y): ").strip().lower()
            if local_ans == "n":
                args.local_translate = False
            else:
                args.local_translate = True
                print("\n  Select Local Translation LLM Model:")
                print("    [1] Qwen2.5-7B-Instruct-4bit (~4.3GB RAM - High Accuracy Sweet Spot, default)")
                print("    [2] Qwen2.5-14B-Instruct-4bit (~9.2GB RAM - Ultimate Quality)")
                print("    [3] Qwen2.5-3B-Instruct-8bit (~3.5GB RAM - Fast Lightweight)")
                llm_choice = input("  Enter LLM model choice [1-3] (default: 1): ").strip()
                if llm_choice == "2":
                    args.mlx_model = "mlx-community/Qwen2.5-14B-Instruct-4bit"
                elif llm_choice == "3":
                    args.mlx_model = "mlx-community/Qwen2.5-3B-Instruct-8bit"
                else:
                    args.mlx_model = "mlx-community/Qwen2.5-7B-Instruct-4bit"
                
            return args
            
        elif choice == "2":
            print("\n--- 📚 Book Translation Configuration ---")
            books_dir = Path("books")
            books_dir.mkdir(exist_ok=True)
            
            # List available books in books/
            books = sorted([f for f in books_dir.iterdir() if f.is_file() and f.suffix.lower() in (".txt", ".pdf", ".docx", ".md")])
            if books:
                print("\n  Available books found in 'books/':")
                for idx, b in enumerate(books):
                    print(f"    [{idx+1}] {b.name}")
                book_sel = input(f"\n  Select a book number [1-{len(books)}] or enter a custom path: ").strip()
                if book_sel.isdigit() and 1 <= int(book_sel) <= len(books):
                    args.book = str(books[int(book_sel) - 1])
                else:
                    args.book = book_sel
            else:
                book_path = input("  Enter path to book file (.txt, .pdf, .docx, .md): ").strip()
                while not book_path:
                    book_path = input("  Path cannot be empty. Enter path: ").strip()
                args.book = book_path

            local_ans = input("\n  Use local translation models? (y/n, default: y): ").strip().lower()
            if local_ans == "n":
                args.local_translate = False
            else:
                args.local_translate = True
                print("\n  Select Local Translation LLM Model:")
                print("    [1] Qwen2.5-7B-Instruct-4bit (~4.3GB RAM - High Accuracy Sweet Spot, default)")
                print("    [2] Qwen2.5-14B-Instruct-4bit (~9.2GB RAM - Ultimate Quality)")
                print("    [3] Qwen2.5-3B-Instruct-8bit (~3.5GB RAM - Fast Lightweight)")
                llm_choice = input("  Enter LLM model choice [1-3] (default: 1): ").strip()
                if llm_choice == "2":
                    args.mlx_model = "mlx-community/Qwen2.5-14B-Instruct-4bit"
                elif llm_choice == "3":
                    args.mlx_model = "mlx-community/Qwen2.5-3B-Instruct-8bit"
                else:
                    args.mlx_model = "mlx-community/Qwen2.5-7B-Instruct-4bit"
                
            return args
            
        elif choice == "3":
            print("\n" + "-" * 60)
            print("                📖 CLI FLAGS & OPTIONS GUIDE")
            print("-" * 60)
            print("Usage: uv run whisper-tools [OPTIONS] [youtube_url]")
            print("\nCore Options:")
            print("  --source-lang LANG       Source language code (default: ar)")
            print("  --target-lang LANGS      Comma-separated targets (default: de,en)")
            print("  --book PATH              Path to local book file for translation")
            print("  --no-local-translate     Disable local MLX models and use Google Translate")
            print("  --target-size SIZE       Target output video size in MB (default: 50)")
            print("  --quality PRESET         Set download quality preset: 1=low, 2=med, 3=high, 4=best")
            print("  --double-check           Double check transcription segment by segment")
            print("  --no-cleanup             Keep intermediate raw video/audio files")
            print("  --sub-layout LAYOUT      Subtitle layout: 1=Top/Mid/Bot stack, 2=Bot stack, 3=Orig Bot")
            print("-" * 60)
            input("\n  Press Enter to return to main menu...")

        elif choice == "4":
            print("\n--- 📥 Interactive Model Pre-Downloader ---")
            print("Select Whisper ASR Model Size:")
            print("  [1] base (150MB)")
            print("  [2] small (460MB)")
            print("  [3] medium (1.5GB)")
            print("  [4] large-v3-turbo-q4 (1.6GB, default)")
            w_choice = input("Enter choice [1-4] (default: 4): ").strip()
            if w_choice == "1":
                args.model = "base"
            elif w_choice == "2":
                args.model = "small"
            elif w_choice == "3":
                args.model = "medium"
            else:
                args.model = "large-v3-turbo-q4"

            print("\nSelect Local Translation LLM Model:")
            print("  [1] Qwen2.5-7B-Instruct-4bit (~4.3GB RAM - High Accuracy Sweet Spot, default)")
            print("  [2] Qwen2.5-14B-Instruct-4bit (~9.2GB RAM - Ultimate Quality)")
            print("  [3] Qwen2.5-3B-Instruct-8bit (~3.5GB RAM - Fast Lightweight)")
            l_choice = input("Enter choice [1-3] (default: 1): ").strip()
            if l_choice == "2":
                args.mlx_model = "mlx-community/Qwen2.5-14B-Instruct-4bit"
            elif l_choice == "3":
                args.mlx_model = "mlx-community/Qwen2.5-3B-Instruct-8bit"
            else:
                args.mlx_model = "mlx-community/Qwen2.5-7B-Instruct-4bit"

            print("\n  Starting pre-download of selected models...")
            target_langs_list = [l.strip() for l in getattr(args, "target_lang", "de,en").split(",") if l.strip()]
            ensure_models_downloaded(args, source_lang=getattr(args, "source_lang", "ar"), target_langs=target_langs_list)
            input("\n  Pre-download complete! Press Enter to return to main menu...")
            
        elif choice == "5":
            print("\nExiting. Goodbye!")
            sys.exit(0)


_LOCK_FILE = Path("output/.gpu_pipeline.lock")

def acquire_gpu_lock():
    """Ensure only one local GPU pipeline process runs at a time to prevent macOS freezes."""
    _LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    if _LOCK_FILE.exists():
        try:
            pid = int(_LOCK_FILE.read_text().strip())
            import os
            os.kill(pid, 0)
            print(f"\n[WARNING] Another instance of whisper-tools (PID {pid}) is already running!")
            print("          Running multiple GPU pipelines concurrently causes macOS system freezes.")
            print("          Please wait for the existing process to finish, or terminate PID {pid}.\n")
        except (ValueError, OSError):
            # Stale lock file
            pass

    import os
    _LOCK_FILE.write_text(str(os.getpid()))

def release_gpu_lock():
    """Release GPU pipeline process lock upon completion or exit."""
    try:
        if _LOCK_FILE.exists():
            _LOCK_FILE.unlink()
    except Exception:
        pass


def main():
    load_dotenv()
    import atexit
    acquire_gpu_lock()
    atexit.register(release_gpu_lock)

    try:
        import mlx.core as mx  # type: ignore
        if hasattr(mx, "set_cache_limit"):
            mx.set_cache_limit(2 * 1024 * 1024 * 1024)
        elif hasattr(mx, "metal") and hasattr(mx.metal, "set_cache_limit"):
            mx.metal.set_cache_limit(2 * 1024 * 1024 * 1024)
    except Exception:
        pass

    args = parse_args()

    # If neither url nor book is provided, run interactive terminal launcher
    if not args.url and not args.book:
        args = run_interactive_launcher(args)

    # Automatically check and pre-download required AI models upfront
    target_langs_list = [l.strip() for l in getattr(args, "target_lang", "de,en").split(",") if l.strip()]
    ensure_models_downloaded(args, source_lang=getattr(args, "source_lang", "ar"), target_langs=target_langs_list)

    # Check for Book Translation Mode (--book argument or positional file path)
    books_dir = Path("books")
    books_dir.mkdir(exist_ok=True)

    book_file_path = None
    if args.book:
        candidate = Path(args.book)
        if candidate.is_file():
            book_file_path = candidate
        elif (books_dir / args.book).is_file():
            book_file_path = books_dir / args.book
    elif args.url:
        candidate = Path(args.url)
        if candidate.is_file():
            book_file_path = candidate
        elif (books_dir / args.url).is_file():
            book_file_path = books_dir / args.url

    if book_file_path and book_file_path.exists():
        try:
            from whisper_tools.book_translation import translate_book_interactive
        except ImportError:
            from .book_translation import translate_book_interactive
        output_dir = Path(args.output_dir)
        translate_book_interactive(book_file_path, output_dir, translate_segments, args)
        return

    if not args.url:
        print("\n  [ERROR] Please provide a YouTube video URL or a book file path (.txt, .pdf, .docx, .md).")
        print("  Usage for YouTube: python main.py <youtube_url>")
        print("  Usage for Books:   python main.py --book <path_to_book>")
        return

    # Enforce maximum target size of 100 MB
    if args.target_size > 100:
        print(f"[WARNING] Target size capped at 100 MB (requested {args.target_size} MB)")
        args.target_size = 100
    # Separate video outputs into output/videos/
    output_dir = Path(args.output_dir)
    if output_dir.name == "output":
        output_dir = output_dir / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)

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

    source_lang_code = getattr(args, "source_lang", "ar").lower()
    target_langs_codes = [l.strip().lower() for l in getattr(args, "target_lang", "de,en").split(",") if l.strip()]
    backend_str = "Local MLX / MarianMT" if getattr(args, "local_translate", True) else "Google Translate API"

    print(f"\n{'='*60}")
    print("  🚀 STARTING VIDEO TRANSLATION PIPELINE")
    print(f"{'='*60}")
    print(f"  🎬 Target Video URL  : {args.url}")
    print(f"  🌐 Source Language   : {source_lang_code.upper()}")
    print(f"  🎯 Target Languages  : {', '.join(t.upper() for t in target_langs_codes)}")
    print(f"  🤖 Model Backend     : {backend_str}")
    print(f"  🎙️ Whisper ASR Model  : {args.model}")
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
    # Define source and target languages dynamically from arguments
    source_lang = getattr(args, "source_lang", "ar").lower()
    target_langs = [l.strip().lower() for l in getattr(args, "target_lang", "de,en").split(",") if l.strip()]

    # Define output paths dynamically
    source_srt = output_dir / f"{base_name}_{source_lang}.srt"
    target_srts = {target: output_dir / f"{base_name}_{target}.srt" for target in target_langs}
    combined_srt = output_dir / f"{base_name}_{source_lang}-" + "-".join(target_langs) + ".srt"
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
        media = download_youtube(args.url, output_dir, base_name, min_quality=args.min_quality, video_format=getattr(args, "quality", None))
        video_path = media["video"]
        audio_path = media["audio"]
        print(f"       Video: {video_path.name}")
        print(f"       Audio: {audio_path.name}")

    # Step 2: Transcribe (skip if source SRT exists)
    if source_srt.exists():
        print(f"\n[2/6] Already transcribed — loading from {source_srt.name}")
        segments = read_srt(source_srt)
        print(f"       {len(segments)} segments loaded")

        # Double-check transcription with Local LLM
        if audio_path and audio_path.exists() and args.double_check:
            print(f"\n  Double-checking transcription with Local LLM ({source_lang})...")
            if verify_transcription_with_llm is not None:
                for seg in segments:
                    original = seg.get("text", "").strip()
                    if original:
                        corrected = verify_transcription_with_llm(original, source_lang=source_lang)
                        if corrected and corrected != original:
                            seg["text"] = corrected
                write_srt(segments, source_srt)
                print(f"  Proofread & updated source SRT: {source_srt.name}")
            else:
                double_check_segments = double_check_srt_audio(
                    segments, audio_path, args.model, source_lang=source_lang,
                    condition_on_prev=args.condition_on_previous,
                    temperature=args.whisper_temperature,
                )
                print(f"  Double-check completed: {len(double_check_segments)} segments")
                segments = double_check_segments
                write_srt(segments, source_srt)
    else:
        print(f"\n[2/6] Transcribing {source_lang.upper()} (model: {args.model})...")
        segments = transcribe_audio(
            audio_path, args.model, source_lang=source_lang,
            condition_on_prev=args.condition_on_previous,
            temperature=args.whisper_temperature,
        )
        write_srt(segments, source_srt)
        print(f"       Source SRT: {source_srt.name} ({len(segments)} segments)")

    # Explicit memory cleanup after transcription stage
    if unload_local_models is not None:
        unload_local_models()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass

    # Step 3: Translate
    all_targets_exist = all(path.exists() for path in target_srts.values())
    needs_retranslate = False

    if all_targets_exist:
        # Check if target SRTs contain untranslated source language text
        for target, path in target_srts.items():
            segs = read_srt(path)
            if segs:
                sample_text = segs[0]["text"]
                if source_lang == "ar" and target in ("de", "en") and is_arabic(sample_text):
                    print(f"\n  [WARNING] {target.upper()} SRT contains Arabic script! Re-translating.")
                    needs_retranslate = True
                    break

    if all_targets_exist and not needs_retranslate:
        print(f"\n[3/6] Already translated — loading target SRTs...")
        source_segments = read_srt(source_srt) if source_srt.exists() else []
        target_segs_dict = {target: read_srt(path) for target, path in target_srts.items()}

        # Merge into segments
        for idx, seg in enumerate(source_segments):
            seg["original_text"] = seg["text"]
            for target, segs in target_segs_dict.items():
                seg[f"text_{target}"] = segs[idx]["text"] if idx < len(segs) else ""
            if target_langs:
                seg["text"] = seg[f"text_{target_langs[0]}"]

        segments = source_segments
        print(f"       {len(segments)} segments loaded")

        if verify_translations_with_report is not None:
            verify_translations_with_report(segments, output_dir, args, source_lang=source_lang, target_langs=target_langs)
    else:
        print(f"\n[3/6] Translating to {', '.join(t.upper() for t in target_langs)}...")
        progress_file = output_dir / "translation_progress.json"
        temp_json = output_dir / "translation_temp.json"
        if progress_file.exists() and not needs_retranslate:
            try:
                import json
                prog_data = json.loads(progress_file.read_text(encoding="utf-8"))
                completed_cnt = prog_data.get("completed_count", 0)
                print(f"  [FOUND PREVIOUS BACKUP] Found translation progress for video '{base_name}' ({completed_cnt} segments completed).")
                res_ans = input("  Resume from existing backup to speed up execution? (y/n, default: y): ").strip().lower()
                if res_ans == "n":
                    progress_file.unlink(missing_ok=True)
                    if temp_json.exists():
                        temp_json.unlink(missing_ok=True)
                    print("  [INFO] Cleared previous backup. Starting fresh video translation from scratch!\n")
                else:
                    print("  [INFO] Resuming from existing backup to speed up video translation!\n")
            except Exception:
                pass

        if needs_retranslate:
            print("  [INFO] Re-translating due to script validation warning")
            if progress_file.exists():
                progress_file.unlink(missing_ok=True)

        if translate_segments is not None:
            segments = translate_segments(segments, output_dir, args, source_lang=source_lang, target_langs=target_langs)

        # Write target language SRTs
        for target, path in target_srts.items():
            t_segs = []
            for seg in segments:
                t_segs.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg.get(f"text_{target}", ""),
                })
            write_srt(t_segs, path)
            print(f"       {target.upper()} SRT: {path.name}")

        # Write combined multi-lingual SRT
        target_segs_dict = {target: read_srt(path) for target, path in target_srts.items()}
        write_multilingual_srt(
            [{"start": s["start"], "end": s["end"], "text": s.get("original_text", s.get("text", ""))} for s in segments],
            target_segs_dict,
            source_lang,
            target_langs,
            combined_srt,
        )
        print(f"       Combined SRT: {combined_srt.name}")

        if verify_translations_with_report is not None:
            verify_translations_with_report(segments, output_dir, args, source_lang=source_lang, target_langs=target_langs)

    # Explicit memory cleanup after translation stage
    if unload_local_models is not None:
        unload_local_models()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass

    # Step 4: Compress video & burn subtitles (skip if exists)
    srt_files = {source_lang: source_srt}
    for target, path in target_srts.items():
        srt_files[target] = path
    srt_files["combined"] = combined_srt

    if compressed_video.exists():
        existing_size_mb = compressed_video.stat().st_size / (1024 * 1024)
        if existing_size_mb <= args.target_size:
            print(f"\n[4/6] Compressed video already exists and is within target size ({existing_size_mb:.1f} MB): {compressed_video.name}")
        else:
            print(f"\n[4/6] Existing compressed video ({existing_size_mb:.1f} MB) exceeds target size ({args.target_size} MB). Re-compressing...")
            compress_video(video_path, compressed_video, args.target_size, srt_files, source_lang, target_langs, args)
    else:
        print(f"\n[4/6] Compressing video (target: {args.target_size} MB)...")
        compress_video(video_path, compressed_video, args.target_size, srt_files, source_lang, target_langs, args)

    # Delete monolingual SRT files (always, keeping only the combined/merged SRT)
    print("\n[5/6] Deleting monolingual SRT files...")
    for srt_file in [source_srt] + list(target_srts.values()):
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
2. **📜 Subtitles**: [`{combined_srt.name}`](file://{combined_srt.absolute()})

---

## 🤖 Dual AI Engine Stack

- **ASR Model**: `mlx-whisper` (`{args.model}`) on Apple Silicon GPU
- **Translation Engine**: `{getattr(args, '_backend', 'local')}`
- **Verification Engine**: `Qwen2.5-14B-Instruct-4bit` (MLX Metal GPU)
"""

    try:
        md_report_path.write_text(md_content, encoding="utf-8")
    except Exception as e:
        print(f"  [WARNING] Could not write report file: {e}")

    print(f"\n{'='*60}")
    print(f"  All done! Files created in '{output_dir}/':")
    print(f"     - {combined_srt.name}  (Combined subtitles SRT)")
    print(f"     - {compressed_video.name}   (Video with burned subtitles)")
    print(f"     - {md_report_path.name}     (Markdown summary report)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
