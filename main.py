"""
YouTube Arabic → German Pipeline
=================================

Downloads a YouTube video, transcribes Arabic audio to SRT,
translates to German SRT (verified with Google Translate),
and compresses the video to ~50 MB with burned subtitles.

Uses MLX-Whisper for fast transcription on Apple Silicon GPU.
Resume support: re-run the same command and it detects existing
output files, skipping completed steps automatically.

Usage:
    python main.py <youtube_url>

Example:
    python main.py https://youtu.be/MgxTrPOkhDU
"""

import argparse
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import cast

from dotenv import load_dotenv
from openai import OpenAI

import mlx_whisper
from deep_translator import GoogleTranslator

def parse_args():
    parser = argparse.ArgumentParser(
        description="YouTube Arabic → German: SRT subtitles + AI audio + compressed video."
    )
    parser.add_argument("url", type=str, help="YouTube video URL")
    parser.add_argument(
        "--model",
        type=str,
        default="medium",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="Whisper model size (default: medium)",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=50,
        help="Target compressed video size in MB (default: 50)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory (default: output/)",
    )
    parser.add_argument(
        "--min-quality",
        action="store_true",
        help="Download minimum quality video/audio to save bandwidth and space",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up temporary files (translation_progress.json, translation_temp.srt, temp audio/video) after completion",
    )
    return parser.parse_args()


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
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            f.write(f"{i}\n")
            f.write(f"{format_time(seg['start'])} --> {format_time(seg['end'])}\n")
            f.write(f"{seg['text'].strip()}\n\n")


def read_srt_arabic(srt_path: Path) -> list:
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


def download_youtube(url: str, output_dir: Path, min_quality: bool = False) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    if min_quality:
        # Download minimum quality: worst video + worst audio
        video_format = "worstvideo[ext=mp4]+worstaudio[ext=m4a]/worst[ext=mp4]/worst"
    else:
        # Default: best quality
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


def verify_translations_with_google(segments: list) -> None:
    """Verify German translations using Google Translate (free API).
    
    Re-translates original Arabic text to German and compares with existing translations.
    """
    print(f"\n  Verifying translations with Google Translate (free API)...")
    try:
        translator = GoogleTranslator(source="ar", target="de")
        total = len(segments)
        verified_count = 0
        mismatch_count = 0
        
        for idx, seg in enumerate(segments, 1):
            # Get original Arabic text (stored before translation)
            original_text = seg.get("original_ar", "")
            if not original_text.strip():
                continue
            
            try:
                # Get Google's translation of original Arabic
                google_translation = translator.translate(original_text)
                google_text = google_translation if isinstance(google_translation, str) else str(google_translation)
                
                # Compare with current German translation
                current_text = seg["text"].strip()
                if google_text.strip() != current_text:
                    mismatch_count += 1
                    print(f"  [VERIFY] Segment {idx}: Google suggests different translation")
                    print(f"           Current: {current_text[:60]}...")
                    print(f"           Google:  {google_text[:60]}...")
                else:
                    verified_count += 1
                    print(f"  [VERIFY] Segment {idx}: ✓ Translation verified")
            except Exception as e:
                print(f"  [VERIFY] Segment {idx}: Error - {e}")
            
            if idx % 10 == 0 or idx == total:
                print(f"  Verification progress: {idx}/{total} ({idx*100//total}%)")
        
        print(f"  Google verification completed: {verified_count} verified, {mismatch_count} mismatches")
    except Exception as e:
        print(f"  [WARNING] Google verification failed: {e}")


def translate_segments(segments: list, output_dir: Path | None = None) -> list:
    """Translate each segment's text from Arabic to German.

    Supports resume capability - if interrupted, re-run to continue from last checkpoint.
    Live backup: saves translation after each segment completes.
    
    Uses best available translation backend in priority order:
    1. OpenAI GPT-4 (best quality, requires OPENAI_API_KEY) - parallel bulk translation
    2. Google Translate (free fallback)
    """
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()

    # Store original Arabic text before translation for verification
    for seg in segments:
        if "original_ar" not in seg:
            seg["original_ar"] = seg["text"]

    # Priority 1: OpenAI GPT-4 (best quality for Arabic→German) - PARALLEL BULK MODE
    if openai_key:
        print(f"  Translating {len(segments)} segments Arabic → German (OpenAI GPT-4 - ultra-fast mode) ...")
        try:
            client = OpenAI(api_key=openai_key)
            total = len(segments)
            batch_size = 20  # Larger batches
            max_workers = 10  # More parallel workers
            
            # Resume support: mark segments already translated
            translated_indices = set()
            if output_dir:
                progress_file = output_dir / "translation_progress.json"
                if progress_file.exists():
                    with open(progress_file) as f:
                        translated_indices = set(json.load(f))
                    if translated_indices:
                        print(f"  Resuming: {len(translated_indices)} segments already translated")
            
            def save_progress():
                """Save translated indices to progress file."""
                if output_dir:
                    progress_file = output_dir / "translation_progress.json"
                    with open(progress_file, "w") as f:
                        json.dump(list(translated_indices), f)
            
            def save_live_backup(batch_end: int):
                """Save translated segments to SRT file immediately after translation."""
                if not output_dir:
                    return
                
                # Save partial German SRT with all translated segments so far
                temp_srt = output_dir / "translation_temp.srt"
                try:
                    print(f"  [DEBUG] Saving live backup to: {temp_srt.absolute()}")
                    with open(temp_srt, "w", encoding="utf-8") as f:
                        translated_count = 0
                        for i, seg in enumerate(segments[:batch_end], start=1):
                            if seg["text"].strip():
                                f.write(f"{i}\n")
                                f.write(f"{format_time(seg['start'])} --> {format_time(seg['end'])}\n")
                                f.write(f"{seg['text'].strip()}\n\n")
                                translated_count += 1
                    print(f"  [DEBUG] Live backup saved: {translated_count} segments")
                except Exception as e:
                    print(f"  [WARNING] Could not save live backup: {e}")
            
            def translate_batch(batch_start: int, batch_end: int) -> int:
                """Translate a batch of segments."""
                # Skip already translated segments
                if any(i in translated_indices for i in range(batch_start, batch_end)):
                    return batch_end
                
                batch = segments[batch_start:batch_end]
                combined_text = "\n\n".join([f"[{i+1}] {seg['text'].strip()}" 
                                            for i, seg in enumerate(batch) if seg['text'].strip()])
                
                if not combined_text:
                    # Mark as translated even if empty
                    for i in range(batch_start, batch_end):
                        translated_indices.add(i)
                    save_progress()
                    return batch_end
                
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": f"""You are a professional Arabic to German translator. 
                        Translate each numbered segment from Arabic to natural, fluent German.
                        Keep the same numbering format [N] at the start of each segment.
                        Preserve the meaning and context of each segment.
                        There are {len(batch)} segments to translate."""},
                        {"role": "user", "content": combined_text}
                    ],
                    temperature=0.3,
                )
                
                translated_text = response.choices[0].message.content
                if translated_text:
                    translated_lines = translated_text.strip().split("\n\n")
                    for i, line in enumerate(translated_lines):
                        if i < len(batch) and line.strip():
                            cleaned = line.strip()
                            if cleaned.startswith("["):
                                cleaned = cleaned.split("]", 1)[1].strip()
                            batch[i]["text"] = cleaned
                
                # Mark as translated
                for i in range(batch_start, batch_end):
                    translated_indices.add(i)
                save_progress()
                save_live_backup(batch_end)  # Save immediately
                
                return batch_end
            
            # Process batches in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for batch_start in range(0, total, batch_size):
                    batch_end = min(batch_start + batch_size, total)
                    # Only submit if not already translated
                    if not any(i in translated_indices for i in range(batch_start, batch_end)):
                        futures.append(executor.submit(translate_batch, batch_start, batch_end))
                
                # Wait for completion and show progress
                completed = 0
                for future in as_completed(futures):
                    batch_end = future.result()
                    completed = max(completed, batch_end)
                    if completed % 10 == 0 or completed == total:
                        print(f"  Progress: {completed}/{total} ({completed*100//total}%)")
            
            print(f"  OpenAI ultra-fast translation completed successfully")
            return segments
        except Exception as e:
            print(f"  OpenAI error ({type(e).__name__}): {e}")
            print("  Falling back to Google Translate...")

    # Priority 2: Google Translate (free fallback)
    print(f"  Translating {len(segments)} segments Arabic → German (Google Translate — free, no key needed) ...")
    print(f"  Tip: Set OPENAI_API_KEY for better quality translations.")
    translator = GoogleTranslator(source="ar", target="de")
    total = len(segments)
    
    # Resume support for Google Translate
    translated_indices = set()
    if output_dir:
        progress_file = output_dir / "translation_progress.json"
        if progress_file.exists():
            with open(progress_file) as f:
                translated_indices = set(json.load(f))
            if translated_indices:
                print(f"  Resuming: {len(translated_indices)} segments already translated")
    
    for idx, seg in enumerate(segments, 1):
        # Skip already translated
        if idx - 1 in translated_indices:
            continue
            
        text = seg["text"].strip()
        if text:
            result = translator.translate(text)
            seg["text"] = result if isinstance(result, str) else str(result)
        
        # Mark as translated and save
        translated_indices.add(idx - 1)
        if output_dir:
            progress_file = output_dir / "translation_progress.json"
            with open(progress_file, "w") as f:
                json.dump(list(translated_indices), f)
            
            # Save live backup
            temp_srt = output_dir / "translation_temp.srt"
            with open(temp_srt, "w", encoding="utf-8") as f:
                for i, s in enumerate(segments[:idx], start=1):
                    if s["text"].strip():
                        f.write(f"{i}\n")
                        f.write(f"{format_time(s['start'])} --> {format_time(s['end'])}\n")
                        f.write(f"{s['text'].strip()}\n\n")
        
        if idx % 5 == 0 or idx == total:
            print(f"  Progress: {idx}/{total} ({idx*100//total}%)")
    return segments


def cleanup_temp_files(output_dir: Path, base_name: str, keep_compressed: bool = True) -> None:
    """Clean up temporary files after pipeline completion."""
    temp_files = [
        output_dir / "translation_progress.json",
        output_dir / "translation_temp.srt",
    ]
    
    # Also clean up the original downloaded video/audio files (video_id prefixed)
    for pattern in ["*_video.mp4", "*.mp3"]:
        for f in output_dir.glob(pattern):
            # Don't delete the compressed video or the final output files
            if not f.name.startswith(base_name):
                temp_files.append(f)
    
    for temp_file in temp_files:
        if temp_file.exists():
            try:
                temp_file.unlink()
                print(f"  Cleaned up: {temp_file.name}")
            except Exception as e:
                print(f"  [WARNING] Could not delete {temp_file.name}: {e}")


def compress_video(video_path: Path, output_path: Path, target_mb: int, arabic_srt: Path | None = None, german_srt: Path | None = None) -> None:
    """Compress video and burn subtitles using ffmpeg with progress bar."""
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
            [
                "ffprobe", "-v", "error", "-show_entries",
                "format=duration", "-of",
                "default=noprint_wrappers=1:nokey=1", str(video_path),
            ],
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
    has_subtitles = False
    if german_srt and german_srt.exists():
        cmd.extend(["-i", str(german_srt)])
        has_subtitles = True
    if arabic_srt and arabic_srt.exists():
        cmd.extend(["-i", str(arabic_srt)])
        has_subtitles = True
    
    # Build filter for burning subtitles
    if has_subtitles and german_srt and german_srt.exists() and arabic_srt and arabic_srt.exists():
        # Burn both subtitles - German on top, Arabic on bottom
        cmd.extend([
            "-filter_complex", 
            f"[0:v]subtitles={str(german_srt)}:force_style='Alignment=Top,FontSize=24,PrimaryColour=&HFFFFFF&'[v1];"
            f"[v1]subtitles={str(arabic_srt)}:force_style='Alignment=Bottom,FontSize=24,PrimaryColour=&HFFFFFF&'[vout]",
            "-map", "[vout]",
            "-map", "0:a?",
        ])
    elif has_subtitles and german_srt and german_srt.exists():
        cmd.extend(["-vf", f"subtitles={str(german_srt)}"])
    elif has_subtitles and arabic_srt and arabic_srt.exists():
        cmd.extend(["-vf", f"subtitles={str(arabic_srt)}"])
    
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
    
    # Parse ffmpeg progress from stderr
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
    print()  # New line after progress

    final_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Compressed video size: {final_mb:.1f} MB")


def main():
    load_dotenv()
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  YouTube → Arabic/German Pipeline")
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
        print("[0/5] Naming output files...")
        base_name = prompt_output_name(args.url)
        print(f"       Base name: {base_name}")

    # Define output paths early
    arabic_srt = output_dir / f"{base_name}_ar.srt"
    german_srt = output_dir / f"{base_name}_de.srt"
    compressed_video = output_dir / f"{base_name}_compressed.mp4"

    # Step 1: Download (skip if video+audio exist)
    mp4_files = sorted(output_dir.glob("*_video.mp4"))
    mp3_files = sorted(output_dir.glob("*.mp3"))
    video_path = mp4_files[0] if mp4_files else None
    audio_path = mp3_files[0] if mp3_files else None

    if video_path and audio_path:
        print("\n[1/5] Video & audio already downloaded")
        print(f"       Video: {video_path.name}")
        print(f"       Audio: {audio_path.name}")
    else:
        print("\n[1/5] Downloading video & audio...")
        media = download_youtube(args.url, output_dir, args.min_quality)
        video_path = media["video"]
        audio_path = media["audio"]
        print(f"       Video: {video_path.name}")
        print(f"       Audio: {audio_path.name}")

    # Step 2: Transcribe (skip if Arabic SRT exists)
    if arabic_srt.exists():
        print(f"\n[2/5] Already transcribed — loading from {arabic_srt.name}")
        segments = read_srt_arabic(arabic_srt)
        print(f"       {len(segments)} segments loaded")
    else:
        print(f"\n[2/5] Transcribing Arabic (model: {args.model})...")
        segments = transcribe_arabic(audio_path, args.model)
        write_srt(segments, arabic_srt)
        print(f"       Arabic SRT: {arabic_srt.name} ({len(segments)} segments)")

    # Step 3: Translate (skip if German SRT exists)
    if german_srt.exists():
        print(f"\n[3/5] Already translated — loading from {german_srt.name}")
        # Load Arabic SRT for verification
        arabic_segments = read_srt_arabic(arabic_srt) if arabic_srt.exists() else []
        segments = read_srt_arabic(german_srt)
        # Store original Arabic texts for verification
        for i, seg in enumerate(segments):
            if i < len(arabic_segments):
                seg["original_ar"] = arabic_segments[i]["text"]
        print(f"       {len(segments)} segments loaded")
        
        # Verify existing translations with Google Translate (free API)
        verify_translations_with_google(segments)
    else:
        print(f"\n[3/5] Translating to German...")
        segments = translate_segments(segments, output_dir)
        write_srt(segments, german_srt)
        print(f"       German SRT: {german_srt.name}")
        
        # Verify translations with Google Translate (free API)
        verify_translations_with_google(segments)

    # Step 4: Compress video (skip if exists)
    if compressed_video.exists():
        print(f"\n[4/5] Compressed video already exists: {compressed_video.name}")
    else:
        print(f"\n[4/5] Compressing video (target: {args.target_size} MB)...")
        compress_video(video_path, compressed_video, args.target_size, arabic_srt, german_srt)

    # Step 5: Clean up temporary files if requested
    if args.cleanup:
        print(f"\n[5/5] Cleaning up temporary files...")
        cleanup_temp_files(output_dir, base_name)

    print(f"\n{'='*60}")
    print(f"  All done! Files in '{output_dir}/':")
    print(f"     - {arabic_srt.name}  (Arabic subtitles)")
    print(f"     - {german_srt.name}   (German subtitles)")
    print(f"     - {compressed_video.name}  (compressed video with burned subtitles)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
