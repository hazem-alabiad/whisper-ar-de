"""
YouTube Arabic → German Pipeline
=================================

Downloads a YouTube video, transcribes Arabic audio to SRT,
translates to German SRT, generates German AI speech,
and compresses the video to ~50 MB.

Uses MLX-Whisper for fast transcription on Apple Silicon GPU.
Resume support: re-run the same command and it detects existing
output files, skipping completed steps automatically.

Usage:
    python main.py <youtube_url>

Example:
    python main.py https://youtu.be/MgxTrPOkhDU
"""

import argparse
import os
import re
import subprocess
from pathlib import Path
from typing import cast

from dotenv import load_dotenv

import deepl
import mlx_whisper
from deep_translator import GoogleTranslator
from gtts import gTTS


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


def download_youtube(url: str, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    video_template = output_dir / "%(id)s_video.%(ext)s"
    subprocess.run(
        ["yt-dlp", "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
         "-o", str(video_template), url],
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


def translate_segments(segments: list) -> list:
    """Translate each segment's text from Arabic to German.

    Uses DeepL if DEEPL_AUTH_KEY is set (better quality),
    otherwise falls back to free Google Translate.
    """
    auth_key = os.getenv("DEEPL_AUTH_KEY", "").strip()
    use_deepl = bool(auth_key)

    if use_deepl:
        print(f"  Translating {len(segments)} segments Arabic → German (DeepL) ...")
        try:
            translator = deepl.Translator(auth_key)
            # Test the API key with a simple translation
            test_result = translator.translate_text("test", source_lang="AR", target_lang="DE")
            print(f"  DeepL API test successful")
            for seg in segments:
                text = seg["text"].strip()
                if text:
                    result = translator.translate_text(text, source_lang="AR", target_lang="DE")
                    translation = result[0] if isinstance(result, list) else result
                    seg["text"] = translation.text
        except deepl.exceptions.AuthorizationException as e:
            print(f"  DeepL auth error details: {e}")
            print("  Falling back to Google Translate.")
            translator = GoogleTranslator(source="ar", target="de")
            for seg in segments:
                text = seg["text"].strip()
                if text:
                    result = translator.translate(text)
                    seg["text"] = result if isinstance(result, str) else str(result)
        except Exception as e:
            print(f"  DeepL error ({type(e).__name__}): {e}")
            print("  Falling back to Google Translate.")
            translator = GoogleTranslator(source="ar", target="de")
            for seg in segments:
                text = seg["text"].strip()
                if text:
                    result = translator.translate(text)
                    seg["text"] = result if isinstance(result, str) else str(result)
    else:
        print(f"  Translating {len(segments)} segments Arabic → German (Google Translate — free, no key needed) ...")
        print(f"  Tip: Set DEEPL_AUTH_KEY environment variable for better quality translations.")
        translator = GoogleTranslator(source="ar", target="de")
        for seg in segments:
            text = seg["text"].strip()
            if text:
                result = translator.translate(text)
                seg["text"] = result if isinstance(result, str) else str(result)
    return segments


def generate_german_audio(segments: list, output_path: Path) -> None:
    """Generate German speech audio from translated segments using gTTS."""
    full_text = " ".join(seg["text"] for seg in segments if seg["text"].strip())
    if not full_text:
        print("  Warning: No German text to convert to speech.")
        return

    print(f"  Generating German audio via gTTS ({len(full_text)} chars)...")
    tts = gTTS(text=full_text, lang="de", slow=False)
    tts.save(str(output_path))


def compress_video(video_path: Path, output_path: Path, target_mb: int) -> None:
    """Compress video to target size using ffmpeg."""
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
    print(f"  Compressing video (this may take a while)...")

    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-c:v", "libx264", "-b:v", f"{video_bitrate}k",
            "-c:a", "aac", "-b:a", f"{audio_bitrate}k",
            "-movflags", "+faststart",
            str(output_path),
        ],
        check=True, capture_output=True, text=True,
    )

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
    german_audio = output_dir / f"{base_name}_de.mp3"
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
        media = download_youtube(args.url, output_dir)
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
        segments = read_srt_arabic(german_srt)
        print(f"       {len(segments)} segments loaded")
    else:
        print(f"\n[3/5] Translating to German...")
        segments = translate_segments(segments)
        write_srt(segments, german_srt)
        print(f"       German SRT: {german_srt.name}")

    # Step 4: German audio (skip if exists)
    if german_audio.exists():
        print(f"\n[4/5] German audio already generated: {german_audio.name}")
    else:
        print(f"\n[4/5] Generating German audio via AI...")
        generate_german_audio(segments, german_audio)
        print(f"       German audio: {german_audio.name}")

    # Step 5: Compress video (skip if exists)
    if compressed_video.exists():
        print(f"\n[5/5] Compressed video already exists: {compressed_video.name}")
    else:
        print(f"\n[5/5] Compressing video (target: {args.target_size} MB)...")
        compress_video(video_path, compressed_video, args.target_size)

    print(f"\n{'='*60}")
    print(f"  All done! Files in '{output_dir}/':")
    print(f"     - {arabic_srt.name}  (Arabic subtitles)")
    print(f"     - {german_srt.name}   (German subtitles)")
    print(f"     - {german_audio.name}  (German AI speech)")
    print(f"     - {compressed_video.name}  (compressed video)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
