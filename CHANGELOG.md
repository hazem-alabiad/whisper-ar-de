# CHANGELOG

All notable changes to **whisper-tools** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-07-24

### Added
- **Strong Local LLM Integration**: Upgraded default verification/translation model to `Qwen2.5-3B-Instruct-4bit` (with support for `Qwen2.5-7B-Instruct-4bit`) powered by `mlx-lm` on Apple Silicon Metal GPU.
- **Dedicated Local Model Cache Directory**: Standardized model storage location in `./model_cache/` (HF/MarianMT models) and `./model_cache/mlx/` for MLX quantized LLMs.
- **Selective Cleanup Utility**: Enhanced `cleanup_temp_files()` to automatically remove intermediate audio files (`.mp3`), uncompressed raw downloads (`*_video.mp4`), temporary translation states (`ytemp.json`), and temporary reports directory while keeping output products (`_compressed.mp4` and combined `.srt`).
- **Interactive & CLI Quality Preset Options**: Added `--quality` flag supporting presets (`1`/`low`, `2`/`medium`, `3`/`high`, `4`/`best`) alongside interactive quality selection for `yt-dlp`.
- **Project Versioning & Changelog**: Created `CHANGELOG.md` and updated `pyproject.toml` to version `1.0.0`.

### Changed
- Refactored `local_translation.py` to ensure local MLX model loading automatically downloads to and resolves from `./model_cache/mlx/`.
- Switched default verification pipeline to pure 100% offline dual local AI execution on Apple Silicon.

### Fixed
- Fixed tuple unpacking mismatch in `mlx_lm.load()`.
- Standardized top-level imports and formatting compliant with PEP 8 and `ruff`.

---

## [0.1.0] - 2026-07-24

### Added
- Initial pipeline with YouTube download, MLX-Whisper transcription, MarianMT translation, and FFmpeg 2-pass video compression.
