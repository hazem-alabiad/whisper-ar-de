# 🎬📖 Whisper-Tools: Multi-Lingual Dual AI YouTube & Book Translation Pipeline

> **🤖 Built 100% via Agentic AI & LLMs**
>
> This entire application — from local Apple Silicon Metal GPU memory management, dynamic multi-lingual translation pipelines, interactive terminal UI launchers, to book PDF/DOCX parsing — was architected, refactored, and crafted using state-of-the-art **Agentic AI LLM pair programming**.
>
> - ⚡ **Apple Silicon Metal Acceleration**: Programmatic GPU cache management & memory recycling.
> - 🧠 **Dual AI Stack**: `mlx-whisper` ASR + `Qwen2.5` / `MarianMT` Neural Translation.
> - 🌐 **Universal Language Engine**: Dynamic translation between arbitrary language pairs.

Local Apple Silicon GPU pipeline for downloading, transcribing, translating, verifying, and burning subtitles for YouTube videos & Books. Supports any source and target translation direction dynamically.

---

## 🎬 1. How to Translate YouTube Videos (Online Link)

You can run the pipeline directly via `uv run whisper-tools` or `python main.py`. Pass the **YouTube URL directly**. The tool will automatically download the video, transcribe audio in your source language with `mlx-whisper`, translate into target languages, and burn hard/soft subtitles onto the compressed output video.

```bash
# Basic YouTube Translation (Defaults: --source-lang ar --target-lang de,en)
python main.py "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
# Or using uv run whisper-tools:
uv run whisper-tools "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"

# Translate German audio to Arabic and English
python main.py "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" --source-lang de --target-lang ar,en

# Specify download quality preset upfront (1=Low, 2=Medium, 3=High, 4=Best)
python main.py "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" --quality 2
```

---

## 📚 2. How to Translate Books (Upload / Local File Path)

To translate a book, place or copy your book file (`.pdf`, `.docx`, `.txt`, `.md`) inside your workspace folder or pass it as a positional argument.

```bash
# Option A: Passing positional argument directly
python main.py my_book.pdf

# Option B: Using the --book flag
python main.py --book my_book.pdf

# Option C: Specifying multi-target languages (Arabic to German & English)
python main.py my_book.pdf --source-lang ar --target-lang de,en
```

### Interactive Terminal Flow for Books

When you launch a book translation without CLI targets, an interactive menu will ask for your target language choice:

```text
Select Target Translation Language(s):
  [1] German (Deutsch)
  [2] English
  [3] Both German & English (Dual Translation)
```

### Deliverables Created in `output/books/<title>/`

- **`<book>_translated_<lang>.txt` / `.docx`**: Dedicated target language translation documents.
- **`<book>_dual_<source>_<targets>.md`**: Side-by-side multi-lingual Markdown edition.

---

## ⚙️ 3. Resource Management & GPU Options

To prevent system-wide lag and WindowServer freezes on macOS, resource constraints are enabled by default:

- **Sequential Local Inference**: Local GPU execution (MLX Qwen / MarianMT) is restricted to `max_workers=1`.
- **Active Memory Recycling**: Whisper and translation models are actively unloaded from Unified Memory at stage boundaries.
- **Metal Cache Limits**: MLX cache size is programmatically capped at 2GB.

### Running with Cloud-Only Fallback (Low Resources)

If you want to bypass local neural models completely to save battery or keep your local GPU free, pass the `--no-local-translate` flag. This runs translations in parallel (`max_workers=3`) using the cloud-based Google Translate API.

```bash
uv run whisper-tools "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" --no-local-translate
```

---

## 🖥️ 4. Interactive Terminal Mode Launcher

If you launch `whisper-tools` without arguments, it will automatically open an interactive menu:

```bash
uv run whisper-tools
```

### Interactive Menu Capabilities

1. **🎬 YouTube Video Setup**: Enter video URL, select source/target languages, choose download quality presets, select Whisper ASR model size (`base`, `small`, `medium`, `large`), and select Local LLM model (`Qwen2.5-3B-8bit` / `Qwen2.5-7B-4bit`).
2. **📚 Book Translation Setup**: Automatically scans your `books/` workspace folder, lets you select a book by number or enter a custom path, select target languages, and pick Local LLM models.
3. **📖 Flags & Options Guide**: Display an in-terminal cheat sheet detailing all available CLI parameters.
4. **📥 Interactive Model Pre-Downloader**: Interactively select your desired Whisper ASR and Local LLM models to pre-download them into `model_cache/` upfront for fast offline execution.
