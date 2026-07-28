# 🎬📖 Whisper-Tools: Multi-Lingual Dual AI YouTube & Book Translation Pipeline

> **🤖 Built 100% via Agentic AI & LLMs**
>
> This entire application — from local Apple Silicon Metal GPU memory management, dynamic multi-lingual translation pipelines, interactive terminal UI launchers, to book PDF/DOCX parsing — was architected, refactored, and crafted using state-of-the-art **Agentic AI LLM pair programming**.
>
> - ⚡ **Apple Silicon Metal Acceleration**: Programmatic GPU cache management & memory recycling.
> - 🧠 **Dual AI Stack**: `mlx-whisper` ASR + `Qwen2.5-7B-Instruct-4bit` (Default) / `Qwen2.5-14B` Neural Translation.
> - 🔄 **Dual-LLM 2-Pass Architecture**: Action Pass (Initial Neural Translation) ➔ Verifier Pass (AI Quality & Nuance Refinement).
> - 🌐 **Universal Language Engine**: Dynamic translation between arbitrary language pairs.

Local Apple Silicon GPU pipeline for downloading, transcribing, translating, verifying, and burning subtitles for YouTube videos & Books. Supports any source and target translation direction dynamically.

---

## 🚀 Quick Start & Running Instructions

### 1. Interactive Terminal Launcher (Recommended)

Simply execute `python main.py` or `uv run whisper-tools` without arguments to open the interactive configuration menu:

```bash
# Launch interactive terminal mode
python main.py

# Or via uv script runner
uv run whisper-tools
```

#### Interactive Menu Options:
1. **🎬 Translate YouTube Video**: Step-by-step guidance for video URL, source/target languages, download quality presets, Whisper ASR model size (`base`, `small`, `medium`, `large-v3-turbo-q4`), and Local LLM model choice.
2. **📚 Translate Local Book**: Automatically scans your `books/` workspace folder, lets you select a book by number or path, select target languages, and pick LLM models.
3. **📖 Flags & Options Guide**: Display an in-terminal cheat sheet detailing all available CLI parameters.
4. **📥 Interactively Select & Pre-download Models**: Pre-warm model caches into `model_cache/` upfront for offline/fast usage.

---

### 2. Direct Command Line (CLI) Usage

#### 🎬 YouTube Video Translation
```bash
# Basic YouTube Translation (Arabic -> German & English)
python main.py "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"

# Specify custom source & target languages (e.g. German -> Arabic & English)
python main.py "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" --source-lang de --target-lang ar,en

# Select specific Whisper model size (base, small, medium, large-v3-turbo-q4)
python main.py "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" --model medium
```

#### 📚 Book Translation (.pdf, .docx, .txt, .md)
```bash
# Translate book passing positional file path directly
python main.py my_book.pdf

# Translate book with explicit target languages (Arabic -> German & English)
python main.py my_book.pdf --source-lang ar --target-lang de,en
```

#### ⚡ Low Resource / Cloud-Only Fallback
```bash
# Bypass local GPU models and use parallel cloud Google Translate API
python main.py "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" --no-local-translate
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

- **`<book>_translated_<lang>.pdf`**: Publisher-quality PDF e-book with running headers, footers, & page numbers.
- **`<book>_translated_<lang>.epub`**: Digital EPUB e-book with table of contents for Apple Books, Kindle, and Kobo.
- **`<book>_translated_<lang>.docx` / `.txt`**: Microsoft Word & plain text translation documents.
- **`<book>_dual_<source>_<targets>.md` / `.pdf` / `.epub`**: Side-by-side multi-lingual edition.

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
