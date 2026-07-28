# 🎬📖 Whisper-Tools: Multi-Lingual Dual AI YouTube & Book Translation Pipeline

Local Apple Silicon GPU pipeline for downloading, transcribing, translating, verifying, and burning subtitles for YouTube videos & Books. Supports any source and target translation direction dynamically.

---

## 🎬 1. How to Translate YouTube Videos (Online Link)

Pass the **YouTube URL directly** to `main.py`. The tool will automatically download the video, transcribe audio in your source language with `mlx-whisper`, translate into target languages, and burn hard/soft subtitles onto the compressed output video.

```bash
# Basic YouTube Translation (Defaults: --source-lang ar --target-lang de,en)
uv run python main.py "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"

# Translate German audio to Arabic and English
uv run python main.py "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" --source-lang de --target-lang ar,en

# Specify download quality preset upfront (1=Low, 2=Medium, 3=High, 4=Best)
uv run python main.py "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" --quality 2
```

---

## 📚 2. How to Translate Books (Upload / Local File Path)

To translate a book, place or copy your book file (`.pdf`, `.docx`, `.txt`, `.md`) inside your workspace folder (e.g. `/Users/hazem/ws/whisper-tools/my_book.pdf`) or provide its absolute path.

```bash
# Option A: Translate interactively
uv run python main.py --book my_book.pdf

# Option B: Run via CLI flags (e.g., German to Arabic)
uv run python main.py --book my_book.docx --source-lang de --target-lang ar
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
uv run python main.py "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" --no-local-translate
```
