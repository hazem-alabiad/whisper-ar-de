# 🎬📖 Whisper-Tools: Dual AI YouTube & Book Translation Pipeline

Local Apple Silicon GPU pipeline for downloading, transcribing, translating, verifying, and burning subtitles for YouTube videos & Books from **Arabic → German / English**.

---

## 🎬 1. How to Translate YouTube Videos (Online Link)

Pass the **YouTube URL directly** to `main.py`. The tool will automatically download the video, transcribe Arabic audio with `mlx-whisper`, translate into German & English using `Qwen2.5-14B-Instruct`, and burn hard/soft subtitles onto the compressed output video.

```bash
# Basic YouTube Translation
uv run python main.py "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"

# Specify download quality preset upfront (1=Low, 2=Medium, 3=High, 4=Best)
uv run python main.py "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" --quality 2
```

---

## 📚 2. How to Translate Books (Upload / Local File Path)

To translate a book, place or copy your book file (`.pdf`, `.docx`, `.txt`, `.md`) inside your workspace folder (e.g. `/Users/hazem/ws/whisper-tools/my_book.pdf`) or provide its absolute path.

```bash
# Option A: Using the --book flag
uv run python main.py --book my_book.pdf

# Option B: Passing the file path directly
uv run python main.py my_book.docx
```

### Interactive Terminal Flow for Books:
When you launch a book translation, an interactive menu will ask for your target language choice:
```text
Select Target Translation Language(s):
  [1] German (Deutsch)
  [2] English
  [3] Both German & English (Dual Translation)
```

### Deliverables Created in `output/`:
- **`<book>_translated_de.txt` / `.docx`**: Full German translation document.
- **`<book>_translated_en.txt` / `.docx`**: Full English translation document.
- **`<book>_dual_ar_de_en.md`**: Side-by-side tri-lingual Markdown edition.
