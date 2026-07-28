"""
Book Translation Module (Arabic → German / English / Both)
==========================================================

Supports translating book files (.txt, .pdf, .docx) using the 3-Tier
fallback translation pipeline with batching, live JSON backup, and resume.
Outputs:
  - Translated Text / Markdown book files
  - Word (.docx) documents (if python-docx is installed)
  - Full structured JSON backup
"""

import json
from pathlib import Path
from typing import List, Dict, Any


def extract_text_from_book(file_path: Path) -> str:
    """Extract full raw text from PDF, DOCX, or TXT book file."""
    ext = file_path.suffix.lower()
    
    if ext == ".txt" or ext == ".md":
        return file_path.read_text(encoding="utf-8")
        
    elif ext == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(str(file_path))
            text_parts = []
            for i, page in enumerate(reader.pages):
                p_text = page.extract_text()
                if p_text:
                    text_parts.append(p_text)
            return "\n\n".join(text_parts)
        except ImportError:
            raise RuntimeError("pypdf is required to read PDF books. Install with: uv add pypdf")
            
    elif ext == ".docx":
        try:
            import docx
            doc = docx.Document(str(file_path))
            return "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        except ImportError:
            raise RuntimeError("python-docx is required to read DOCX books. Install with: uv add python-docx")
            
    else:
        raise ValueError(f"Unsupported book format: {ext}. Supported formats: .txt, .pdf, .docx, .md")


def chunk_text_into_paragraphs(text: str, max_chunk_chars: int = 1000) -> List[Dict[str, Any]]:
    """Split book text into coherent paragraph segments for translation."""
    raw_paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for p in raw_paragraphs:
        if current_length + len(p) > max_chunk_chars and current_chunk:
            combined = "\n".join(current_chunk)
            chunks.append({
                "id": len(chunks) + 1,
                "text": combined,
                "original_ar": combined,
            })
            current_chunk = [p]
            current_length = len(p)
        else:
            current_chunk.append(p)
            current_length += len(p)
            
    if current_chunk:
        combined = "\n".join(current_chunk)
        chunks.append({
            "id": len(chunks) + 1,
            "text": combined,
            "original_ar": combined,
        })
        
    return chunks


def translate_book_interactive(book_path: Path, output_dir: Path, translate_segments_fn, args) -> Path:
    """Interactively process and translate a book from Arabic to German, English, or Both with dedicated directory & live backup."""
    base_name = book_path.stem
    book_output_dir = output_dir / "books" / base_name
    book_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"  📚 Book Translation Mode: {book_path.name}")
    print(f"  📁 Dedicated Book Output Directory: {book_output_dir}")
    print(f"{'='*60}")
    
    # 1. Target Language Selection
    print("\n  Select Target Translation Language(s):")
    print("    [1] German (Deutsch)")
    print("    [2] English")
    print("    [3] Both German & English (Dual Translation)")
    lang_choice = input("\n  Enter choice [1-3] (default: 3 Both): ").strip()
    if lang_choice == "1":
        target_mode = "de"
    elif lang_choice == "2":
        target_mode = "en"
    else:
        target_mode = "both"
        
    print(f"  Selected target mode: {target_mode.upper()}")
    
    # 2. Extract & Chunk Text
    print(f"\n[1/3] Extracting text from {book_path.name}...")
    raw_text = extract_text_from_book(book_path)
    segments = chunk_text_into_paragraphs(raw_text)
    print(f"       Extracted {len(segments)} paragraph chunks from book.")
    
    # 3. Translate Segments via Pipeline with dedicated directory for live JSON backups
    print(f"\n[2/3] Translating book paragraphs using 3-Tier AI Engine...")
    translated_segments = translate_segments_fn(segments, book_output_dir, args)
    
    # 4. Save Output Book Files
    print(f"\n[3/3] Saving translated book deliverables...")
    
    # German Output
    if target_mode in ("de", "both"):
        de_txt_path = book_output_dir / f"{base_name}_translated_de.txt"
        de_text = "\n\n".join([s.get("text_de", s.get("text", "")) for s in translated_segments])
        de_txt_path.write_text(de_text, encoding="utf-8")
        print(f"       German Book (.txt): {de_txt_path.name}")
        
        try:
            import docx
            doc = docx.Document()
            doc.add_heading(f"{base_name} - German Translation", 0)
            for s in translated_segments:
                p_text = s.get("text_de", s.get("text", "")).strip()
                if p_text:
                    doc.add_paragraph(p_text)
            de_docx_path = book_output_dir / f"{base_name}_translated_de.docx"
            doc.save(str(de_docx_path))
            print(f"       German Book (.docx): {de_docx_path.name}")
        except Exception:
            pass

    # English Output
    if target_mode in ("en", "both"):
        en_txt_path = book_output_dir / f"{base_name}_translated_en.txt"
        en_text = "\n\n".join([s.get("text_en", "") for s in translated_segments])
        en_txt_path.write_text(en_text, encoding="utf-8")
        print(f"       English Book (.txt): {en_txt_path.name}")
        
        try:
            import docx
            doc = docx.Document()
            doc.add_heading(f"{base_name} - English Translation", 0)
            for s in translated_segments:
                p_text = s.get("text_en", "").strip()
                if p_text:
                    doc.add_paragraph(p_text)
            en_docx_path = book_output_dir / f"{base_name}_translated_en.docx"
            doc.save(str(en_docx_path))
            print(f"       English Book (.docx): {en_docx_path.name}")
        except Exception:
            pass

    # Dual Language Side-by-Side Markdown
    if target_mode == "both":
        dual_md_path = book_output_dir / f"{base_name}_dual_ar_de_en.md"
        md_lines = [f"# 📖 {base_name} - Tri-Lingual Book Edition\n"]
        for s in translated_segments:
            ar = s.get("original_ar", "").strip()
            de = s.get("text_de", "").strip()
            en = s.get("text_en", "").strip()
            md_lines.append(f"### Segment {s['id']}\n")
            md_lines.append(f"**Arabic**: {ar}\n")
            md_lines.append(f"**German**: {de}\n")
            if en:
                md_lines.append(f"**English**: {en}\n")
            md_lines.append("---\n")
        dual_md_path.write_text("\n".join(md_lines), encoding="utf-8")
        print(f"       Tri-Lingual Book (.md): {dual_md_path.name}")

    print(f"\n{'='*60}")
    print(f"  🎉 Book Translation Complete! Output files saved in '{book_output_dir}/'")
    print(f"{'='*60}\n")
    return book_output_dir
