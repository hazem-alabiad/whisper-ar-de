"""E-Book Exporter module for whisper-tools.

Generates publisher-quality PDF and digital EPUB e-books from translated text/markdown.
"""

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Try importing PDF rendering engine (reportlab)
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch  # noqa: F401
    from reportlab.pdfgen import canvas
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    _REPORTLAB_AVAILABLE = True
except ImportError:
    _REPORTLAB_AVAILABLE = False

# Try importing EPUB engine (ebooklib & markdown)
try:
    import ebooklib  # noqa: F401
    import markdown
    from ebooklib import epub

    _EPUB_AVAILABLE = True
except ImportError:
    _EPUB_AVAILABLE = False


class NumberedCanvas(canvas.Canvas):
    """Canvas class to generate 'Page X of Y' and running headers for PDF books."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []
        self._page_count = 0

    def showPage(self):
        self._page_count += 1
        self._saved_page_states.append((self._page_count, dict(self.__dict__)))
        super().showPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for page_number, state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(page_number, num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_number: int, page_count: int):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#666666"))

        # Skip running header/footer on first cover/title page
        if page_number > 1:
            # Header line
            self.setStrokeColor(colors.HexColor("#DDDDDD"))
            self.setLineWidth(0.5)
            self.line(54, letter[1] - 40, letter[0] - 54, letter[1] - 40)
            self.drawString(
                54, letter[1] - 35, "Whisper-Tools AI E-Book Translation Edition"
            )

            # Footer line & page number
            self.line(54, 45, letter[0] - 54, 45)
            page_text = f"Page {page_number} of {page_count}"
            self.drawRightString(letter[0] - 54, 30, page_text)

        self.restoreState()


def export_to_pdf(
    content: str,
    output_path: str | Path,
    title: str = "Translated Book",
    author: str = "Whisper-Tools AI",
) -> bool:
    """Renders text or Markdown content into a publisher-quality PDF document.

    Args:
        content: Translated text or Markdown content.
        output_path: Output file path (.pdf).
        title: Book title.
        author: Author name.

    Returns:
        True if export succeeded, False otherwise.
    """
    if not _REPORTLAB_AVAILABLE:
        print("  [WARNING] reportlab library not found. Cannot export PDF.")
        return False

    try:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        doc = SimpleDocTemplate(
            str(out_path),
            pagesize=letter,
            leftMargin=54,  # 0.75 in
            rightMargin=54,
            topMargin=54,
            bottomMargin=54,
        )

        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "BookTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=28,
            textColor=colors.HexColor("#1A252F"),
            alignment=1,  # Center
            spaceAfter=20,
        )

        author_style = ParagraphStyle(
            "BookAuthor",
            parent=styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=12,
            leading=14,
            textColor=colors.HexColor("#555555"),
            alignment=1,
            spaceAfter=30,
        )

        h1_style = ParagraphStyle(
            "BookH1",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#2C3E50"),
            spaceBefore=18,
            spaceAfter=10,
        )

        h2_style = ParagraphStyle(
            "BookH2",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#34495E"),
            spaceBefore=14,
            spaceAfter=8,
        )

        body_style = ParagraphStyle(
            "BookBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=15,
            textColor=colors.HexColor("#222222"),
            spaceBefore=4,
            spaceAfter=8,
        )

        story = []

        # Add Title & Author Cover Block
        story.append(Spacer(1, 20))
        story.append(Paragraph(title, title_style))
        story.append(Paragraph(f"Translated by {author}", author_style))
        story.append(Spacer(1, 15))

        # Split content into paragraphs/lines
        lines = content.splitlines()
        current_para = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if current_para:
                    p_text = " ".join(current_para)
                    story.append(Paragraph(p_text, body_style))
                    current_para = []
                continue

            if stripped.startswith("# "):
                if current_para:
                    story.append(Paragraph(" ".join(current_para), body_style))
                    current_para = []
                story.append(Paragraph(stripped[2:].strip(), h1_style))
            elif stripped.startswith("## "):
                if current_para:
                    story.append(Paragraph(" ".join(current_para), body_style))
                    current_para = []
                story.append(Paragraph(stripped[3:].strip(), h2_style))
            elif stripped.startswith("### "):
                if current_para:
                    story.append(Paragraph(" ".join(current_para), body_style))
                    current_para = []
                story.append(Paragraph(stripped[4:].strip(), h2_style))
            else:
                current_para.append(stripped)

        if current_para:
            story.append(Paragraph(" ".join(current_para), body_style))

        doc.build(story, canvasmaker=NumberedCanvas)
        print(f"  [SUCCESS] Created PDF e-book: {out_path.name}")
        return True
    except Exception as e:
        print(f"  [ERROR] PDF export failed: {e}")
        return False


def export_to_epub(
    content: str,
    output_path: str | Path,
    title: str = "Translated Book",
    author: str = "Whisper-Tools AI",
    language: str = "en",
) -> bool:
    """Renders Markdown/text content into a digital EPUB e-book.

    Args:
        content: Text or Markdown content.
        output_path: Output file path (.epub).
        title: Book title.
        author: Author name.
        language: E-book language code (e.g. 'de', 'en', 'ar').

    Returns:
        True if export succeeded, False otherwise.
    """
    if not _EPUB_AVAILABLE:
        print("  [WARNING] ebooklib/markdown libraries not found. Cannot export EPUB.")
        return False

    try:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        book = epub.EpubBook()

        # Set metadata
        clean_id = re.sub(r"[^a-zA-Z0-9_]", "_", title.lower())
        book.set_identifier(f"whisper_tools_id_{clean_id}")
        book.set_title(title)
        book.set_language(language)
        book.add_author(author)

        # HTML conversion with Markdown extensions
        html_body = markdown.markdown(
            content,
            extensions=["tables", "fenced_code", "nl2br", "toc"],
        )

        # CSS Stylesheet
        css = """
        @charset "utf-8";
        body {
            font-family: Georgia, serif;
            line-height: 1.6;
            margin: 5%;
            color: #111111;
        }
        h1, h2, h3 {
            font-family: sans-serif;
            color: #1a252f;
            margin-top: 1.5em;
            margin-bottom: 0.5em;
            line-height: 1.2;
        }
        h1 { border-bottom: 1px solid #e1e8ed; padding-bottom: 0.3em; font-size: 1.8em; }
        h2 { font-size: 1.4em; }
        p {
            margin-bottom: 1em;
            text-align: justify;
            text-justify: inter-word;
        }
        """

        style_item = epub.EpubItem(
            uid="style_nav",
            file_name="style/default.css",
            media_type="text/css",
            content=css,
        )
        book.add_item(style_item)

        # Main Chapter Document
        chapter = epub.EpubHtml(
            title=title,
            file_name="chap_1.xhtml",
            lang=language,
        )
        chapter.content = f"<html><head><title>{title}</title></head><body><h1>{title}</h1>{html_body}</body></html>"
        chapter.add_item(style_item)

        book.add_item(chapter)

        # TOC & Navigation
        book.toc = [epub.Link("chap_1.xhtml", title, "chap_1")]
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())

        # Spine structure
        book.spine = ["nav", chapter]

        # Write to disk
        epub.write_epub(str(out_path), book, {})
        print(f"  [SUCCESS] Created EPUB e-book: {out_path.name}")
        return True
    except Exception as e:
        print(f"  [ERROR] EPUB export failed: {e}")
        return False
