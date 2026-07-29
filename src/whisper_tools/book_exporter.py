"""E-Book Exporter module for whisper-tools.

Generates publisher-quality PDF and digital EPUB e-books from translated text/markdown.
Full Arabic (RTL) support across PDF, EPUB, and Markdown output formats.
"""

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Arabic text utilities ─────────────────────────────────────────────────────

# Unicode ranges covering Arabic, Arabic Supplement, Arabic Extended-A/B,
# Arabic Presentation Forms-A/B, and Arabic Mathematical Alphabetic Symbols.
_ARABIC_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)

# Pattern that matches a dual-book segment label like "**AR (Source)**:" or "**DE**:"
# Group 1 = full label prefix (e.g. "**AR (Source)**: ")
# Group 2 = everything after the label
_LABEL_RE = re.compile(r"^(\*\*[A-Za-z ]+(?:\([^)]*\))?\*\*:\s*)(.*)", re.DOTALL)


def _has_arabic(text: str) -> bool:
    """Return True if *text* contains any Arabic-script codepoint."""
    return bool(_ARABIC_RE.search(text))


# ── arabic-reshaper + python-bidi (required for PDF rendering) ────────────────
try:
    import arabic_reshaper  # type: ignore[import-untyped]
    from bidi.algorithm import get_display  # type: ignore[import-untyped]

    _ARABIC_SHAPING_AVAILABLE = True
except ImportError:
    _ARABIC_SHAPING_AVAILABLE = False
    logger.debug("arabic-reshaper / python-bidi not installed; Arabic PDF glyphs may be broken.")


def _shape_arabic(text: str) -> str:
    """Reshape + reorder Arabic text for correct PDF rendering.

    arabic_reshaper joins isolated letter forms into their contextual connected
    forms.  get_display reverses the visual order so that ReportLab (LTR-only)
    renders the glyphs in the correct right-to-left reading sequence.
    """
    if not _ARABIC_SHAPING_AVAILABLE:
        return text
    reshaped = arabic_reshaper.reshape(text)
    result = get_display(reshaped)
    # get_display may return bytes in some library versions – normalise to str.
    return result.decode("utf-8") if isinstance(result, bytes) else result


def _shape_mixed_line(text: str) -> tuple[str | None, str]:
    """Split a dual-book label line into (latin_label, arabic_body).

    For lines like "**AR (Source)**: <arabic text>", returns the label
    as plain text and the reshaped Arabic body separately so that each
    part can be rendered with the correct font in ReportLab.

    For lines without a recognised label, returns (None, shaped_text).
    """
    m = _LABEL_RE.match(text)
    if m:
        label = m.group(1)
        body = m.group(2).strip()
        if _has_arabic(body):
            return label, _shape_arabic(body)
        return label, body
    if _has_arabic(text):
        return None, _shape_arabic(text)
    return None, text


# ── PDF rendering engine (reportlab) ─────────────────────────────────────────

# Preferred system Arabic fonts (macOS + common Linux paths).
_ARABIC_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Al Nile.ttc",
    "/System/Library/Fonts/GeezaPro.ttc",
    "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
    "/usr/share/fonts/arabic/Amiri-Regular.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
]
_ARABIC_FONT_PATH: str | None = next(
    (p for p in _ARABIC_FONT_CANDIDATES if Path(p).exists()), None
)
_ARABIC_FONT_NAME = "ArabicFont"

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch  # noqa: F401
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    # Register Arabic TTF font once at import time.
    if _ARABIC_FONT_PATH:
        try:
            pdfmetrics.registerFont(TTFont(_ARABIC_FONT_NAME, _ARABIC_FONT_PATH))
            _ARABIC_FONT_REGISTERED = True
            logger.debug("Registered Arabic PDF font from %s", _ARABIC_FONT_PATH)
        except Exception as _font_err:
            logger.warning("Could not register Arabic font %s: %s", _ARABIC_FONT_PATH, _font_err)
            _ARABIC_FONT_REGISTERED = False
    else:
        _ARABIC_FONT_REGISTERED = False
        logger.warning("No system Arabic TTF font found; Arabic PDF glyphs may render as boxes.")

    _REPORTLAB_AVAILABLE = True
except ImportError:
    _REPORTLAB_AVAILABLE = False
    _ARABIC_FONT_REGISTERED = False

# ── EPUB engine (ebooklib & markdown) ────────────────────────────────────────
try:
    import ebooklib  # noqa: F401
    import markdown
    from ebooklib import epub

    _EPUB_AVAILABLE = True
except ImportError:
    _EPUB_AVAILABLE = False


# ── NumberedCanvas ────────────────────────────────────────────────────────────

class NumberedCanvas(canvas.Canvas):
    """Canvas class to generate 'Page X of Y' and running headers for PDF books."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states: list[tuple[int, dict]] = []
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

        if page_number > 1:
            self.setStrokeColor(colors.HexColor("#DDDDDD"))
            self.setLineWidth(0.5)
            self.line(54, letter[1] - 40, letter[0] - 54, letter[1] - 40)
            self.drawString(54, letter[1] - 35, "Whisper-Tools AI E-Book Translation Edition")
            self.line(54, 45, letter[0] - 54, 45)
            self.drawRightString(letter[0] - 54, 30, f"Page {page_number} of {page_count}")

        self.restoreState()


# ── PDF export ────────────────────────────────────────────────────────────────

def export_to_pdf(
    content: str,
    output_path: str | Path,
    title: str = "Translated Book",
    author: str = "Whisper-Tools AI",
) -> bool:
    """Render text or Markdown content into a publisher-quality PDF document.

    Handles mixed Arabic/Latin content: Arabic paragraphs are reshaped and
    right-aligned using the registered Arabic TTF font; Latin paragraphs use
    the standard Helvetica body style.

    Args:
        content:     Translated text or Markdown content.
        output_path: Output file path (.pdf).
        title:       Book title.
        author:      Author name.

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
            leftMargin=54,   # 0.75 in
            rightMargin=54,
            topMargin=54,
            bottomMargin=54,
        )

        styles = getSampleStyleSheet()
        ar_font = _ARABIC_FONT_NAME if _ARABIC_FONT_REGISTERED else "Helvetica"

        # ── Styles ────────────────────────────────────────────────────────────
        title_style = ParagraphStyle(
            "BookTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=28,
            textColor=colors.HexColor("#1A252F"),
            alignment=TA_CENTER,
            spaceAfter=20,
        )
        author_style = ParagraphStyle(
            "BookAuthor",
            parent=styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=12,
            leading=14,
            textColor=colors.HexColor("#555555"),
            alignment=TA_CENTER,
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
            alignment=TA_LEFT,
            spaceBefore=4,
            spaceAfter=8,
        )
        # Pure Arabic paragraph: right-aligned, larger font for readability.
        ar_body_style = ParagraphStyle(
            "ArabicBody",
            parent=styles["BodyText"],
            fontName=ar_font,
            fontSize=12,
            leading=18,
            textColor=colors.HexColor("#222222"),
            alignment=TA_RIGHT,
            spaceBefore=4,
            spaceAfter=8,
        )
        # Mixed paragraph: label in Helvetica, Arabic body in Arabic font.
        # ReportLab XML markup is used to switch fonts inline.
        mixed_style = ParagraphStyle(
            "MixedBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=18,
            textColor=colors.HexColor("#222222"),
            alignment=TA_LEFT,
            spaceBefore=4,
            spaceAfter=8,
        )

        story: list = []
        story.append(Spacer(1, 20))
        story.append(Paragraph(title, title_style))
        story.append(Paragraph(f"Translated by {author}", author_style))
        story.append(Spacer(1, 15))

        # ── Content parsing ───────────────────────────────────────────────────
        lines = content.splitlines()
        current_para: list[str] = []

        def _flush_para(buf: list[str]) -> None:
            """Flush the line buffer as an appropriately styled paragraph."""
            if not buf:
                return
            p_text = " ".join(buf)
            buf.clear()
            _emit_paragraph(p_text)

        def _emit_paragraph(text: str) -> None:
            """Add a single paragraph to *story* with the correct style."""
            if not text.strip():
                return

            label, body = _shape_mixed_line(text)

            if label is not None:
                # Dual-language line: render label in Helvetica, body in Arabic font.
                # ReportLab XML: escape < > & in the label; body is already shaped.
                safe_label = label.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                safe_body = body.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                if _ARABIC_FONT_REGISTERED and _has_arabic(body):
                    xml = (
                        f'<font name="Helvetica" size="10">{safe_label}</font>'
                        f'<font name="{_ARABIC_FONT_NAME}" size="12">{safe_body}</font>'
                    )
                else:
                    xml = f"{safe_label}{safe_body}"
                story.append(Paragraph(xml, mixed_style))
            elif _has_arabic(body):
                story.append(Paragraph(body, ar_body_style))
            else:
                safe = body.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                story.append(Paragraph(safe, body_style))

        for line in lines:
            stripped = line.strip()

            if not stripped:
                _flush_para(current_para)
                continue

            if stripped.startswith("# "):
                _flush_para(current_para)
                story.append(Paragraph(stripped[2:].strip(), h1_style))
            elif stripped.startswith("## "):
                _flush_para(current_para)
                story.append(Paragraph(stripped[3:].strip(), h2_style))
            elif stripped.startswith("### "):
                _flush_para(current_para)
                story.append(Paragraph(stripped[4:].strip(), h2_style))
            elif stripped == "---":
                _flush_para(current_para)
                story.append(Spacer(1, 6))
            else:
                # Flush any pending buffer before a label line so each
                # language block stays as its own paragraph.
                if _LABEL_RE.match(stripped) and current_para:
                    _flush_para(current_para)
                current_para.append(stripped)

        _flush_para(current_para)

        doc.build(story, canvasmaker=NumberedCanvas)
        print(f"  [SUCCESS] Created PDF e-book: {out_path.name}")
        return True

    except Exception as e:
        print(f"  [ERROR] PDF export failed: {e}")
        logger.exception("PDF export failed")
        return False


# ── EPUB helpers ──────────────────────────────────────────────────────────────

def _add_arabic_attrs_to_html(html: str) -> str:
    """Post-process HTML from markdown to add dir/lang attributes to Arabic paragraphs.

    For each <p>…</p> block that contains Arabic codepoints, this adds
    dir="rtl" lang="ar" xml:lang="ar" so EPUB readers render the text RTL.
    """
    def _replace_p(m: re.Match) -> str:  # type: ignore[type-arg]
        inner = m.group(1)
        if _has_arabic(inner):
            return f'<p dir="rtl" lang="ar" xml:lang="ar">{inner}</p>'
        return m.group(0)

    return re.sub(r"<p>(.*?)</p>", _replace_p, html, flags=re.DOTALL)


# ── EPUB export ───────────────────────────────────────────────────────────────

def export_to_epub(
    content: str,
    output_path: str | Path,
    title: str = "Translated Book",
    author: str = "Whisper-Tools AI",
    language: str = "en",
) -> bool:
    """Render Markdown/text content into a digital EPUB e-book.

    Arabic paragraphs receive dir="rtl" lang="ar" attributes in the HTML and
    are styled with an embedded Arabic-capable CSS font stack.

    Args:
        content:     Text or Markdown content.
        output_path: Output file path (.epub).
        title:       Book title.
        author:      Author name.
        language:    E-book language code (e.g. 'de', 'en', 'ar').

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
        clean_id = re.sub(r"[^a-zA-Z0-9_]", "_", title.lower())
        book.set_identifier(f"whisper_tools_id_{clean_id}")
        book.set_title(title)
        book.set_language(language)
        book.add_author(author)

        # Convert Markdown → HTML, then fix Arabic paragraph attributes.
        html_body = markdown.markdown(
            content,
            extensions=["tables", "fenced_code", "nl2br", "toc"],
        )
        html_body = _add_arabic_attrs_to_html(html_body)

        css = """\
@charset "utf-8";

body {
    font-family: Georgia, "Times New Roman", serif;
    line-height: 1.7;
    margin: 5%;
    color: #111111;
}

h1, h2, h3 {
    font-family: "Helvetica Neue", Arial, sans-serif;
    color: #1a252f;
    margin-top: 1.5em;
    margin-bottom: 0.5em;
    line-height: 1.2;
}
h1 {
    border-bottom: 1px solid #e1e8ed;
    padding-bottom: 0.3em;
    font-size: 1.8em;
}
h2 { font-size: 1.4em; }

p {
    margin-bottom: 1em;
    text-align: justify;
    text-justify: inter-word;
}

/* ── Arabic / RTL paragraphs ── */
p[lang="ar"],
p[xml\\:lang="ar"],
[dir="rtl"] {
    font-family:
        "Geeza Pro",          /* macOS / iOS system Arabic */
        "Al Nile",            /* macOS supplemental */
        "Traditional Arabic", /* Windows */
        "Noto Naskh Arabic",  /* Linux / Android */
        "Amiri",              /* open-source Naskh */
        serif;
    direction: rtl;
    text-align: right;
    unicode-bidi: embed;
    font-size: 1.15em;
    line-height: 1.9;
}
"""

        style_item = epub.EpubItem(
            uid="style_nav",
            file_name="style/default.css",
            media_type="text/css",
            content=css,
        )
        book.add_item(style_item)

        chapter = epub.EpubHtml(
            title=title,
            file_name="chap_1.xhtml",
            lang=language,
        )
        # ebooklib wraps content in html/head/body internally; supply only the
        # body fragment.  Adding xml:lang via the lang attribute on EpubHtml
        # ensures correct EPUB3 language tagging.
        chapter.content = f"<h1>{title}</h1>{html_body}"
        chapter.add_item(style_item)

        book.add_item(chapter)
        book.toc = [epub.Link("chap_1.xhtml", title, "chap_1")]
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        book.spine = ["nav", chapter]

        epub.write_epub(str(out_path), book, {})
        print(f"  [SUCCESS] Created EPUB e-book: {out_path.name}")
        return True

    except Exception as e:
        print(f"  [ERROR] EPUB export failed: {e}")
        logger.exception("EPUB export failed")
        return False


# ── Markdown helpers ──────────────────────────────────────────────────────────

def enrich_markdown_arabic(content: str) -> str:
    """Return *content* with Arabic paragraphs wrapped in HTML RTL blocks.

    Standard Markdown has no built-in RTL directive.  Most modern renderers
    (GitHub, Obsidian, VS Code Preview, Typora, etc.) honour inline HTML, so
    wrapping Arabic-only segments in ``<div dir="rtl">`` makes them display
    correctly without breaking the Markdown structure for Latin paragraphs.

    The function is idempotent – already-wrapped blocks are left untouched.
    """
    out_lines: list[str] = []
    in_arabic_block = False

    for line in content.splitlines():
        stripped = line.strip()

        # Don't re-wrap already-wrapped blocks or HTML/code lines.
        if stripped.startswith("<") or stripped.startswith("```"):
            if in_arabic_block:
                out_lines.append("</div>\n")
                in_arabic_block = False
            out_lines.append(line)
            continue

        if _has_arabic(stripped) and stripped:
            if not in_arabic_block:
                out_lines.append('\n<div dir="rtl" lang="ar">\n')
                in_arabic_block = True
            out_lines.append(line)
        else:
            if in_arabic_block:
                out_lines.append("\n</div>\n")
                in_arabic_block = False
            out_lines.append(line)

    if in_arabic_block:
        out_lines.append("\n</div>\n")

    return "\n".join(out_lines)
