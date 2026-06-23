"""
Word document builder for Multi-Agent Studio.

Takes the final pipeline state and produces a .docx file as bytes.
The bytes are returned directly to Streamlit's download button —
no file is written to disk.

Handles markdown-style headings (## and ###), bullet points (* or -),
and bold markers (**text**) by converting them to proper Word styles.

Color scheme matches the reference documents:
  Title:   #1E3A5F  dark navy,   20pt bold
  H1:      #1E3A5F  dark navy,   16pt bold
  H2:      #2E75B6  medium blue, 13pt bold
  H3:      #C8860A  amber/gold,  12pt bold
  Body:    #1A1A1A  near-black,  11pt
"""

import io
import re
from datetime import date
from docx import Document
from docx.shared import Pt, Inches, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


# ── Color palette ─────────────────────────────────────────────────────────────
NAVY   = RGBColor(0x1E, 0x3A, 0x5F)   # H1 + Title
BLUE   = RGBColor(0x2E, 0x75, 0xB6)   # H2
AMBER  = RGBColor(0xC8, 0x86, 0x0A)   # H3
GREY   = RGBColor(0x88, 0x88, 0x88)   # meta text / footer
BLACK  = RGBColor(0x1A, 0x1A, 0x1A)   # body text
RULE   = RGBColor(0x2E, 0x75, 0xB6)   # horizontal rule under title


def _set_document_styles(doc: Document) -> None:
    """
    Configures base Normal style, margins, and heading styles.
    Heading styles are set on the style objects so every heading
    in the document inherits the right color and size automatically.
    """
    # Base font and margins
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    normal.font.color.rgb = BLACK

    for section in doc.sections:
        section.top_margin    = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin   = Inches(1.2)
        section.right_margin  = Inches(1.2)

    # Heading 1 — dark navy, 16pt bold
    h1 = doc.styles["Heading 1"]
    h1.font.name       = "Calibri"
    h1.font.size       = Pt(16)
    h1.font.bold       = True
    h1.font.color.rgb  = NAVY
    h1.paragraph_format.space_before = Pt(18)
    h1.paragraph_format.space_after  = Pt(6)
    h1.paragraph_format.keep_with_next = True

    # Heading 2 — medium blue, 13pt bold
    h2 = doc.styles["Heading 2"]
    h2.font.name       = "Calibri"
    h2.font.size       = Pt(13)
    h2.font.bold       = True
    h2.font.color.rgb  = BLUE
    h2.paragraph_format.space_before = Pt(14)
    h2.paragraph_format.space_after  = Pt(4)
    h2.paragraph_format.keep_with_next = True

    # Heading 3 — amber/gold, 12pt bold
    h3 = doc.styles["Heading 3"]
    h3.font.name       = "Calibri"
    h3.font.size       = Pt(12)
    h3.font.bold       = True
    h3.font.color.rgb  = AMBER
    h3.paragraph_format.space_before = Pt(10)
    h3.paragraph_format.space_after  = Pt(3)
    h3.paragraph_format.keep_with_next = True


def _add_horizontal_rule(doc: Document) -> None:
    """Adds a thin blue horizontal rule using a bottom border on an empty paragraph."""
    para = doc.add_paragraph()
    para.paragraph_format.space_before = Pt(2)
    para.paragraph_format.space_after  = Pt(10)

    pPr = para._element.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"),   "single")
    bottom.set(qn("w:sz"),    "12")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "2E75B6")
    pBdr.append(bottom)
    pPr.append(pBdr)


def _add_bold_runs(paragraph, text: str) -> None:
    """
    Adds text to a paragraph, converting **bold** markers to bold runs.
    Preserves font color of the parent paragraph style.
    """
    parts = re.split(r"(\*\*.*?\*\*)", text)
    for part in parts:
        if part.startswith("**") and part.endswith("**"):
            run = paragraph.add_run(part[2:-2])
            run.bold = True
        elif part:
            paragraph.add_run(part)


def _is_bullet(line: str) -> tuple[bool, str]:
    """
    Returns (True, text) if the line is a bullet point.
    Recognises: '* text', '- text', '• text'
    """
    stripped = line.strip()
    for prefix in ("* ", "- ", "• "):
        if stripped.startswith(prefix):
            return True, stripped[len(prefix):]
    return False, stripped


def _add_markdown_content(doc: Document, text: str) -> None:
    """
    Parses markdown-style text and adds it to the document.

    Converts:
        ## Heading   →  Word Heading 1  (navy)
        ### Heading  →  Word Heading 2  (blue)
        #### Heading →  Word Heading 3  (amber)
        * text / - text  →  List Bullet style
        **bold**     →  bold run
        blank line   →  paragraph spacer
        plain text   →  Normal paragraph
    """
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("#### "):
            doc.add_heading(stripped[5:], level=3)

        elif stripped.startswith("### "):
            doc.add_heading(stripped[4:], level=2)

        elif stripped.startswith("## "):
            doc.add_heading(stripped[3:], level=1)

        elif stripped == "":
            # Blank line — add a small spacer only if previous was not a heading
            doc.add_paragraph("")

        else:
            is_bul, bul_text = _is_bullet(stripped)
            if is_bul:
                para = doc.add_paragraph(style="List Bullet")
                para.paragraph_format.left_indent  = Inches(0.3)
                para.paragraph_format.space_after  = Pt(3)
                _add_bold_runs(para, bul_text)
            else:
                para = doc.add_paragraph()
                para.paragraph_format.space_after = Pt(6)
                _add_bold_runs(para, stripped)

        i += 1


def build_research_doc(state: dict) -> bytes:
    """
    Builds a formatted Word document from the completed research pipeline state.

    Args:
        state: the final ResearchState dict after all agents have run.

    Returns:
        bytes: the .docx file contents, ready for st.download_button.
    """
    doc = Document()
    _set_document_styles(doc)

    topic      = state.get("topic", "Research Report")
    title      = state.get("title") or topic
    final_text = state.get("final", state.get("draft", "No output generated."))
    sources    = state.get("sources", [])
    model_used = state.get("model_used", "unknown")
    today      = date.today().strftime("%B %d, %Y")

    # ── Title block ──────────────────────────────────────────────────────────
    title_para = doc.add_paragraph()
    title_para.alignment = WD_ALIGN_PARAGRAPH.LEFT
    title_para.paragraph_format.space_before = Pt(0)
    title_para.paragraph_format.space_after  = Pt(4)
    title_run = title_para.add_run(title)
    title_run.font.name  = "Calibri"
    title_run.font.size  = Pt(22)
    title_run.font.bold  = True
    title_run.font.color.rgb = NAVY

    meta_para = doc.add_paragraph()
    meta_para.paragraph_format.space_after = Pt(2)
    meta_run = meta_para.add_run(f"Research Assistant  |  {today}")
    meta_run.font.size      = Pt(9)
    meta_run.font.color.rgb = GREY

    # Blue rule under the title — same visual device as the reference documents
    _add_horizontal_rule(doc)

    # ── Main content ─────────────────────────────────────────────────────────
    _add_markdown_content(doc, final_text)

    # ── Sources ──────────────────────────────────────────────────────────────
    if sources:
        doc.add_paragraph("")
        doc.add_heading("Sources", level=1)
        for i, url in enumerate(sources, 1):
            para = doc.add_paragraph(style="List Number")
            para.paragraph_format.space_after = Pt(2)
            run = para.add_run(url)
            run.font.size = Pt(9)
            run.font.color.rgb = GREY

    # ── Footer ───────────────────────────────────────────────────────────────
    doc.add_paragraph("")
    footer_para = doc.add_paragraph()
    footer_run = footer_para.add_run(
        f"Generated by Multi-Agent Studio — Research Assistant  |  Model: {model_used}  |  {today}\n"
        "AI-generated output. Review before use."
    )
    footer_run.font.size      = Pt(8)
    footer_run.font.color.rgb = GREY

    # ── Return as bytes ──────────────────────────────────────────────────────
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.read()
