"""Converts CanonFodder_requirementsanddesigndocumentation.md to PDF via fpdf2."""

from __future__ import annotations

import re
from pathlib import Path

from fpdf import FPDF

SRC = Path(__file__).parent / "CanonFodder_requirementsanddesigndocumentation.md"
DST = Path(__file__).parent / "CanonFodder_requirementsanddesigndocumentation.pdf"
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


class DocPDF(FPDF):
    """PDF with header/footer and markdown rendering."""

    def __init__(self):
        """Initialises PDF with Unicode-capable DejaVu fonts."""
        super().__init__()
        self.add_font("DejaVu", "", str(FONT_DIR / "DejaVuSans.ttf"))
        self.add_font("DejaVu", "B", str(FONT_DIR / "DejaVuSans-Bold.ttf"))
        self.add_font("DejaVuMono", "", str(FONT_DIR / "DejaVuSansMono.ttf"))

    def header(self):
        """Renders page header."""
        self.set_font("DejaVu", "", 7)
        self.cell(0, 5, "c9r (CanonFodder) \u2014 Requirements & Design Documentation v0.8", align="C")
        self.ln(6)

    def footer(self):
        """Renders page footer with page number."""
        self.set_y(-15)
        self.set_font("DejaVu", "", 7)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def chapter_title(self, level: int, title: str):
        """Renders a heading at the given level."""
        sizes = {1: 15, 2: 12, 3: 10}
        self.set_font("DejaVu", "B", sizes.get(level, 10))
        if level <= 2:
            self.ln(4)
        self.multi_cell(0, 7, title)
        self.ln(2)

    def body_text(self, text: str):
        """Renders body text with basic bold/code inline formatting."""
        self.set_font("DejaVu", "", 9)
        self.multi_cell(0, 5, text)
        self.ln(1)

    def code_block(self, text: str):
        """Renders a code block with monospace font and grey background."""
        self.set_font("DejaVuMono", "", 7)
        self.set_fill_color(240, 240, 240)
        for line in text.split("\n"):
            self.cell(0, 4, "  " + line, fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)
        self.set_font("DejaVu", "", 9)

    def horizontal_rule(self):
        """Renders a horizontal line."""
        self.ln(3)
        y = self.get_y()
        self.line(self.l_margin, y, self.w - self.r_margin, y)
        self.ln(3)


def render_md(pdf: DocPDF, md_text: str):
    """Parses markdown and renders into the PDF."""
    in_code = False
    code_buf: list[str] = []
    for line in md_text.split("\n"):
        # Toggling code blocks
        if line.strip().startswith("```"):
            if in_code:
                pdf.code_block("\n".join(code_buf))
                code_buf.clear()
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            code_buf.append(line)
            continue
        # Horizontal rule
        if line.strip() == "---":
            pdf.horizontal_rule()
            continue
        # Headings
        m = re.match(r"^(#{1,3})\s+(.*)", line)
        if m:
            pdf.chapter_title(len(m.group(1)), m.group(2))
            continue
        # Stripping bold markdown for plain text rendering
        clean = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
        clean = re.sub(r"`(.*?)`", r"\1", clean)
        # Skipping empty lines but adding small space
        if not clean.strip():
            pdf.ln(2)
            continue
        # Bullet points
        if clean.strip().startswith("- "):
            pdf.set_font("DejaVu", "", 9)
            pdf.cell(5)
            pdf.multi_cell(0, 5, clean.strip())
            pdf.ln(1)
            continue
        # Numbered lists
        nm = re.match(r"^(\d+)\.\s+(.*)", clean.strip())
        if nm:
            pdf.set_font("DejaVu", "", 9)
            pdf.cell(5)
            pdf.multi_cell(0, 5, clean.strip())
            pdf.ln(1)
            continue
        pdf.body_text(clean)


def main():
    """Builds the PDF from the markdown source."""
    md_text = SRC.read_text(encoding="utf-8")
    pdf = DocPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    render_md(pdf, md_text)
    pdf.output(str(DST))
    print(f"PDF written to {DST} ({DST.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
