"""Render downloads/research-paper.md to a typeset PDF with ReportLab.

Supported markup (deliberately small):
  %T title, %T2 subtitle, %A author line
  %ABS .. %ENDABS (abstract block)   %ABSEN .. %ENDABS (english abstract)
  # / ## / ### headings
  blank-line separated paragraphs, "- " bullet lists
  $$ ... $$ display formula line (centred)
  !fig:path|caption      !table:caption  |a|b| rows  !endtable
  ``` code fences ``` (monospace block)
  **bold** -> gothic face, *italic* -> serif italic-ish
"""
import re, sys
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                               Spacer, Image, Table, TableStyle, KeepTogether, Preformatted)
import os

SERIF, GOTHIC, MONO = "HYSMyeongJo-Medium", "HYGoThic-Medium", "Mono"
pdfmetrics.registerFont(UnicodeCIDFont(SERIF))
try:
    pdfmetrics.registerFont(UnicodeCIDFont(GOTHIC))
except Exception:
    GOTHIC = SERIF
_mp = "/usr/local/lib/python3.12/site-packages/matplotlib/mpl-data/fonts/ttf"
for name, f in (("Mono", "DejaVuSansMono.ttf"), ("MonoB", "DejaVuSansMono-Bold.ttf")):
    try:
        pdfmetrics.registerFont(TTFont(name, os.path.join(_mp, f)))
    except Exception:
        pass

# The built-in Korean CID face (HYSMyeongJo-Medium) has no glyph for a few
# characters used in the papers: the middle dot, the minus sign, the floor and
# ceiling bars, the empty set, the set-minus sign, and Latin diacritics in
# reference author names. Those are drawn with DejaVuSans instead, so they are
# neither dropped nor replaced by blanks.
FALLBACK = "Sym"
MISSING_GLYPHS = "·Éáéóćčń∅−∖⌈⌉⌊⌋"
try:
    pdfmetrics.registerFont(TTFont(FALLBACK, os.path.join(_mp, "DejaVuSans.ttf")))
except Exception:
    FALLBACK = None

AUTHOR = "허태혁 (Taehyeok Heo)"
SRC = sys.argv[1]
OUT = sys.argv[2]
FOOTER_TITLE = sys.argv[3] if len(sys.argv) > 3 else ""

ss = getSampleStyleSheet()
ST = {
    "body": ParagraphStyle("body", parent=ss["BodyText"], fontName=SERIF, fontSize=10,
                           leading=16.2, alignment=TA_JUSTIFY, spaceAfter=6,
                           wordWrap="CJK"),
    "abs": ParagraphStyle("abs", parent=ss["BodyText"], fontName=SERIF, fontSize=9.2,
                          leading=14.6, alignment=TA_JUSTIFY, leftIndent=14,
                          rightIndent=14, spaceAfter=5, wordWrap="CJK"),
    "alert": ParagraphStyle("alert", parent=ss["BodyText"], fontName=SERIF, fontSize=9,
                            leading=13.5, alignment=TA_JUSTIFY, leftIndent=8, rightIndent=8,
                            textColor=colors.HexColor("#111111"), wordWrap="CJK"),
    "h1": ParagraphStyle("h1", fontName=GOTHIC, fontSize=14, leading=19, spaceBefore=16,
                         spaceAfter=7, textColor=colors.HexColor("#12294a")),
    "h2": ParagraphStyle("h2", fontName=GOTHIC, fontSize=11.6, leading=16, spaceBefore=11,
                         spaceAfter=5, textColor=colors.HexColor("#12294a")),
    "h3": ParagraphStyle("h3", fontName=GOTHIC, fontSize=10.4, leading=14.5, spaceBefore=8,
                         spaceAfter=3),
    "title": ParagraphStyle("title", fontName=GOTHIC, fontSize=17.5, leading=24,
                            alignment=TA_CENTER, spaceAfter=6),
    "sub": ParagraphStyle("sub", fontName=SERIF, fontSize=11.4, leading=16,
                          alignment=TA_CENTER, spaceAfter=4, textColor=colors.HexColor("#333333")),
    "auth": ParagraphStyle("auth", fontName=SERIF, fontSize=8.8, leading=13,
                           alignment=TA_CENTER, spaceAfter=10,
                           textColor=colors.HexColor("#444444")),
    "cap": ParagraphStyle("cap", fontName=SERIF, fontSize=8.4, leading=12.4,
                          alignment=TA_LEFT, spaceBefore=3, spaceAfter=10,
                          textColor=colors.HexColor("#222222"), wordWrap="CJK"),
    "bullet": ParagraphStyle("bullet", parent=ss["BodyText"], fontName=SERIF, fontSize=10,
                             leading=15.8, alignment=TA_JUSTIFY, leftIndent=14,
                             bulletIndent=4, spaceAfter=3, wordWrap="CJK"),
    "formula": ParagraphStyle("formula", fontName=SERIF, fontSize=9.8, leading=15,
                              alignment=TA_CENTER, spaceBefore=3, spaceAfter=6,
                              wordWrap="CJK"),
    "ref": ParagraphStyle("ref", fontName=SERIF, fontSize=8.7, leading=12.6,
                          alignment=TA_JUSTIFY, leftIndent=16, firstLineIndent=-16,
                          spaceAfter=2.5, wordWrap="CJK"),
    "code": ParagraphStyle("code", fontName="Mono", fontSize=7.4, leading=10.2,
                           leftIndent=8, backColor=colors.HexColor("#f4f4f4"),
                           borderPadding=4, spaceBefore=3, spaceAfter=8),
}


def _fix_sub(t):
    for a, b in (("₂", "2"), ("₃", "3"), ("₁", "1"), ("₀", "0")):
        t = t.replace(a, b)
    return t


def esc(t):
    t = _fix_sub(t)
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    if FALLBACK:
        t = re.sub("[%s]" % MISSING_GLYPHS,
                   lambda m: '<font name="%s">%s</font>' % (FALLBACK, m.group(0)), t)
    t = re.sub(r"\*\*(.+?)\*\*", r'<font name="%s">\1</font>' % GOTHIC, t)
    t = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", t)
    return t


def build():
    lines = open(SRC, encoding="utf-8").read().split("\n")
    flow, i = [], 0
    # ---- title block
    title = sub = auth = None
    body = []
    for ln in lines:
        if ln.startswith("%T "):
            title = ln[3:].strip()
        elif ln.startswith("%T2 "):
            sub = ln[4:].strip()
        elif ln.startswith("%A "):
            auth = ln[3:].strip()
        else:
            body.append(ln)
    flow.append(Spacer(1, 6))
    flow.append(Paragraph(esc(title), ST["title"]))
    if sub:
        flow.append(Paragraph(esc(sub), ST["sub"]))
    if auth:
        flow.append(Paragraph(esc(auth), ST["auth"]))
    flow.append(Spacer(1, 4))

    lines = body
    i = 0
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if not s:
            i += 1
            continue
        # ---- abstracts
        if s in ("%ABS", "%ABSEN"):
            i += 1
            buf = []
            while i < len(lines) and lines[i].strip() not in ("%ENDABS", "%ABSEN"):
                if lines[i].strip():
                    buf.append(lines[i].strip())
                i += 1
            i += 1
            label = "초록" if s == "%ABS" else "Abstract"
            flow.append(Paragraph(f"<font name='{GOTHIC}' size=9.5>{label}</font>",
                                  ParagraphStyle("al", fontName=GOTHIC, fontSize=9.5,
                                                 leading=13, spaceBefore=6, spaceAfter=2)))
            for x in buf:
                flow.append(Paragraph(esc(x), ST["abs"]))
            flow.append(Spacer(1, 6))
            continue
        # ---- headings
        if s.startswith("### "):
            flow.append(Paragraph(esc(s[4:]), ST["h3"])); i += 1; continue
        if s.startswith("## "):
            flow.append(Paragraph(esc(s[3:]), ST["h2"])); i += 1; continue
        if s.startswith("# "):
            flow.append(Paragraph(esc(s[2:]), ST["h1"])); i += 1; continue
        # ---- code fence
        if s.startswith("```"):
            i += 1
            buf = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i]); i += 1
            i += 1
            flow.append(Preformatted("\n".join(buf), ST["code"]))
            continue
        # ---- display formula
        if s.startswith("$$"):
            txt = s.strip("$").strip()
            flow.append(Paragraph(esc(txt), ST["formula"])); i += 1; continue
        # ---- figure
        if s.startswith("!fig:"):
            payload = s[5:]
            path, cap = (payload.split("|", 1) + [""])[:2]
            path = path.strip()
            if os.path.exists(path):
                from reportlab.lib.utils import ImageReader
                iw, ih = ImageReader(path).getSize()
                w = 470
                h = w * ih / iw
                flow.append(KeepTogether([Image(path, width=w, height=h),
                                          Paragraph(esc(cap.strip()), ST["cap"])]))
            else:
                flow.append(Paragraph("[missing figure: %s]" % esc(path), ST["cap"]))
            i += 1
            continue
        # ---- table
        if s.startswith("!table:"):
            cap = s[7:].strip()
            i += 1
            rows = []
            while i < len(lines) and not lines[i].strip().startswith("!endtable"):
                r = lines[i].strip()
                if r.startswith("|"):
                    cells = [c.strip() for c in r.strip("|").split("|")]
                    rows.append(cells)
                i += 1
            i += 1
            if not rows:
                continue
            ncol = max(len(r) for r in rows)
            rows = [r + [""] * (ncol - len(r)) for r in rows]
            data = [[Paragraph(esc(c), ParagraphStyle("tc", fontName=SERIF, fontSize=7.8,
                                                      leading=10.6, wordWrap="CJK"))
                     for c in r] for r in rows]
            tw = 500
            first = max(1, int(ncol * 0.34)) if ncol > 2 else ncol
            widths = [tw * 0.30] + [tw * 0.70 / (ncol - 1)] * (ncol - 1) if ncol > 1 else [tw]
            t = Table(data, colWidths=widths, repeatRows=1)
            t.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f7")),
                ("FONTNAME", (0, 0), (-1, 0), GOTHIC),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ]))
            flow.append(KeepTogether([t, Paragraph(esc(cap), ST["cap"])]))
            continue
        # ---- bullets
        if s.startswith("- "):
            buf = [s[2:]]
            i += 1
            while i < len(lines) and lines[i].strip().startswith("- "):
                buf.append(lines[i].strip()[2:]); i += 1
            for b in buf:
                flow.append(Paragraph(esc(b), ST["bullet"], bulletText="•"))
            flow.append(Spacer(1, 3))
            continue
        # ---- reference list
        if re.match(r"^\[[^\]]+\]\s", s):
            buf = [s]
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith(("#", "- ", "!fig", "!table", "$$")):
                buf.append(lines[i].strip()); i += 1
            flow.append(Paragraph(esc(" ".join(buf)), ST["ref"]))
            continue
        # ---- paragraph
        buf = [s]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(
                r"^(#|##|###|- |!fig:|!table:|\$\$|```|\[[^\]]+\] )", lines[i].strip()):
            buf.append(lines[i].strip()); i += 1
        flow.append(Paragraph(esc(" ".join(buf)), ST["body"]))

    # Keep a heading with the block it introduces when that block is a table,
    # so the heading is never stranded at the foot of a page.
    merged, i = [], 0
    while i < len(flow):
        f = flow[i]
        if (isinstance(f, Paragraph) and getattr(f, "style", None) is not None
                and f.style.name in ("h1", "h2", "h3")
                and i + 1 < len(flow) and isinstance(flow[i + 1], KeepTogether)):
            inner = getattr(flow[i + 1], "_content", None)
            if inner is not None:
                merged.append(KeepTogether([f] + list(inner)))
                i += 2
                continue
        merged.append(f)
        i += 1
    flow = merged

    doc = BaseDocTemplate(OUT, pagesize=A4,
                          leftMargin=47, rightMargin=47, topMargin=44, bottomMargin=46,
                          title=title, author=AUTHOR,
                          subject=sub or "", creator=AUTHOR, keywords="")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")

    def footer(canvas, docu):
        canvas.saveState()
        canvas.setFont(SERIF, 7.6)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(47, 26, (AUTHOR + " — " + FOOTER_TITLE) if FOOTER_TITLE else AUTHOR)
        canvas.drawRightString(A4[0] - 47, 26, "%d" % docu.page)
        canvas.setStrokeColor(colors.HexColor("#cccccc"))
        canvas.line(47, 36, A4[0] - 47, 36)
        canvas.restoreState()

    doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=footer)])
    doc.build(flow)
    print("wrote", OUT, os.path.getsize(OUT), "bytes")


if __name__ == "__main__":
    build()
