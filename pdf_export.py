"""PDF 导出（学自 ResumeCraftr 的 PyMuPDF 管线）。

ResumeCraftr 用 PyMuPDF 直接排版，无 LaTeX 依赖，输出纯文本可解析 = ATS 友好。
适配中文：Noto Sans SC 字体（/opt/demo/resume/fonts/）。
"""
from __future__ import annotations

import re
from pathlib import Path

FONT_DIR = Path("/opt/demo/resume/fonts")
FONT_REGULAR = str(FONT_DIR / "NotoSansSC-Regular.otf")
FONT_BOLD = str(FONT_DIR / "NotoSansSC-Bold.otf")


def _parse_markdown(md: str) -> list[tuple[str, str]]:
    """Markdown → [(类型, 文本)]，类型: h1/h2/bullet/text。"""
    blocks: list[tuple[str, str]] = []
    for line in md.splitlines():
        line = line.rstrip()
        if not line.strip():
            continue
        # 去 Markdown 标记
        clean = re.sub(r"\*\*(.+?)\*\*", r"\1", line).strip()
        clean = re.sub(r"^#{1,6}\s*", "", clean).strip()
        clean = re.sub(r"^[-*]\s+", "", clean).strip()
        if re.match(r"^#{1}\s", line):
            blocks.append(("h1", clean))
        elif re.match(r"^#{2,3}\s", line):
            blocks.append(("h2", clean))
        elif re.match(r"^[-*]\s", line):
            blocks.append(("bullet", clean))
        else:
            blocks.append(("text", clean))
    return blocks


def _subset_font(font_path: str, text: str) -> bytes:
    """用 fonttools 按文本子集化字体，返回字体字节（几 KB 而非几 MB）。"""
    from fontTools import subset
    from io import BytesIO
    options = subset.Options()
    ss = subset.Subsetter(options)
    ss.populate(text=text)
    font = subset.load_font(font_path, options)
    ss.subset(font)
    buf = BytesIO()
    font.save(buf)
    return buf.getvalue()


def render_pdf(markdown_text: str) -> bytes:
    """渲染简历 PDF，返回字节。"""
    import pymupdf

    # 收集全部字符，子集化字体
    all_text = markdown_text + "• 0123456789.%"
    font_buffers: dict[str, bytes] = {}
    for alias, path in (("sans", FONT_REGULAR), ("sansb", FONT_BOLD)):
        if Path(path).exists():
            try:
                font_buffers[alias] = _subset_font(path, all_text)
            except Exception:
                font_buffers[alias] = Path(path).read_bytes()

    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)  # A4
    margin = 56
    cursor = margin
    max_width = 595 - 2 * margin

    def _register(pg) -> None:
        for alias, buf in font_buffers.items():
            pg.insert_font(fontname=alias, fontbuffer=buf)

    _register(page)

    def ensure(extra: float) -> None:
        nonlocal page, cursor
        if cursor + extra < 842 - margin:
            return
        page = doc.new_page(width=595, height=842)
        _register(page)

    def write(text: str, size: float, bold: bool = False, indent: float = 0.0,
              gap: float = 4.0, color: tuple[float, float, float] = (0.11, 0.13, 0.16)) -> None:
        nonlocal cursor
        font = "sansb" if bold else "sans"
        # 简单换行：按字符数估算（中文等宽近似）
        chars_per_line = max(20, int(max_width / (size * 0.95)))
        chunks = [text[i:i + chars_per_line] for i in range(0, len(text), chars_per_line)] or [""]
        for chunk in chunks:
            ensure(size * 1.6)
            page.insert_text((margin + indent, cursor), chunk,
                             fontname=font, fontsize=size, color=color)
            cursor += size * 1.5
        cursor += gap

    for kind, text in _parse_markdown(markdown_text):
        if not text:
            continue
        if kind == "h1":
            write(text, 20, bold=True, gap=6)
        elif kind == "h2":
            cursor += 6
            write(text, 13, bold=True, gap=4, color=(0.09, 0.36, 1.0))
            # 标题下划线
            ensure(2)
            page.draw_line((margin, cursor), (595 - margin, cursor),
                           color=(0.09, 0.36, 1.0), width=1.2)
            cursor += 8
        elif kind == "bullet":
            write("•  " + text, 10.5, indent=12, gap=2)
        else:
            write(text, 10.5, gap=3)

    out = doc.tobytes()
    doc.close()
    return bytes(out)
