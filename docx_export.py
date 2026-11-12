"""Word (DOCX) 导出：把定稿 Markdown 渲染为 .docx。

与 ``pdf_export.render_pdf`` 共用同一套块模型（h1 / h2 / bullet / text），
区别是这里保留行内 ``**加粗**``，写成真正的 Word 加粗 run。

``render_docx`` 是纯函数：只接收 Markdown 文本并返回 .docx 字节，不读写文件。
"""
from __future__ import annotations

import re
from io import BytesIO
from typing import NamedTuple

# 行内加粗：**文本**（与 pdf_export._parse_markdown 的识别口径一致）
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")

# 中文字体：优先微软雅黑，回退到 Word 默认字体；字体缺失时 Word 自行替换，
# 文本本身始终以 UTF-8 存入 .docx，不会乱码。
_CJK_FONT = "Microsoft YaHei"

_H1_COLOR = (0x1C, 0x21, 0x28)
_H2_COLOR = (0x17, 0x5C, 0xFF)
_BODY_COLOR = (0x1C, 0x21, 0x28)


class _Block(NamedTuple):
    """一个 Markdown 块：kind 为 h1/h2/bullet/text，text 保留 ``**`` 标记。"""

    kind: str
    text: str


def _parse_blocks(markdown_text: str) -> list[_Block]:
    """Markdown → 块列表，块划分与 ``pdf_export._parse_markdown`` 一致。

    与 PDF 版本不同的是：这里只去掉标题/列表前缀，不删除 ``**``，
    以便后续按加粗切分 run。
    """
    blocks: list[_Block] = []
    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if re.match(r"^#\s", line):
            text = re.sub(r"^#{1,6}\s*", "", line).strip()
            blocks.append(_Block("h1", text))
        elif re.match(r"^#{2,3}\s", line):
            text = re.sub(r"^#{1,6}\s*", "", line).strip()
            blocks.append(_Block("h2", text))
        elif re.match(r"^[-*]\s", line):
            text = re.sub(r"^[-*]\s+", "", line).strip()
            blocks.append(_Block("bullet", text))
        else:
            text = re.sub(r"^#{1,6}\s*", "", line).strip()
            blocks.append(_Block("text", text))
    return blocks


def _add_runs(paragraph, text: str) -> None:
    """把含 ``**加粗**`` 的文本写入段落，加粗片段生成 bold run。"""
    position = 0
    for match in _BOLD_RE.finditer(text):
        if match.start() > position:
            paragraph.add_run(text[position:match.start()])
        bold_run = paragraph.add_run(match.group(1))
        bold_run.bold = True
        position = match.end()
    if position < len(text):
        paragraph.add_run(text[position:])


def _set_style_font(style, font_name: str, size_pt: float, color: tuple[int, int, int] | None = None) -> None:
    """给 Word 样式同时设置西文与中文字体，避免中文回退成方框。"""
    from docx.oxml.ns import qn
    from docx.shared import Pt, RGBColor

    style.font.name = font_name
    style.font.size = Pt(size_pt)
    if color is not None:
        style.font.color.rgb = RGBColor(*color)
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.get_or_add_rFonts()
    rfonts.set(qn("w:ascii"), font_name)
    rfonts.set(qn("w:hAnsi"), font_name)
    rfonts.set(qn("w:eastAsia"), font_name)


def render_docx(markdown_text: str) -> bytes:
    """把定稿 Markdown 渲染为 .docx，返回文件字节。

    块映射：h1 → 一级标题，h2 → 二级标题，bullet → 项目符号列表项，
    text → 普通段落；行内 ``**...**`` 保留为加粗 run。
    """
    from docx import Document
    from docx.shared import Cm, Pt

    document = Document()

    section = document.sections[0]
    section.page_width = Cm(21.0)   # A4，与 pdf_export 的 595×842pt 页面一致
    section.page_height = Cm(29.7)
    section.left_margin = Cm(2.0)
    section.right_margin = Cm(2.0)
    section.top_margin = Cm(1.8)
    section.bottom_margin = Cm(1.8)

    _set_style_font(document.styles["Normal"], _CJK_FONT, 10.5, _BODY_COLOR)
    _set_style_font(document.styles["Heading 1"], _CJK_FONT, 20, _H1_COLOR)
    _set_style_font(document.styles["Heading 2"], _CJK_FONT, 13, _H2_COLOR)
    _set_style_font(document.styles["List Bullet"], _CJK_FONT, 10.5, _BODY_COLOR)

    for block in _parse_blocks(markdown_text):
        if not block.text:
            continue
        if block.kind == "h1":
            paragraph = document.add_paragraph(style="Heading 1")
            paragraph.paragraph_format.space_before = Pt(0)
            paragraph.paragraph_format.space_after = Pt(6)
        elif block.kind == "h2":
            paragraph = document.add_paragraph(style="Heading 2")
            paragraph.paragraph_format.space_before = Pt(8)
            paragraph.paragraph_format.space_after = Pt(4)
        elif block.kind == "bullet":
            paragraph = document.add_paragraph(style="List Bullet")
            paragraph.paragraph_format.space_after = Pt(2)
        else:
            paragraph = document.add_paragraph(style="Normal")
            paragraph.paragraph_format.space_after = Pt(3)
        _add_runs(paragraph, block.text)

    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()