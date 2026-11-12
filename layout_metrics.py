"""版面诊断指标：只量度 render_pdf() 的渲染结果，不改渲染逻辑。

用途：给 PDF 导出做「版面体检」，回答两个问题——
1. 一共排了几页（pages）；
2. 末页用到哪（last_page_fill，末页最后一个文本块底边距页高的比例，0–1）。

它是纯函数：只读输入字节，不写文件、不碰网络、不依赖 render_pdf 的实现细节；
PyMuPDF 读回 PDF 后按「文本块」取几何信息，因此指标与实际渲染一致。
"""
from __future__ import annotations

from typing import Optional, TypedDict


class LayoutMetrics(TypedDict):
    """layout_metrics() 的返回结构。"""

    pages: int
    """PDF 页数（至少 1）。"""

    last_page_fill: float
    """末页最后一个文本块底边 / 页高，取值 0–1。

    没有可量度的文本块（空页、只画了线条之类）时为 0.0。
    """


def _last_text_block(page) -> Optional[tuple]:
    """末页最后一个文本块；没有文本块时返回 None。"""
    last = None
    for block in page.get_text("blocks"):
        # blocks 形如 (x0, y0, x1, y1, text, block_no, block_type)；block_type 0 = 文本
        if len(block) > 6 and block[6] != 0:
            continue
        if not str(block[4]).strip():
            continue
        last = block
    return last


def _last_page_fill(page) -> float:
    """末页最后一个文本块底边距页高的比例；无文本块时返回 0.0。

    「最后一个」取 blocks 序列末项——PyMuPDF 按版面顺序（自上而下）返回文本块，
    对 render_pdf 的常规单栏输出即阅读顺序的最后一块。
    """
    height = float(page.rect.height)
    if height <= 0:
        return 0.0
    block = _last_text_block(page)
    if block is None:
        return 0.0
    return round(min(max(float(block[3]) / height, 0.0), 1.0), 4)


def layout_metrics(pdf_bytes: bytes) -> LayoutMetrics:
    """量度 render_pdf() 产出的 PDF 字节，返回 {pages, last_page_fill}。

    - pages：页数；
    - last_page_fill：末页最后一个文本块底边距页高的比例（0–1），
      没有文本块时为 0.0。

    输入不是可解析的 PDF 时抛 PyMuPDF 的异常（不吞错，交由调用方判断）。
    """
    import pymupdf

    doc = pymupdf.open(stream=bytes(pdf_bytes), filetype="pdf")
    try:
        pages = doc.page_count
        if pages == 0:
            return {"pages": 0, "last_page_fill": 0.0}
        fill = _last_page_fill(doc[pages - 1])
        return {"pages": pages, "last_page_fill": fill}
    finally:
        doc.close()
