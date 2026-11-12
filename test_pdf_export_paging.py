"""render_pdf 翻页回归测试。

历史 bug：``ensure()`` 新建页面后没有把 ``cursor`` 重置回页边距，导致新页上
的第一个 ``write()`` 立即再次触发换页，从第 2 页起每个文本块各占一页，
末尾还留空白页（45 个条目渲染成 7 页）。

测试自带字体：优先用 pymupdf 内置 CJK 字体，摆脱 /opt/demo/resume/fonts
这类机器级路径依赖，CI 与本地都能跑。
"""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pymupdf

import pdf_export

# compact 密度：margin = int(56 * 0.8) = 44
PAGE_MARGIN = 44
A4_HEIGHT = 842


def _build_resume(count: int) -> str:
    lines = ["# 张三", "", "**求职方向：大模型应用研发**", "", "## 项目经历", ""]
    lines.extend(f"- 项目条目 {i}：负责后端服务与数据处理" for i in range(1, count + 1))
    return "\n".join(lines)


def _text_blocks(pdf_bytes: bytes) -> list[list[tuple]]:
    """返回每页的文本块（x0, y0, x1, y1, text, ...），保留页序。"""
    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        return [page.get_text("blocks") for page in document]
    finally:
        document.close()


class RenderPdfPagingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = TemporaryDirectory()
        font_path = Path(cls._temporary.name) / "DroidSansFallback.otf"
        font_path.write_bytes(pymupdf.Font("cjk").buffer)
        cls._patchers = [
            patch.object(pdf_export, "FONT_REGULAR", str(font_path)),
            patch.object(pdf_export, "FONT_BOLD", str(font_path)),
        ]
        for patcher in cls._patchers:
            patcher.start()

    @classmethod
    def tearDownClass(cls) -> None:
        for patcher in reversed(cls._patchers):
            patcher.stop()
        cls._temporary.cleanup()

    def test_multi_page_resume_uses_reasonable_page_count(self) -> None:
        """41/42/45 个条目应各占 1-3 页，且末页不是空白页。"""
        for count in (41, 42, 45):
            with self.subTest(count=count):
                pages = _text_blocks(pdf_export.render_pdf(_build_resume(count)))
                self.assertLessEqual(
                    len(pages), 3,
                    f"{count} 个条目渲染成 {len(pages)} 页，疑似每个文本块各占一页",
                )
                for index, blocks in enumerate(pages, start=1):
                    self.assertTrue(
                        "".join(block[4] for block in blocks).strip(),
                        f"第 {index} 页是空白页",
                    )
                whole_text = "".join(block[4] for page in pages for block in page)
                for item in range(1, count + 1):
                    self.assertIn(f"项目条目 {item}：", whole_text)
                last_page_text = "".join(block[4] for block in pages[-1])
                self.assertIn(f"项目条目 {count}：", last_page_text)

    def test_page_break_resets_cursor_to_top_margin(self) -> None:
        """第 2 页的首个文本块必须从页顶开始，而不是沿用上一页的页底位置。"""
        pages = _text_blocks(pdf_export.render_pdf(_build_resume(45)))
        self.assertGreater(len(pages), 1)
        first_block_y = min(block[1] for block in pages[1])
        self.assertLess(
            first_block_y, PAGE_MARGIN + 20,
            f"第 2 页首块 y0={first_block_y:.1f}，cursor 未重置到页边距 {PAGE_MARGIN}",
        )
        self.assertLess(first_block_y, A4_HEIGHT / 2)


if __name__ == "__main__":
    unittest.main()