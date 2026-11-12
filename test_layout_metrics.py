"""layout_metrics() 测试：三份样本（偏空 / 刚好满一页 / 溢出到两页）。

指标必须与实际渲染一致——这里不信任 layout_metrics 自己的几何计算，
每条用例都用 PyMuPDF 独立重算一遍「末页最后一个文本块底边 / 页高」，
再与 layout_metrics 的返回值比对。

只读 pdf_export.render_pdf() 的产出，不改渲染逻辑。
运行：python -m pytest test_layout_metrics.py -v
"""
import unittest
from pathlib import Path

import pdf_export
from layout_metrics import layout_metrics

# ---------------------------------------------------------------- 字体准备
# render_pdf 硬编码 /opt/demo/resume/fonts/ 下的两个 Noto Sans SC 字体。
# 该目录不在仓库里：本机缺失时（如 CI 未跑 scripts/setup_local_fonts.py）
# 从系统字体里挑一个能渲染中文的顶上，只改本进程内 pdf_export 的常量，
# 不写文件、不改渲染逻辑；系统里也一个都没有就整体跳过。
_FONT_CANDIDATES = (
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "Regular"),
    ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", "Regular"),
    ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttf", "Regular"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "Regular"),
)
_BOLD_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)


def _ensure_fonts() -> bool:
    """保证 render_pdf 有字体可用；返回 False 表示本机无法渲染（调用方跳过）。"""
    if Path(pdf_export.FONT_REGULAR).exists() and Path(pdf_export.FONT_BOLD).exists():
        return True
    regular = next((p for p, _ in _FONT_CANDIDATES if Path(p).exists()), None)
    bold = next((p for p in _BOLD_CANDIDATES if Path(p).exists()), regular)
    if regular is None or bold is None:
        return False
    pdf_export.FONT_REGULAR = regular
    pdf_export.FONT_BOLD = bold
    return True


_RENDER_AVAILABLE = _ensure_fonts()

# ---------------------------------------------------------------- 样本构造
# 简历样本：标题 + 二级标题 + n 条单行要点。版面上的推进量只由 render_pdf
# 里的字号常量决定（与字体无关），所以「多少条刚好排满一页」是可复现的。
HEADER = "# 张伟\n\n## 工作经历"
BULLET = "负责跨部门数据分析与交付，推动口径统一，覆盖三个业务线"


def _resume(count: int) -> str:
    """n 条要点的简历 Markdown。"""
    return "\n".join([HEADER] + [f"- {BULLET}" for _ in range(count)])


# 三份样本的要点条数（compact 密度下实测标定，见各用例注释）
EMPTY_ISH_COUNT = 26   # 约六成满
FULL_PAGE_COUNT = 40   # 刚好排满一页、再放一条就溢出
OVERFLOW_COUNT = 41    # 第一条溢出到第二页 -> 共两页


def _render(count: int) -> bytes:
    return pdf_export.render_pdf(_resume(count))


def _actual_last_page_fill(pdf_bytes: bytes) -> float:
    """独立重算：末页最后一个文本块底边 / 页高。不走 layout_metrics 的实现。

    「最后一个」= blocks 序列末项（PyMuPDF 的版面/阅读顺序，即先插入的那一块）。
    """
    import pymupdf

    doc = pymupdf.open(stream=pdf_bytes)
    try:
        page = doc[doc.page_count - 1]
        blocks = [b for b in page.get_text("blocks") if str(b[4]).strip()]
        if not blocks:
            return 0.0
        return float(blocks[-1][3]) / float(page.rect.height)
    finally:
        doc.close()


def _page_texts(pdf_bytes: bytes) -> list[str]:
    import pymupdf

    doc = pymupdf.open(stream=pdf_bytes)
    try:
        return [page.get_text() for page in doc]
    finally:
        doc.close()


@unittest.skipUnless(_RENDER_AVAILABLE, "本机没有可用于 render_pdf 的中文字体，跳过 pdf_export 的渲染用例")
class LayoutMetricsTests(unittest.TestCase):
    def assert_metrics_consistent(self, metrics: dict, pdf_bytes: bytes) -> None:
        """返回结构、取值范围，以及「与实际渲染一致」。"""
        self.assertEqual(set(metrics), {"pages", "last_page_fill"})
        self.assertEqual(metrics["pages"], len(_page_texts(pdf_bytes)))
        fill = metrics["last_page_fill"]
        self.assertIsInstance(fill, float)
        self.assertGreaterEqual(fill, 0.0)
        self.assertLessEqual(fill, 1.0)
        # 与实际渲染一致：与独立重算的几何值相同（允许 1e-3 的取整误差）
        self.assertAlmostEqual(fill, _actual_last_page_fill(pdf_bytes), delta=1e-3)

    def test_partial_page_about_sixty_percent(self):
        """偏空：单页，约六成满，填充率落在预期区间。"""
        pdf = _render(EMPTY_ISH_COUNT)
        metrics = layout_metrics(pdf)

        self.assertEqual(metrics["pages"], 1)
        self.assertGreaterEqual(metrics["last_page_fill"], 0.60)
        self.assertLessEqual(metrics["last_page_fill"], 0.70)
        self.assert_metrics_consistent(metrics, pdf)

    def test_full_page(self):
        """刚好满一页：单页，填充率接近 1（内容一直排到页底边距）。"""
        pdf = _render(FULL_PAGE_COUNT)
        metrics = layout_metrics(pdf)

        self.assertEqual(metrics["pages"], 1)
        self.assertGreaterEqual(metrics["last_page_fill"], 0.90)
        self.assertLessEqual(metrics["last_page_fill"], 1.0)
        self.assert_metrics_consistent(metrics, pdf)

        # 「刚好」再放一条就排不下：下一份样本必须溢出，证明前一份是临界点
        self.assertEqual(layout_metrics(_render(OVERFLOW_COUNT))["pages"], 2)

    def test_overflow_two_pages(self):
        """溢出到两页：页数为 2，末页填充率落在预期区间。"""
        pdf = _render(OVERFLOW_COUNT)
        metrics = layout_metrics(pdf)

        self.assertEqual(metrics["pages"], 2)
        self.assertGreaterEqual(metrics["last_page_fill"], 0.90)
        self.assertLessEqual(metrics["last_page_fill"], 1.0)
        self.assert_metrics_consistent(metrics, pdf)

        # 溢出页上确实有文本，不是空白页
        self.assertTrue(_page_texts(pdf)[1].strip())

    def test_partial_page_has_no_second_page(self):
        """偏空样本的页数是 1（防止样本构造失效后其它用例假通过）。"""
        self.assertEqual(len(_page_texts(_render(EMPTY_ISH_COUNT))), 1)

    def test_rejects_non_pdf_bytes(self):
        """输入不是 PDF 时抛异常，不返回伪造指标。"""
        with self.assertRaises(Exception):
            layout_metrics(b"not a pdf")


if __name__ == "__main__":
    unittest.main()
