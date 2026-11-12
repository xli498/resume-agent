"""docx_export.render_docx 的测试：块映射、行内加粗与中文完整性。"""
import unittest
from io import BytesIO

try:
    from docx import Document
except ModuleNotFoundError:  # CI 若未安装依赖，测试跳过而不是收集失败
    Document = None

from docx_export import render_docx

SAMPLE_MARKDOWN = """# 张三

## 联系方式
13800000000 · zhangsan@example.com

## 工作经历
- 某咨询公司｜运营助理｜2025.07—2025.08
- 参与**客户沟通**与资料整理，输出Excel汇总。

## 项目经历
**数据看板**：用Python清洗数据，产出周报。

## 技能证书
Excel、Python
"""


@unittest.skipIf(Document is None, "未安装 python-docx，跳过 DOCX 导出测试")
class DocxExportTests(unittest.TestCase):
    def setUp(self):
        self.docx_bytes = render_docx(SAMPLE_MARKDOWN)
        self.document = Document(BytesIO(self.docx_bytes))
        self.paragraphs = self.document.paragraphs

    def paragraphs_with_style(self, style_name):
        return [p for p in self.paragraphs if p.style.name == style_name]

    def test_returns_openable_docx_bytes(self):
        self.assertIsInstance(self.docx_bytes, bytes)
        self.assertTrue(self.docx_bytes.startswith(b"PK"), "docx 应为 zip 容器")
        # 打开无异常即证明文件结构合法
        self.assertGreater(len(self.paragraphs), 0)

    def test_h1_maps_to_heading_1(self):
        level_one = self.paragraphs_with_style("Heading 1")
        self.assertEqual([p.text for p in level_one], ["张三"])

    def test_h2_maps_to_heading_2(self):
        level_two = [p.text for p in self.paragraphs_with_style("Heading 2")]
        self.assertEqual(level_two, ["联系方式", "工作经历", "项目经历", "技能证书"])

    def test_bullets_map_to_list_bullet(self):
        bullets = self.paragraphs_with_style("List Bullet")
        self.assertEqual(len(bullets), 2)
        self.assertEqual(bullets[0].text, "某咨询公司｜运营助理｜2025.07—2025.08")
        self.assertEqual(bullets[1].text, "参与客户沟通与资料整理，输出Excel汇总。")

    def test_text_maps_to_normal_paragraph(self):
        texts = [p.text for p in self.paragraphs_with_style("Normal")]
        self.assertIn("13800000000 · zhangsan@example.com", texts)
        self.assertIn("数据看板：用Python清洗数据，产出周报。", texts)
        self.assertIn("Excel、Python", texts)

    def test_inline_bold_becomes_bold_run(self):
        body = next(p for p in self.paragraphs if p.text == "参与客户沟通与资料整理，输出Excel汇总。")
        bold_runs = [run for run in body.runs if run.bold]
        self.assertEqual([run.text for run in bold_runs], ["客户沟通"])
        plain_runs = [run.text for run in body.runs if not run.bold]
        self.assertIn("参与", plain_runs)
        self.assertIn("与资料整理，输出Excel汇总。", plain_runs)

    def test_bold_bullet_keeps_bold_run(self):
        bullet = next(p for p in self.paragraphs if p.text == "参与客户沟通与资料整理，输出Excel汇总。")
        bold_runs = [run for run in bullet.runs if run.bold]
        self.assertEqual(len(bold_runs), 1)
        self.assertEqual(bold_runs[0].text, "客户沟通")

    def test_leading_bold_and_text_are_split(self):
        paragraph = next(p for p in self.paragraphs if p.text == "数据看板：用Python清洗数据，产出周报。")
        self.assertTrue(paragraph.runs[0].bold)
        self.assertEqual(paragraph.runs[0].text, "数据看板")

    def test_chinese_content_is_not_garbled_or_lost(self):
        full_text = "\n".join(p.text for p in self.paragraphs)
        self.assertNotIn("\ufffd", full_text)
        for fragment in ("张三", "联系方式", "工作经历", "客户沟通", "资料整理", "技能证书"):
            self.assertIn(fragment, full_text)

    def test_markdown_markers_do_not_leak(self):
        full_text = "\n".join(p.text for p in self.paragraphs)
        self.assertNotIn("**", full_text)
        self.assertNotIn("# ", full_text)
        self.assertNotIn("- 某咨询公司", full_text)

    def test_empty_markdown_yields_valid_docx(self):
        empty = render_docx("")
        self.assertTrue(empty.startswith(b"PK"))
        document = Document(BytesIO(empty))
        self.assertEqual(document.paragraphs, [])


if __name__ == "__main__":
    unittest.main()