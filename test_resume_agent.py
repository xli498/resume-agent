import json
import os
import subprocess
import sys
import unittest
import tempfile
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

import llm_client
import delivery as delivery_module
from llm_client import call_llm
from delivery import (
    _trim_dark_side_canvas,
    export_delivery_package,
    export_single_page_pdf,
    qa_delivery,
    render_pdf_preview,
)
import main as main_module
from main import (
    build_analysis,
    build_confirmation_questions_report,
    build_fact_ledger,
    build_llm_alignment_report,
    build_revision_diff_report,
    build_evidence_mapping_report,
    build_llm_prompt,
    match_keywords,
    build_final_resume,
    build_targeted_resume_draft,
    mock_llm,
    parse_llm_json,
    parse_resume,
    prepare_output_dir,
    validate_resume_draft,
    validate_resume_evidence,
    validate_llm_revisions,
)
from main import render_resume_image
from workflow import run_python_workflow, run_langgraph_workflow, build_langgraph_workflow


class ResumeAgentTests(unittest.TestCase):
    def test_font_candidates_include_windows_font_directories(self):
        from delivery import _font_candidates
        candidates = [str(path).lower() for path in _font_candidates()]
        if os.name == "nt":
            self.assertTrue(any("fonts" in path for path in candidates))
        else:
            self.assertTrue(any("/usr/share/fonts" in path for path in candidates))

    def test_fsync_directory_is_best_effort_on_windows(self):
        from delivery import _fsync_directory
        with tempfile.TemporaryDirectory() as directory:
            _fsync_directory(Path(directory))

    @unittest.skipUnless(os.name == "nt", "仅 Windows 需要目录句柄刷新回归")
    def test_fsync_directory_flushes_windows_directory_handle(self):
        from delivery import _fsync_directory
        with tempfile.TemporaryDirectory() as directory:
            with patch("delivery._flush_windows_directory") as flush:
                _fsync_directory(Path(directory))
            flush.assert_called_once_with(Path(directory))
    def test_job_title_override_only_changes_header_direction(self):
        source = """张三

个人信息
张三

求职意向
大模型应用研发实习生

工作经历
某公司｜财务助理实习生
"""
        result = build_final_resume(source, jd_text="岗位名称：大模型应用研发实习生", job_title="大模型应用研发")
        self.assertIn("**求职方向：大模型应用研发**", result)
        self.assertIn("某公司｜财务助理实习生", result)

    def test_python_workflow_accepts_explicit_job_title(self):
        result = run_python_workflow(
            "张三\n个人信息\n张三\n工作经历\n某公司｜研发实习生",
            "岗位名称：大模型应用研发实习生",
            job_title="大模型应用研发",
        )
        self.assertIn("**求职方向：大模型应用研发**", result["final_resume"])
        self.assertIn("某公司｜研发实习生", result["final_resume"])

    def test_photo_cleanup_only_trims_black_right_canvas(self):
        image = __import__("PIL.Image", fromlist=["Image"]).new("RGB", (100, 120), "#3A78B8")
        for x in range(80, 100):
            for y in range(120):
                image.putpixel((x, y), (0, 0, 0))
        cleaned = _trim_dark_side_canvas(image)
        self.assertEqual(cleaned.size, (80, 120))

    def test_photo_cleanup_trims_noisy_near_black_right_canvas(self):
        image = __import__("PIL.Image", fromlist=["Image"]).new("RGB", (100, 120), "#3A78B8")
        for x in range(80, 100):
            for y in range(120):
                value = 12 if (x + y) % 23 else 34
                image.putpixel((x, y), (value, value, value))
        cleaned = _trim_dark_side_canvas(image)
        self.assertEqual(cleaned.size, (80, 120))

    def test_photo_cleanup_keeps_full_height_dark_gray_subject_or_background(self):
        image = __import__("PIL.Image", fromlist=["Image"]).new("RGB", (100, 120), "#3A78B8")
        for x in range(80, 100):
            for y in range(120):
                image.putpixel((x, y), (35, 35, 35))
        cleaned = _trim_dark_side_canvas(image)
        self.assertEqual(cleaned.size, (100, 120))

    def test_photo_cleanup_does_not_trim_partial_black_clothing(self):
        image = __import__("PIL.Image", fromlist=["Image"]).new("RGB", (100, 120), "#3A78B8")
        for x in range(80, 100):
            for y in range(55, 120):
                image.putpixel((x, y), (8, 8, 8))
        cleaned = _trim_dark_side_canvas(image)
        self.assertEqual(cleaned.size, (100, 120))

    def test_photo_cleanup_transparency_is_composited_on_white(self):
        image = __import__("PIL.Image", fromlist=["Image"]).new("RGBA", (100, 120), (58, 120, 184, 255))
        for x in range(80, 100):
            for y in range(120):
                image.putpixel((x, y), (0, 0, 0, 0))
        cleaned = _trim_dark_side_canvas(image)
        self.assertEqual(cleaned.size, (100, 120))
        self.assertEqual(cleaned.getpixel((90, 60)), (255, 255, 255))

    def test_delivery_package_with_photo_preserves_single_page_and_readable_font(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            photo = root / "photo.png"
            __import__("PIL.Image", fromlist=["Image"]).new("RGB", (300, 420), "#4A90E2").save(photo)
            resume = "# 张三\n\n**求职方向：大模型应用研发**\n\n## 教育背景\n\n某大学｜本科\n\n## 实习经历\n\n某公司｜研发实习生"
            result = export_delivery_package(resume, root, photo_path=photo)
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["page_count"], 1)

    def test_pdf_photo_zone_has_white_background_and_no_text_collision(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            photo = root / "photo.png"
            pdf = root / "resume.pdf"
            png = root / "resume.png"
            __import__("PIL.Image", fromlist=["Image"]).new("RGB", (300, 420), "#4A90E2").save(photo)
            content = (
                "# " + "超长候选人标题验证照片区域自动避让" * 5 + "\n\n"
                "**求职方向：AI Agent / 大模型应用研发**\n\n"
                "## 教育背景\n\n某财经大学｜投资学｜本科"
            )
            export_single_page_pdf(content, pdf, photo_path=photo)
            render_pdf_preview(pdf, png)
            import fitz
            document = fitz.open(pdf)
            try:
                page = document[0]
                width = page.rect.width
                photo_zone = fitz.Rect(width - 45 - 64, 34, width - 45, 34 + 82)
                words = page.get_text("words")
                stressed_words = [word for word in words if word[1] < photo_zone.y1]
                self.assertGreaterEqual(len({round(word[1]) for word in stressed_words}), 3)
                self.assertGreater(max(word[2] for word in stressed_words), 430)
                for word in words:
                    self.assertFalse(fitz.Rect(word[:4]).intersects(photo_zone), word[4])
            finally:
                document.close()
            with __import__("PIL.Image", fromlist=["Image"]).open(png) as rendered:
                self.assertEqual(rendered.getpixel((2, 2)), (255, 255, 255))

    def test_pdf_photo_export_reopens_temp_image_on_windows(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            photo = root / "photo.png"
            pdf = root / "resume.pdf"
            __import__("PIL.Image", fromlist=["Image"]).new("RGB", (120, 160), "#4A90E2").save(photo)
            export_single_page_pdf("# Name\n\n## Experience\n\nBuilt reliable systems", pdf, photo_path=photo)
            self.assertTrue(pdf.is_file())
            self.assertGreater(pdf.stat().st_size, 0)

    def test_delivery_ats_bytes_preserve_utf8_newlines(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            resume = "# Name\n\n## Experience\n\nBuilt reliable systems\n"
            result = export_delivery_package(resume, root)
            self.assertEqual(result["status"], "pass")
            self.assertEqual((root / "final-resume-ats.md").read_bytes(), resume.encode("utf-8"))

    def test_fsync_file_accepts_windows_file_handles(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.bin"
            path.write_bytes(b"artifact")
            from delivery import _fsync_file
            _fsync_file(path)

    def test_write_private_text_preserves_utf8_newlines(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "resume.md"
            main_module.write_private_text(path, "# Name\n\nBody\n")
            self.assertEqual(path.read_bytes(), b"# Name\n\nBody\n")

    def test_delivery_qa_rejects_png_replaced_after_render(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "resume.pdf"
            png = root / "resume.png"
            content = "# 张三\n\n## 教育背景\n\n某大学｜本科"
            export_single_page_pdf(content, pdf)
            render_pdf_preview(pdf, png)
            __import__("PIL.Image", fromlist=["Image"]).new(
                "RGB", (1191, 1684), "white"
            ).save(png)
            result = qa_delivery(content, pdf, png)
            self.assertEqual(result["status"], "fail")
            self.assertIn("PNG预览与PDF内容不一致", result["findings"])

    def test_langgraph_job_title_stays_inside_validated_graph(self):
        try:
            result = run_langgraph_workflow(
                "张三\n个人信息\n张三\n工作经历\n某公司｜研发实习生",
                "岗位名称：大模型应用研发实习生",
                job_title="大模型应用研发",
            )
        except RuntimeError:
            self.skipTest("LangGraph 未安装")
        self.assertEqual(result["violations"], [])
        self.assertIn("**求职方向：大模型应用研发**", result["final_resume"])

    def test_delivery_refuses_to_shrink_below_readable_font(self):
        with TemporaryDirectory() as directory:
            content = "# 候选人\n\n## 项目经历\n\n" + "很长的项目描述。" * 2500
            with self.assertRaisesRegex(ValueError, "10.5pt"):
                export_single_page_pdf(content, Path(directory) / "resume.pdf")

    def test_fact_ledger_is_stable_and_traceable(self):
        facts = build_fact_ledger("张三\n教育背景\n投资学本科\n技能\nPython")
        self.assertEqual([item["fact_id"] for item in facts], ["F001", "F002", "F003"])
        self.assertEqual(facts[1]["section"], "教育背景")

    def test_llm_revision_requires_valid_evidence_ids(self):
        result = {"resume_revisions": [{
            "section": "技能证书", "content": "Python", "evidence_ids": ["F003"]
        }]}
        accepted, findings = validate_llm_revisions(
            "张三\n教育背景\n投资学本科\n技能\nPython", result
        )
        self.assertEqual(len(accepted), 1)
        self.assertEqual(findings, [])
        rejected, findings = validate_llm_revisions(
            "张三\n技能\nPython",
            {"resume_revisions": [{"section": "技能证书", "content": "Python", "evidence_ids": ["F999"]}]},
        )
        self.assertEqual(rejected, [])
        self.assertTrue(findings)

    def test_llm_revision_rejects_unverified_number(self):
        accepted, findings = validate_llm_revisions(
            "项目经历\n整理公司资料",
            {"resume_revisions": [{"section": "项目经历", "content": "整理100家公司资料", "evidence_ids": ["F001"]}]},
        )
        self.assertEqual(accepted, [])
        self.assertTrue(any("100家" in item for item in findings))

    def test_llm_revision_rejects_unsupported_natural_language_claims(self):
        accepted, findings = validate_llm_revisions(
            "张三\n技能\nPython",
            {"resume_revisions": [{
                "section": "技能证书",
                "content": "精通 Python，具备领导力并主导跨部门协作。",
                "evidence_ids": ["F002"],
            }]},
        )
        self.assertEqual(accepted, [])
        self.assertTrue(any("未支持的断言" in item or "人工核验" in item for item in findings))

    def test_parse_llm_json_rejects_empty_evidence_ids(self):
        raw = json.dumps({
            "conclusion": "待核对", "strengths": [], "gaps": [],
            "resume_revisions": [{"section": "技能证书", "content": "Python", "evidence_ids": []}],
        }, ensure_ascii=False)
        with self.assertRaises(ValueError):
            parse_llm_json(raw)

    def test_delivery_qa_rejects_different_pdf_content(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_path, png_path, ats_path = root / "resume.pdf", root / "resume.png", root / "resume.md"
            export_single_page_pdf("# 甲\n\n甲甲甲", pdf_path)
            render_pdf_preview(pdf_path, png_path)
            ats_path.write_text("# 乙\n\n乙乙乙", encoding="utf-8")
            result = qa_delivery("# 乙\n\n乙乙乙", pdf_path, png_path, ats_path)
            self.assertEqual(result["status"], "fail")
            self.assertTrue(any("PDF提取文本" in item for item in result["findings"]))

    def test_delivery_qa_rejects_mismatched_ats(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_path, png_path, ats_path = root / "resume.pdf", root / "resume.png", root / "resume.md"
            source = "# 甲\n\n正文"
            export_single_page_pdf(source, pdf_path)
            render_pdf_preview(pdf_path, png_path)
            ats_path.write_text("另一个版本", encoding="utf-8")
            result = qa_delivery(source, pdf_path, png_path, ats_path)
            self.assertEqual(result["status"], "fail")
            self.assertIn("ATS Markdown 与最终简历不一致", result["findings"])

    def test_delivery_qa_rejects_lost_ascii_token_space(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_path, png_path, ats_path = root / "resume.pdf", root / "resume.png", root / "resume.md"
            export_single_page_pdf("# AB", pdf_path)
            render_pdf_preview(pdf_path, png_path)
            ats_path.write_text("# A B", encoding="utf-8")
            result = qa_delivery("# A B", pdf_path, png_path, ats_path)
            self.assertEqual(result["status"], "fail")

    def test_delivery_package_failure_does_not_publish_partial_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("delivery.qa_delivery", return_value={"status": "fail", "findings": ["测试失败"]}):
                with self.assertRaises(ValueError):
                    export_delivery_package("# 候选人\n\n正文", root)
            self.assertFalse((root / "final-resume-ats.md").exists())
            self.assertFalse((root / "final-resume.pdf").exists())
            self.assertFalse((root / "final-resume.png").exists())

    def test_delivery_publish_rolls_back_all_files_on_replace_failure(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            names = ("final-resume-ats.md", "final-resume.pdf", "final-resume.png")
            for name in names:
                (root / name).write_bytes(("old-" + name).encode())
            original = {name: (root / name).read_bytes() for name in names}
            real_replace = os.replace
            published = 0

            def fail_on_second_publish(source, target):
                nonlocal published
                if Path(source).name.startswith("final-resume"):
                    published += 1
                    if published == 2:
                        raise OSError("replace failed")
                return real_replace(source, target)

            with patch("delivery.os.replace", side_effect=fail_on_second_publish):
                with self.assertRaises(OSError):
                    export_delivery_package("# 候选人\n\n正文", root)
            self.assertEqual(original, {name: (root / name).read_bytes() for name in names})

    def test_delivery_publish_rolls_back_each_publish_position(self):
        names = ("final-resume-ats.md", "final-resume.pdf", "final-resume.png")
        for fail_position in (1, 2, 3):
            with self.subTest(fail_position=fail_position), TemporaryDirectory() as directory:
                root = Path(directory)
                for name in names:
                    (root / name).write_bytes(("old-" + name).encode())
                original = {name: (root / name).read_bytes() for name in names}
                real_replace = os.replace
                published = 0

                def fail_publish(source, target):
                    nonlocal published
                    if Path(source).name.startswith("final-resume"):
                        published += 1
                        if published == fail_position:
                            raise OSError("replace failed")
                    return real_replace(source, target)

                with patch("delivery.os.replace", side_effect=fail_publish):
                    with self.assertRaises(OSError):
                        export_delivery_package("# 候选人\n\n正文", root)
                self.assertEqual(original, {name: (root / name).read_bytes() for name in names})

    def test_delivery_publish_without_old_suite_leaves_nothing_on_failure(self):
        names = ("final-resume-ats.md", "final-resume.pdf", "final-resume.png")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            real_replace = os.replace
            published = 0

            def fail_second_publish(source, target):
                nonlocal published
                if Path(source).name.startswith("final-resume"):
                    published += 1
                    if published == 2:
                        raise OSError("replace failed")
                return real_replace(source, target)

            with patch("delivery.os.replace", side_effect=fail_second_publish):
                with self.assertRaises(OSError):
                    export_delivery_package("# 候选人\n\n正文", root)
            self.assertTrue(all(not (root / name).exists() for name in names))

    def test_delivery_keyboard_interrupt_rolls_back_old_suite(self):
        names = ("final-resume-ats.md", "final-resume.pdf", "final-resume.png")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in names:
                (root / name).write_bytes(("old-" + name).encode())
            original = {name: (root / name).read_bytes() for name in names}
            real_replace = os.replace
            published = 0

            def interrupt_second_publish(source, target):
                nonlocal published
                if Path(source).name.startswith("final-resume"):
                    published += 1
                    if published == 2:
                        raise KeyboardInterrupt
                return real_replace(source, target)

            with patch("delivery.os.replace", side_effect=interrupt_second_publish):
                with self.assertRaises(KeyboardInterrupt):
                    export_delivery_package("# 候选人\n\n正文", root)
            self.assertEqual(original, {name: (root / name).read_bytes() for name in names})

    def test_delivery_publishes_immutable_release_and_atomic_pointer(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            qa = export_delivery_package("# 候选人\n\n正文", root)
            pointer = json.loads((root / "current-release.json").read_text(encoding="utf-8"))
            release_dir = root / pointer["path"]
            self.assertEqual(pointer, qa["release"])
            self.assertTrue(release_dir.is_dir())
            if os.name != "nt":
                self.assertEqual(release_dir.stat().st_mode & 0o777, 0o700)
            for name in ("final-resume-ats.md", "final-resume.pdf", "final-resume.png"):
                self.assertEqual((release_dir / name).read_bytes(), (root / name).read_bytes())
                if os.name != "nt":
                    self.assertEqual((release_dir / name).stat().st_mode & 0o777, 0o600)
            manifest = release_dir / "release-manifest.json"
            self.assertEqual(
                pointer["manifest_sha256"],
                __import__("hashlib").sha256(manifest.read_bytes()).hexdigest(),
            )

    def test_delivery_pointer_failure_preserves_previous_authoritative_release(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            export_delivery_package("# 旧版\n\n正文", root)
            previous_pointer = (root / "current-release.json").read_bytes()
            previous_release = json.loads(previous_pointer)["path"]
            real_replace = os.replace

            def fail_pointer_replace(source, target):
                if Path(target).name == "current-release.json":
                    raise OSError("pointer replace failed")
                return real_replace(source, target)

            with patch("delivery.os.replace", side_effect=fail_pointer_replace):
                with self.assertRaises(OSError):
                    export_delivery_package("# 新版\n\n正文", root)
            self.assertEqual((root / "current-release.json").read_bytes(), previous_pointer)
            self.assertTrue((root / previous_release).is_dir())

    def test_delivery_rejects_symlink_release_paths(self):
        with TemporaryDirectory() as directory, TemporaryDirectory() as outside:
            root = Path(directory)
            try:
                (root / "releases").symlink_to(Path(outside), target_is_directory=True)
            except OSError as error:
                if os.name == "nt" and getattr(error, "winerror", None) == 1314:
                    self.skipTest("当前 Windows 账户未启用创建符号链接权限")
                raise
            with self.assertRaises(ValueError):
                export_delivery_package("# 候选人\n\n正文", root)

    def test_delivery_directory_fsync_failure_restores_previous_pointer(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            export_delivery_package("# 旧版\n\n正文", root)
            previous_pointer = (root / "current-release.json").read_bytes()
            real_fsync_directory = delivery_module._fsync_directory
            calls = 0

            def fail_output_sync(path):
                nonlocal calls
                calls += 1
                if Path(path) == root:
                    raise OSError("directory fsync failed")
                return real_fsync_directory(path)

            with patch("delivery._fsync_directory", side_effect=fail_output_sync):
                with self.assertRaisesRegex(RuntimeError, "指针持久化失败"):
                    export_delivery_package("# 新版\n\n正文", root)
            self.assertGreater(calls, 0)
            self.assertEqual((root / "current-release.json").read_bytes(), previous_pointer)

    def test_cli_manifest_has_external_checksum_and_private_permissions(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            process = subprocess.run(
                [sys.executable, "main.py", "--resume", "examples/resume.txt", "--jd", "examples/jd.txt",
                 "--mock-llm", "--export-package", "--output-dir", str(output)],
                cwd=Path(__file__).parent, capture_output=True, text=True, check=False,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            manifest = json.loads((output / "run-manifest.json").read_text(encoding="utf-8"))
            checksum = (output / "run-manifest.sha256").read_text(encoding="utf-8")
            self.assertIn("run-manifest.json", checksum)
            self.assertNotIn("run-manifest.json", manifest["produced"])
            if os.name != "nt":
                self.assertEqual(output.stat().st_mode & 0o777, 0o700)
                self.assertTrue(all(path.stat().st_mode & 0o777 == 0o600 for path in output.iterdir() if path.is_file()))

    def test_cli_unknown_error_is_sanitized_and_exit_two(self):
        with patch.object(main_module, "_run_cli", side_effect=ValueError("PRIVATE-CONTENT")):
            with patch("sys.stderr") as stderr:
                with self.assertRaises(SystemExit) as raised:
                    main_module.main()
        self.assertEqual(raised.exception.code, 2)
        rendered = " ".join(str(call) for call in stderr.write.call_args_list)
        self.assertNotIn("PRIVATE-CONTENT", rendered)

    def test_cli_type_and_key_errors_are_sanitized_and_exit_two(self):
        for error in (TypeError("PRIVATE-TYPE"), KeyError("PRIVATE-KEY")):
            with self.subTest(error=type(error).__name__):
                with patch.object(main_module, "_run_cli", side_effect=error):
                    with patch("sys.stderr") as stderr:
                        with self.assertRaises(SystemExit) as raised:
                            main_module.main()
                self.assertEqual(raised.exception.code, 2)
                rendered = " ".join(str(call) for call in stderr.write.call_args_list)
                self.assertNotIn("PRIVATE-", rendered)

    def test_manifest_excludes_unmanaged_legacy_files(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            output.mkdir()
            (output / "legacy-suite.md").write_text("legacy", encoding="utf-8")
            process = subprocess.run(
                [sys.executable, "main.py", "--resume", "examples/resume.txt", "--jd", "examples/jd.txt",
                 "--mock-llm", "--export-package", "--output-dir", str(output)],
                cwd=Path(__file__).parent, capture_output=True, text=True, check=False,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            manifest = json.loads((output / "run-manifest.json").read_text(encoding="utf-8"))
            self.assertNotIn("legacy-suite.md", manifest["produced"])
            self.assertNotIn("legacy-suite.md", manifest["artifacts"])

    def test_final_resume_rejects_semantic_rewrite_without_exact_support(self):
        source = "张三\n技能\nPython"
        result = {"resume_revisions": [{
            "section": "技能证书", "content": "Python 数据处理", "evidence_ids": ["F002"]
        }]}
        resume = build_final_resume(source, result, "职位名称：数据分析")
        self.assertNotIn("Python 数据处理", resume)
        self.assertIn("Python", resume)

    def test_parse_resume_sections(self):
        text = "教育背景\n投资学本科\n项目经历\n数据项目"
        result = parse_resume(text)
        self.assertEqual(result["教育背景"], ["投资学本科"])
        self.assertEqual(result["项目经历"], ["数据项目"])

    def test_parse_resume_supports_common_section_aliases_and_preamble(self):
        text = "# 张三\n联系方式\n13800000000\n实习经历\n某公司实习\n专业技能\nExcel"
        result = parse_resume(text)
        self.assertIn("张三", result["个人信息"])
        self.assertIn("某公司实习", result["工作经历"])
        self.assertIn("Excel", result["技能证书"])

    def test_parse_resume_supports_english_markdown_headings_and_name(self):
        result = parse_resume("**Jane Doe**\n## Education\nBSc\n## Experience\nX")
        self.assertEqual(result["个人信息"], ["Jane Doe"])
        self.assertEqual(result["教育背景"], ["BSc"])
        self.assertEqual(result["工作经历"], ["X"])

    def test_parse_resume_supports_numbered_and_colon_headings(self):
        result = parse_resume("一、教育背景：\n投资学本科\n### **个人优势**\n沟通协调")
        self.assertEqual(result["教育背景"], ["投资学本科"])
        self.assertEqual(result["自我评价"], ["沟通协调"])

    def test_prepare_output_dir_removes_stale_runtime_artifacts_only(self):
        with TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            (output / "final-resume.md").write_text("old", encoding="utf-8")
            (output / "custom-delivery.png").write_text("keep", encoding="utf-8")
            prepare_output_dir(output)
            self.assertFalse((output / "final-resume.md").exists())
            self.assertTrue((output / "custom-delivery.png").exists())

    def test_prepare_output_dir_rejects_symlink_directory(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            real = root / "real"
            real.mkdir()
            link = root / "link"
            try:
                link.symlink_to(real, target_is_directory=True)
            except OSError as error:
                if os.name == "nt" and getattr(error, "winerror", None) == 1314:
                    self.skipTest("当前 Windows 账户未启用创建符号链接权限")
                raise
            with self.assertRaisesRegex(ValueError, "符号链接"):
                prepare_output_dir(link)

    def test_prepare_output_dir_unlinks_stale_symlink_without_touching_target(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "private.txt"
            target.write_text("keep", encoding="utf-8")
            output = root / "output"
            output.mkdir()
            try:
                (output / "final-resume.md").symlink_to(target)
            except OSError as error:
                if os.name == "nt" and getattr(error, "winerror", None) == 1314:
                    self.skipTest("当前 Windows 账户未启用创建符号链接权限")
                raise
            prepare_output_dir(output)
            self.assertEqual(target.read_text(encoding="utf-8"), "keep")
            self.assertFalse((output / "final-resume.md").exists())

    def test_final_resume_does_not_invent_default_self_evaluation(self):
        source = "张三\n教育背景\n本科"
        result = build_final_resume(source, {}, "职位名称：区域销售\n销售策略")
        self.assertNotIn("市场营销与客户运营实习经历", result)
        self.assertNotIn("## 自我评价", result)

    def test_langgraph_rejects_invalid_llm_result_schema(self):
        with self.assertRaisesRegex(ValueError, "resume_revisions 必须是列表"):
            run_langgraph_workflow(
                "张三\n技能证书\nPython",
                "职位名称：开发",
                {"conclusion": "x", "strengths": [], "gaps": [], "resume_revisions": "invalid"},
            )

    def test_cli_accepts_custom_output_dir(self):
        from main import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--output-dir", type=Path, default=None)
        args = parser.parse_args(["--output-dir", "custom-output"])
        self.assertEqual(args.output_dir, Path("custom-output"))

    def test_r_does_not_match_inside_other_words(self):
        result = match_keywords("Python 和 GitHub", ["R", "Python"])
        self.assertFalse(result["R"])
        self.assertTrue(result["Python"])

    def test_english_keywords_use_word_boundaries(self):
        result = match_keywords("Micropython is excellent at SQL-like work", ["Python", "Excel", "SQL"])
        self.assertFalse(result["Python"])
        self.assertFalse(result["Excel"])
        self.assertTrue(result["SQL"])

    def test_analysis_contains_match_matrix(self):
        result = build_analysis("Python 数据分析 GitHub", "岗位职责：数据分析")
        self.assertGreaterEqual(len(result["match_matrix"]), 1)
        self.assertIn("数据分析", result["match_matrix"][0]["requirement"])

    def test_semantic_match_matrix_binds_fact_ids(self):
        result = build_analysis(
            "张三\n工作经历\n负责客户关系维护与社群运营",
            "岗位职责：客户运营",
        )
        item = next(row for row in result["match_matrix"] if row["requirement"] == "客户运营")
        self.assertEqual(item["match_level"], "待补证据")
        self.assertEqual(item["evidence_ids"], [])
        self.assertEqual(item["semantic_evidence_ids"], ["F002"])
        self.assertIn("客户关系", item["matched_terms"])

    def test_weak_semantic_relation_still_requires_confirmation(self):
        cases = [
            ("负责合规审核流程", "风险控制"),
            ("负责制作业务报表", "数据分析"),
        ]
        for resume, requirement in cases:
            with self.subTest(requirement=requirement):
                result = build_analysis(f"张三\n工作经历\n{resume}", f"岗位要求：{requirement}")
                item = next(row for row in result["match_matrix"] if row["requirement"] == requirement)
                self.assertEqual(item["match_level"], "待补证据")
                self.assertEqual(item["evidence_ids"], [])
                self.assertTrue(item["semantic_evidence_ids"])
                self.assertTrue(any(q["requirement"] == requirement for q in result["confirmation_questions"]))

    def test_missing_requirement_creates_confirmation_question(self):
        result = build_analysis("张三\n技能\nExcel", "优先具备 Salesforce 经验")
        questions = result["confirmation_questions"]
        self.assertTrue(any(item["requirement"] == "Salesforce" for item in questions))
        report = build_confirmation_questions_report(questions)
        self.assertIn("确认前不会写入简历", report)
        self.assertIn("个人行动", report)

    def test_confirmed_evidence_requirement_has_no_question(self):
        result = build_analysis("张三\n技能\nSalesforce", "要求：Salesforce")
        self.assertFalse(any(item["requirement"] == "Salesforce" for item in result["confirmation_questions"]))

    def test_priority_certificate_is_priority_and_uses_certificate_question(self):
        result = build_analysis("张三\n技能\nExcel", "优先具备CFA证书")
        item = next(row for row in result["match_matrix"] if row["requirement"] == "CFA")
        self.assertEqual(item["requirement_type"], "优先条件")
        question = next(q for q in result["confirmation_questions"] if q["requirement"] == "CFA")
        self.assertIn("证书名称", question["question"])

    def test_non_algorithm_jd_uses_general_requirements(self):
        result = build_analysis("张三\n教育背景\n本科\n工作经历\n客户运营\n技能证书\nExcel", "财务岗位\n要求：Excel、客户运营")
        requirements = [item["requirement"] for item in result["match_matrix"]]
        self.assertIn("Excel", requirements)
        self.assertIn("客户运营", requirements)
        self.assertNotIn("机器学习与特征工程", requirements)
        self.assertTrue(result["keyword_matches"]["Excel"])

    def test_jd_extracts_unknown_tool_without_fixed_template(self):
        result = build_analysis("张三\n技能\nExcel", "运营岗位\n要求：Tableau、Salesforce")
        matrix_text = " ".join(item["requirement"] for item in result["match_matrix"])
        self.assertIn("Tableau", matrix_text)
        self.assertIn("Salesforce", matrix_text)
        self.assertFalse(any(item["match_level"] == "较强匹配" for item in result["match_matrix"]))

    def test_jd_noise_words_are_not_keywords(self):
        result = build_analysis(
            "张三\n技能\nExcel",
            "岗位要求：熟练使用 Excel，具备良好沟通能力；负责客户运营",
        )
        keywords = set(result["keyword_matches"])
        self.assertNotIn("熟练", keywords)
        self.assertNotIn("负责", keywords)
        self.assertNotIn("岗位要求", keywords)

    def test_complex_jd_does_not_create_connector_or_title_questions(self):
        result = build_analysis(
            "示例候选人\n教育背景\n某财经大学 投资学 本科\n技能证书\nExcel",
            "职位名称：AI Agent 产品运营\n"
            "任职要求：本科及以上学历，专业不限\n"
            "优先具备 CFA 证书或 Salesforce 使用经验\n"
            "岗位职责：负责客户运营、数据分析与 AI Agent 工作流建设\n"
            "需具备风险控制经验，并能使用 Kubernetes 完成部署",
        )
        requirements = {item["requirement"] for item in result["match_matrix"]}
        self.assertTrue({"本科", "CFA", "Salesforce", "客户运营", "数据分析", "AI Agent", "工作流建设", "风险控制", "Kubernetes"} <= requirements)
        self.assertFalse({"职位名称", "优先具备", "证书或", "使用经验", "并能使用", "完成部署"} & requirements)
        questions = {item["requirement"]: item["question"] for item in result["confirmation_questions"]}
        self.assertIn("证书名称", questions["CFA"])
        self.assertIn("具体场景", questions["Salesforce"])

    def test_dynamic_chinese_tools_and_certificates_are_extracted(self):
        result = build_analysis(
            "张三\n技能证书\nExcel",
            "要求：熟练使用飞书多维表格、钉钉和企业微信；持有证券从业资格、教师资格证或普通话等级证书",
        )
        requirements = {item["requirement"] for item in result["match_matrix"]}
        self.assertTrue(
            {"飞书多维表格", "钉钉", "企业微信", "证券从业资格", "教师资格证", "普通话等级证书"}
            <= requirements
        )

    def test_complex_jd_preserves_relation_and_constraints(self):
        result = build_analysis(
            "张三\n教育背景\n本科\n技能证书\nPython",
            "任职要求：本科及以上学历，具备 Python 或 Java 经验；有 3 年以上项目经验优先",
        )
        items = {item["requirement"]: item for item in result["match_matrix"]}
        self.assertEqual(items["本科"]["requirement_type"], "硬性条件")
        self.assertEqual(items["Python"]["relation"], "OR")
        self.assertEqual(items["Java"]["relation"], "OR")
        self.assertEqual(items["Python"]["requirement_type"], "硬性条件")
        self.assertEqual(items["Java"]["requirement_type"], "硬性条件")
        self.assertEqual(items["工作经验"]["requirement_type"], "优先条件")
        self.assertEqual(items["工作经验"]["minimum_years"], 3)
        self.assertEqual(items["工作经验"]["operator"], ">=")

        self.assertEqual(items["Java"]["match_level"], "OR组已满足")
        self.assertNotIn("Java", {q["requirement"] for q in result["confirmation_questions"]})

    def test_comma_separated_requirements_keep_local_relation(self):
        result = build_analysis(
            "张三\n技能\nPython Docker",
            "任职要求：具备 Python 或 Java 经验，熟悉 Docker 和 Kubernetes",
        )
        items = {item["requirement"]: item for item in result["match_matrix"]}
        self.assertEqual(items["Python"]["relation"], "OR")
        self.assertEqual(items["Java"]["relation"], "OR")
        self.assertEqual(items["Docker"]["relation"], "AND")
        self.assertEqual(items["Kubernetes"]["relation"], "AND")
        self.assertEqual(items["Java"]["match_level"], "OR组已满足")
        self.assertEqual(items["Kubernetes"]["match_level"], "缺口")

    def test_year_constraint_requires_explicit_year_evidence(self):
        cases = [
            ("项目经验\n负责项目经验", "待补证据", None),
            ("项目经验\n具备2年项目经验", "不满足", 2),
            ("项目经验\n具备4年项目经验", "满足", 4),
        ]
        for resume, status, years in cases:
            with self.subTest(status=status):
                result = build_analysis(resume, "任职要求：3年以上项目经验")
                item = next(row for row in result["match_matrix"] if row["requirement"] == "项目经验")
                self.assertEqual(item["constraint_status"], status)
                self.assertEqual(item["evidenced_years"], years)
                if status == "满足":
                    self.assertEqual(item["match_level"], "较强匹配")
                else:
                    self.assertTrue(any(q["requirement"] == "项目经验" for q in result["confirmation_questions"]))

    def test_one_sided_and_range_year_constraints_are_evaluated(self):
        cases = [
            ("4年项目经验", "3年以上项目经验", ">=", 3, None, "满足"),
            ("4年项目经验", "3-5年项目经验", "between", 3, 5, "满足"),
            ("6年项目经验", "3-5年项目经验", "between", 3, 5, "不满足"),
            ("4年项目经验", "3年以内项目经验", "<=", None, 3, "不满足"),
        ]
        for resume, jd, operator, minimum, maximum, status in cases:
            with self.subTest(jd=jd, resume=resume):
                result = build_analysis(f"张三\n项目经验\n{resume}", f"任职要求：{jd}")
                item = next(row for row in result["match_matrix"] if row["requirement"] == "项目经验")
                self.assertEqual(item["operator"], operator)
                self.assertEqual(item["minimum_years"], minimum)
                self.assertEqual(item["maximum_years"], maximum)
                self.assertEqual(item["constraint_status"], status)

    def test_age_range_constraint_is_structured(self):
        result = build_analysis("张三\n本科", "任职要求：年龄18-35岁，本科及以上学历")
        age_item = next(item for item in result["match_matrix"] if item.get("age_range"))
        self.assertEqual(age_item["age_range"], [18, 35])
        self.assertEqual(age_item["operator"], "between")

    def test_education_level_comparison_uses_highest_explicit_degree(self):
        cases = [
            ("张三\n教育背景\n硕士", "本科", "满足"),
            ("张三\n教育背景\n本科", "硕士", "不满足"),
            ("张三\n教育背景\n某大学金融学", "本科", "待补证据"),
        ]
        for resume, requirement, status in cases:
            with self.subTest(requirement=requirement, status=status):
                result = build_analysis(resume, f"任职要求：{requirement}及以上学历")
                item = next(row for row in result["match_matrix"] if row["requirement"] == requirement)
                self.assertEqual(item["constraint_status"], status)
        higher = build_analysis("张三\n教育背景\n硕士", "任职要求：本科及以上学历")
        item = next(row for row in higher["match_matrix"] if row["requirement"] == "本科")
        self.assertEqual(item["evidenced_education"], "硕士")
        self.assertEqual(item["match_level"], "较强匹配")

    def test_nested_and_or_condition_keeps_inner_group_isolated(self):
        result = build_analysis(
            "张三\n技能\nPython Docker",
            "任职要求：具备 Python 且（Docker 或 Kubernetes）经验",
        )
        items = {item["requirement"]: item for item in result["match_matrix"]}
        self.assertEqual(items["Python"]["relation"], "AND")
        self.assertEqual(items["Docker"]["relation"], "OR")
        self.assertEqual(items["Kubernetes"]["relation"], "OR")
        self.assertEqual(items["Docker"]["group_id"], items["Kubernetes"]["group_id"])
        self.assertEqual(items["Kubernetes"]["match_level"], "OR组已满足")

    def test_and_group_exposes_unsatisfied_group_status(self):
        result = build_analysis("张三\n技能\nPython", "任职要求：Python 和 Docker")
        items = {item["requirement"]: item for item in result["match_matrix"]}
        self.assertEqual(items["Python"]["group_id"], items["Docker"]["group_id"])
        self.assertEqual(items["Python"]["group_status"], "未满足")
        self.assertEqual(items["Docker"]["group_status"], "未满足")

    def test_age_constraint_requires_explicit_age_evidence(self):
        result = build_analysis("张三\n教育背景\n本科", "任职要求：年龄18-35岁")
        item = next(row for row in result["match_matrix"] if row.get("age_range"))
        self.assertEqual(item["constraint_status"], "待补证据")
        self.assertTrue(any(q["requirement"] == "年龄" for q in result["confirmation_questions"]))

    def test_one_sided_age_constraints_are_structured_and_evaluated(self):
        cases = [
            ("27岁", "年龄25岁以上", ">=", 25, None, "满足"),
            ("36岁", "年龄不超过35岁", "<=", None, 35, "不满足"),
        ]
        for resume, jd, operator, minimum, maximum, status in cases:
            with self.subTest(jd=jd):
                result = build_analysis(f"张三\n{resume}", f"任职要求：{jd}")
                item = next(row for row in result["match_matrix"] if row["requirement"] == "年龄")
                self.assertEqual(item["operator"], operator)
                self.assertEqual(item["minimum_age"], minimum)
                self.assertEqual(item["maximum_age"], maximum)
                self.assertEqual(item["constraint_status"], status)

    def test_negative_jd_constraint_is_explicit(self):
        result = build_analysis("张三\n技能证书\nExcel", "不要求英语；无需销售经验")
        items = {item["requirement"]: item for item in result["match_matrix"]}
        self.assertTrue({"英语", "销售经验"} <= items.keys())
        self.assertTrue(all(items[name]["is_negative"] for name in ("英语", "销售经验")))
        self.assertTrue(all(items[name]["constraint_status"] == "不适用" for name in ("英语", "销售经验")))
        self.assertTrue(all(items[name]["match_level"] == "不构成门槛" for name in ("英语", "销售经验")))
        self.assertFalse({"英语", "销售经验"} & {q["requirement"] for q in result["confirmation_questions"]})

    def test_constraint_evaluator_preserves_evidence_source_by_constraint_kind(self):
        result = build_analysis(
            "张三\n教育背景\n硕士\n个人信息\n28岁\n项目经验\n4年项目经验",
            "任职要求：本科及以上学历；年龄18-35岁；3年以上项目经验",
        )
        items = {item["requirement"]: item for item in result["match_matrix"]}
        for requirement in ("本科", "年龄", "项目经验"):
            self.assertEqual(items[requirement]["constraint_status"], "满足")
            self.assertTrue(items[requirement]["evidence_ids"])
            self.assertEqual(items[requirement]["match_level"], "较强匹配")

    def test_multiword_tools_and_slash_certificates_are_normalized(self):
        result = build_analysis(
            "张三\n技能证书\nPower BI",
            "要求：熟悉 Power BI；持有 CFA/CPA 之一",
        )
        requirements = {item["requirement"] for item in result["match_matrix"]}
        self.assertIn("Power BI", requirements)
        self.assertIn("CFA", requirements)
        self.assertIn("CPA", requirements)
        self.assertNotIn("Power", requirements)
        self.assertNotIn("BI", requirements)
        certs = [item for item in result["match_matrix"] if item["requirement"] in {"CFA", "CPA"}]
        self.assertTrue(all(item["relation"] == "OR" for item in certs))

    def test_common_english_words_are_not_dynamic_requirements(self):
        result = build_analysis(
            "张三\n技能证书\nExcel",
            "Product Owner role requires good communication with Team; familiar with Salesforce",
        )
        requirements = {item["requirement"] for item in result["match_matrix"]}
        self.assertIn("Salesforce", requirements)
        self.assertFalse(
            {"Product", "Owner", "role", "good", "communication", "Team", "familiar", "with"}
            & requirements
        )

    def test_jd_does_not_use_short_generic_words_or_empty_fallback(self):
        result = build_analysis("有客户沟通经验", "岗位要求\n学历\n能力\n以及")
        keywords = set(result["keyword_matches"])
        self.assertNotIn("学历", keywords)
        self.assertNotIn("能力", keywords)
        self.assertNotIn("以及", keywords)
        self.assertEqual(result["match_matrix"], [])

    def test_prompt_contains_truth_boundary(self):
        result = build_analysis("Python", "算法工程师")
        prompt = build_llm_prompt("Python", "算法工程师", result)
        self.assertIn("不得编造", prompt)
        self.assertIn("岗位 JD", prompt)

    def test_mock_llm_returns_valid_json(self):
        result = parse_llm_json(mock_llm("测试提示词"))
        self.assertIn("Python 数据处理", result["strengths"])
        self.assertIsInstance(result["resume_revisions"], list)

    def test_llm_json_rejects_invalid_json(self):
        with self.assertRaisesRegex(ValueError, "不是有效 JSON"):
            parse_llm_json("不是 JSON")

    def test_llm_json_rejects_missing_field(self):
        with self.assertRaisesRegex(ValueError, "缺少字段"):
            parse_llm_json('{"conclusion": "只有结论"}')

    def test_llm_json_rejects_non_object_top_level(self):
        with self.assertRaisesRegex(ValueError, "顶层必须是对象"):
            parse_llm_json('[]')

    def test_targeted_resume_draft_passes_truth_gate(self):
        result = parse_llm_json(mock_llm("测试提示词"))
        draft = build_targeted_resume_draft("原始简历", result)
        self.assertIn("候选稿", draft)
        self.assertEqual(validate_resume_draft(draft), [])

    def test_llm_alignment_report_distinguishes_adopted_and_pending(self):
        result = {
            "resume_revisions": [
                {"section": "技能证书", "content": "Python", "evidence_ids": ["F002"]},
                {"section": "项目经历", "content": "完成推荐系统开发", "evidence_ids": ["F001"]},
                {"section": "不建议当前添加的内容", "content": "熟练掌握 Hadoop", "evidence_ids": ["F001"]},
            ]
        }
        report = build_llm_alignment_report(
            result,
            "## 技能证书\nPython\n## 项目经历\n数据整理",
            "技能证书\nPython\n项目经历\n数据整理",
        )
        self.assertIn("已采纳", report)
        self.assertIn("已拒绝", report)

    def test_revision_diff_report_marks_adopted_and_rejected(self):
        result = {
            "resume_revisions": [
                {"section": "技能证书", "content": "Python", "evidence_ids": ["F001"]},
                {"section": "项目经历", "content": "主导推荐系统", "evidence_ids": ["F001"]},
            ]
        }
        report = build_revision_diff_report(
            "技能证书\nPython",
            result,
            "## 技能证书\nPython\n",
        )
        self.assertIn("| 采纳 |", report)
        self.assertIn("| 拒绝 |", report)
        self.assertIn("F001", report)
        self.assertIn("自动拒绝明细", report)

    def test_revision_diff_does_not_mark_unknown_section_as_adopted(self):
        source = "张三\n技能\nPython"
        result = {
            "resume_revisions": [
                {"section": "自定义模块", "content": "Python", "evidence_ids": ["F002"]},
            ]
        }
        final_resume = build_final_resume(source, result, "职位名称：数据分析")
        report = build_revision_diff_report(source, result, final_resume)
        self.assertIn("| 拒绝 |", report)
        self.assertNotIn("| 采纳 |", report)

    def test_evidence_mapping_report_marks_direct_and_manual_review(self):
        report = build_evidence_mapping_report(
            "工作经历\n负责2个客户群维护",
            "## 工作经历\n负责2个客户群维护\n具备优秀销售能力",
        )
        self.assertIn("直接命中原文", report)
        self.assertIn("需人工核对", report)

    def test_evidence_mapping_report_marks_partial_rewrite(self):
        report = build_evidence_mapping_report(
            "负责2个客户群维护并跟进客户",
            "维护客户关系并持续跟进客户需求",
        )
        self.assertIn("部分命中，需核对改写", report)

    def test_truth_gate_detects_unverified_claim(self):
        self.assertIn("风险控制模型", validate_resume_draft("完成风险控制模型"))

    def test_truth_gate_detects_rephrased_unverified_claim(self):
        self.assertTrue(validate_resume_draft("熟练掌握 Hadoop 与 Spark"))

    def test_evidence_gate_detects_new_number(self):
        findings = validate_resume_evidence("参与数据整理", "参与数据整理，完成100家资料处理")
        self.assertTrue(any("100家" in item for item in findings))

    def test_evidence_gate_allows_source_numbers(self):
        self.assertEqual(
            validate_resume_evidence("负责2个客户群维护", "负责2个客户群维护"), []
        )

    def test_final_resume_contains_required_sections_and_truth_boundary(self):
        resume = build_final_resume()
        for section in ("个人信息", "求职意向", "教育背景", "项目经历", "工作经历", "技能证书", "自我评价"):
            self.assertIn(section, resume)
        self.assertNotIn("## 个人概述", resume)
        self.assertIn("脱敏演示版本", resume)
        self.assertNotIn("个人真实姓名", resume)
        self.assertEqual(validate_resume_draft(resume), [])

    def test_final_resume_can_be_built_from_input_resume(self):
        resume = build_final_resume("教育背景\n投资学本科\n项目经历\n数据整理", {}, "算法岗位")
        self.assertIn("## 教育背景", resume)
        self.assertIn("投资学本科", resume)
        self.assertIn("## 项目经历", resume)

    def test_final_resume_uses_dynamic_name_and_sections(self):
        resume = build_final_resume("张三\n联系方式\n13800000000\n实习经历\n财务助理", {}, "财务岗位")
        self.assertTrue(resume.startswith("# 张三"))
        self.assertIn("## 工作经历", resume)

    def test_render_resume_image_rejects_header_collision(self):
        with TemporaryDirectory() as temp_dir:
            photo = Path(temp_dir) / "photo.jpg"
            output = Path(temp_dir) / "resume.png"
            from PIL import Image
            Image.new("RGB", (100, 100), "blue").save(photo)
            with patch("PIL.ImageDraw.ImageDraw.textbbox", return_value=(1300, 50, 1500, 100)):
                with self.assertRaisesRegex(ValueError, "重叠"):
                    render_resume_image("# 标题", photo, output)

    def test_render_resume_image_fits_long_dynamic_header(self):
        with TemporaryDirectory() as temp_dir:
            photo = Path(temp_dir) / "photo.jpg"
            output = Path(temp_dir) / "resume.png"
            from PIL import Image
            Image.new("RGB", (100, 100), "blue").save(photo)
            render_resume_image(
                "张三\n联系方式\n13800000000｜very-long@example.com\n教育背景\n本科",
                photo,
                output,
                "一个非常非常长的岗位名称以及额外方向说明",
            )
            self.assertTrue(output.is_file())

    def test_render_resume_image_handles_very_long_resume_without_fixed_height(self):
        with TemporaryDirectory() as temp_dir:
            photo = Path(temp_dir) / "photo.jpg"
            output = Path(temp_dir) / "resume.png"
            from PIL import Image
            Image.new("RGB", (100, 100), "blue").save(photo)
            text = "张三\n教育背景\n" + ("本科经历与课程说明。" * 500)
            render_resume_image(text, photo, output)
            with Image.open(output) as image:
                self.assertGreater(image.height, 2600)

    def test_render_resume_image_supports_sales_and_ats_templates(self):
        with TemporaryDirectory() as temp_dir:
            photo = Path(temp_dir) / "photo.jpg"
            sales = Path(temp_dir) / "sales.png"
            ats = Path(temp_dir) / "ats.png"
            from PIL import Image
            Image.new("RGB", (100, 100), "blue").save(photo)
            text = "张三\n教育背景\n投资学本科\n技能证书\nPython"
            render_resume_image(text, photo, sales, template="sales")
            render_resume_image(text, photo, ats, template="ats")
            with Image.open(sales) as sales_image, Image.open(ats) as ats_image:
                self.assertEqual(sales_image.size, ats_image.size)
                self.assertNotEqual(sales_image.getpixel((10, 10)), ats_image.getpixel((10, 10)))

    def test_render_resume_image_wraps_long_ascii_token_without_overflow(self):
        with TemporaryDirectory() as temp_dir:
            photo = Path(temp_dir) / "photo.jpg"
            output = Path(temp_dir) / "resume.png"
            from PIL import Image
            Image.new("RGB", (100, 100), "blue").save(photo)
            long_url = "https://example.com/" + "a" * 500
            render_resume_image("张三\n项目经历\n" + long_url, photo, output)
            with Image.open(output) as image:
                self.assertEqual(image.width, 1600)

    def test_python_workflow_is_framework_free_and_validated(self):
        result = run_python_workflow("教育背景\n投资学本科", "算法岗位")
        self.assertIn("analysis", result)
        self.assertIn("投资学本科", result["final_resume"])
        self.assertEqual(validate_resume_draft(result["final_resume"]), [])

    def test_python_and_langgraph_match_analysis_are_consistent(self):
        resume = "教育背景\n投资学本科\n技能\nPython、Excel"
        jd = "数据分析岗位\n要求：Python、Excel"
        python_result = run_python_workflow(resume, jd)
        try:
            langgraph_result = run_langgraph_workflow(resume, jd)
        except RuntimeError:
            self.skipTest("LangGraph 未安装")
        self.assertEqual(
            python_result["analysis"]["keyword_matches"],
            langgraph_result["analysis"]["keyword_matches"],
        )
        self.assertEqual(python_result["final_resume"], langgraph_result["final_resume"])

    def test_langgraph_is_optional_and_explicit(self):
        try:
            workflow = build_langgraph_workflow()
        except RuntimeError as error:
            self.assertIn("未安装 LangGraph", str(error))
        else:
            self.assertIsNotNone(workflow)

    @patch.dict(os.environ, {}, clear=True)
    def test_llm_requires_environment_variables(self):
        with self.assertRaisesRegex(RuntimeError, "缺少 API_BASE_URL"):
            call_llm("测试")

    @patch.dict(
        os.environ,
        {"API_BASE_URL": "http://127.0.0.1:1", "API_KEY": "test-only", "MODEL": "test-model"},
        clear=True,
    )
    def test_llm_connection_error_is_explained(self):
        with patch("llm_client.request.urlopen", side_effect=URLError("test failure")):
            with self.assertRaisesRegex(RuntimeError, "模型接口连接失败"):
                call_llm("测试")

    @patch.dict(
        os.environ,
        {"API_BASE_URL": "http://example.invalid", "API_KEY": "test", "MODEL": "test-model"},
        clear=True,
    )
    def test_llm_http_error_does_not_leak_response_body(self):
        response = MagicMock()
        response.read.return_value = b"secret prompt echo"
        error_obj = HTTPError("http://example.invalid", 500, "failure", {}, response)
        with patch("llm_client.request.urlopen", side_effect=error_obj):
            with self.assertRaisesRegex(RuntimeError, "模型接口返回 HTTP 500") as caught:
                call_llm("测试")
        self.assertNotIn("secret prompt echo", str(caught.exception))

    @patch.dict(
        os.environ,
        {"API_BASE_URL": "https://secret-host.invalid/v1", "API_KEY": "secret-key", "MODEL": "secret-model"},
        clear=True,
    )
    def test_llm_log_does_not_record_endpoint_model_key_or_prompt(self):
        with TemporaryDirectory() as directory:
            log_path = Path(directory) / "llm-call.log"
            with patch("llm_client.Path", return_value=log_path.parent / "llm_client.py"):
                with patch("llm_client.request.urlopen", side_effect=URLError("offline")):
                    with self.assertRaises(RuntimeError):
                        call_llm("PRIVATE-PROMPT")
            content = (log_path.parent / "output" / "llm-call.log").read_text(encoding="utf-8")
            self.assertNotIn("secret-host", content)
            self.assertNotIn("secret-model", content)
            self.assertNotIn("secret-key", content)
            self.assertNotIn("PRIVATE-PROMPT", content)

    def test_llm_log_rejects_symlink(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            module_path = root / "llm_client.py"
            output = root / "output"
            output.mkdir()
            target = root / "target.log"
            target.write_text("keep", encoding="utf-8")
            try:
                (output / "llm-call.log").symlink_to(target)
            except OSError as error:
                if os.name == "nt" and getattr(error, "winerror", None) == 1314:
                    self.skipTest("当前 Windows 账户未启用创建符号链接权限")
                raise
            with patch("llm_client.Path", return_value=module_path):
                with self.assertRaisesRegex(RuntimeError, "符号链接"):
                    llm_client._write_log("request started=true")
            self.assertEqual(target.read_text(encoding="utf-8"), "keep")

    def test_cli_invalid_utf8_writes_error_to_stderr(self):
        with TemporaryDirectory() as directory:
            bad = Path(directory) / "bad.txt"
            bad.write_bytes(b"\xff\xfe")
            process = subprocess.run(
                [sys.executable, "main.py", "--resume", str(bad), "--jd", "examples/jd.txt"],
                cwd=Path(__file__).parent, capture_output=True, text=True, check=False,
            )
            self.assertEqual(process.returncode, 2)
            self.assertIn("UTF-8", process.stderr)
            self.assertNotIn("UTF-8", process.stdout)


if __name__ == "__main__":
    unittest.main()
