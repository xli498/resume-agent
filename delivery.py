"""同步生成 ATS Markdown、单页 A4 PDF、PNG 预览并执行最小交付 QA。"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


DELIVERY_NAMES = ("final-resume-ats.md", "final-resume.pdf", "final-resume.png")


def _flush_windows_directory(path: Path) -> None:
    """Flush a directory handle so rename metadata reaches stable storage."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
        wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    ]
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.FlushFileBuffers.argtypes = [wintypes.HANDLE]
    kernel32.FlushFileBuffers.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.CreateFileW(
        str(path),
        0x40000000,  # GENERIC_WRITE
        0x00000001 | 0x00000002 | 0x00000004,  # share read/write/delete
        None,
        3,  # OPEN_EXISTING
        0x02000000,  # FILE_FLAG_BACKUP_SEMANTICS (directory handle)
        None,
    )
    invalid = ctypes.c_void_p(-1).value
    if handle in (None, invalid) or getattr(handle, "value", handle) == invalid:
        error = ctypes.get_last_error()
        raise OSError(error, f"无法打开目录句柄：{path}")
    try:
        if not kernel32.FlushFileBuffers(handle):
            error = ctypes.get_last_error()
            raise OSError(error, f"无法刷新目录句柄：{path}")
    finally:
        kernel32.CloseHandle(handle)


def _fsync_directory(path: Path) -> None:
    """同步目录项；无法确认持久化时让发布明确失败。"""
    if os.name == "nt":
        _flush_windows_directory(path)
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_file(path: Path) -> None:
    # Windows requires a writable handle for FlushFileBuffers/os.fsync.
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())


def _publish_release_snapshot(staging_dir: Path, output_dir: Path) -> dict[str, Any]:
    """发布版本化快照，再以单个普通 JSON 文件原子切换权威版本。"""
    releases_dir = output_dir / "releases"
    pointer_path = output_dir / "current-release.json"
    if releases_dir.is_symlink() or pointer_path.is_symlink():
        raise ValueError("交付版本路径不能是符号链接")
    releases_dir.mkdir(mode=0o700, exist_ok=True)
    releases_dir.chmod(0o700)
    release_id = secrets.token_hex(12)
    release_dir = releases_dir / release_id
    release_dir.mkdir(mode=0o700)
    artifacts: dict[str, dict[str, Any]] = {}
    pointer_switched = False
    pointer_temp = output_dir / f".current-release-{release_id}.tmp"
    previous_pointer = pointer_path.read_bytes() if pointer_path.exists() else None
    try:
        for name in DELIVERY_NAMES:
            source = staging_dir / name
            target = release_dir / name
            target.write_bytes(source.read_bytes())
            target.chmod(0o600)
            _fsync_file(target)
            artifacts[name] = {
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "bytes": target.stat().st_size,
                "mode": oct(target.stat().st_mode & 0o777),
            }
        release_manifest = {
            "release_id": release_id,
            "artifacts": artifacts,
        }
        manifest_path = release_dir / "release-manifest.json"
        manifest_path.write_text(
            json.dumps(release_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest_path.chmod(0o600)
        _fsync_file(manifest_path)
        _fsync_directory(release_dir)
        _fsync_directory(releases_dir)
        pointer = {
            "release_id": release_id,
            "path": f"releases/{release_id}",
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        }
        pointer_temp.write_text(
            json.dumps(pointer, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        pointer_temp.chmod(0o600)
        _fsync_file(pointer_temp)
        os.replace(pointer_temp, pointer_path)
        pointer_switched = True
        try:
            _fsync_directory(output_dir)
        except BaseException as sync_error:
            # 指针替换后若目录同步失败，恢复运行时可见的旧指针；仍不声称
            # 能覆盖断电、内核或存储设备故障等宿主级崩溃边界。
            rollback_pointer = output_dir / f".current-release-rollback-{release_id}.tmp"
            try:
                if previous_pointer is None:
                    if pointer_path.exists():
                        pointer_path.unlink()
                else:
                    rollback_pointer.write_bytes(previous_pointer)
                    rollback_pointer.chmod(0o600)
                    _fsync_file(rollback_pointer)
                    os.replace(rollback_pointer, pointer_path)
                pointer_switched = False
            finally:
                if rollback_pointer.exists() and not rollback_pointer.is_symlink():
                    rollback_pointer.unlink()
            raise RuntimeError("交付指针持久化失败，已恢复旧版本") from sync_error
        return pointer
    except BaseException:
        # 指针尚未切换时，新版本目录不是权威版本，可安全保留供排障；
        # 正常异常路径尽量清理，强杀时也不会影响旧指针。
        if pointer_temp.exists() and not pointer_temp.is_symlink():
            pointer_temp.unlink()
        if not pointer_switched:
            for child in release_dir.iterdir() if release_dir.exists() else ():
                if child.is_file() and not child.is_symlink():
                    child.unlink()
            if release_dir.exists():
                release_dir.rmdir()
        raise


def _font_candidates(*, bold: bool = False) -> tuple[str, ...]:
    candidates: list[str] = []
    if os.name == "nt":
        windows_root = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
        local_fonts = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "Windows" / "Fonts"
        names = ("msyhbd.ttc", "simhei.ttf", "simsunb.ttf", "Dengb.ttf") if bold else ("msyh.ttc", "simhei.ttf", "simsun.ttc", "Deng.ttf")
        candidates.extend(str(directory / name) for directory in (windows_root, local_fonts) for name in names)
    candidates.extend((
        "/usr/share/fonts/HarmonyFont/Harmony-Regular.ttf",
        "/usr/share/fonts/HarmonyOS_Sans_SC/HarmonyOS_Sans_SC_Regular.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ))
    return tuple(candidates)


def _font_path(*, bold: bool = False) -> str:
    candidates = _font_candidates(bold=bold)
    # Preserve the established Linux font selection and its measured layout;
    # Windows has no fc-match and uses the explicit system-font candidates.
    if os.name != "nt" and shutil.which("fc-match"):
        family = "HarmonyHeiTi:style=Bold" if bold else "HarmonyHeiTi"
        matched = subprocess.run(["fc-match", "-f", "%{file}", family], capture_output=True, text=True, check=False).stdout.strip()
        if matched and Path(matched).is_file():
            return matched
    for candidate in candidates:
        if Path(candidate).is_file():
            return candidate
    if os.name == "nt" and shutil.which("fc-match"):
        family = "HarmonyHeiTi:style=Bold" if bold else "HarmonyHeiTi"
        matched = subprocess.run(["fc-match", "-f", "%{file}", family], capture_output=True, text=True, check=False).stdout.strip()
        if matched and Path(matched).is_file():
            return matched
    raise RuntimeError("未找到可用于 PDF 的中文字体")


def _plain_lines(markdown: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for raw in markdown.splitlines():
        line = raw.strip()
        if not line or line.startswith("> **真实性说明："):
            continue
        if line.startswith("# "):
            rows.append(("title", line[2:].strip()))
        elif line.startswith("## "):
            rows.append(("heading", line[3:].strip()))
        elif line.startswith("- "):
            rows.append(("body", "• " + line[2:].strip()))
        else:
            rows.append(("body", re.sub(r"\*\*(.*?)\*\*", r"\1", line)))
    return rows


def _wrap_pdf(text: str, font_name: str, size: float, max_width: float) -> list[str]:
    from reportlab.pdfbase.pdfmetrics import stringWidth

    lines: list[str] = []
    current = ""
    for char in text:
        candidate = current + char
        if current and stringWidth(candidate, font_name, size) > max_width:
            lines.append(current)
            current = char
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]


def _trim_dark_side_canvas(image: Image.Image) -> Image.Image:
    """保守裁掉证件照右侧连续的近黑附加画布，不改变人物主体比例。"""
    oriented = ImageOps.exif_transpose(image)
    if oriented.mode in {"RGBA", "LA"} or "transparency" in oriented.info:
        rgba = oriented.convert("RGBA")
        white = Image.new("RGBA", rgba.size, "white")
        white.alpha_composite(rgba)
        rgb = white.convert("RGB")
    else:
        rgb = oriented.convert("RGB")
    width, height = rgb.size
    pixels = rgb.load()
    cut = width
    # 最多裁掉右侧 30%，避免把大面积深色背景、黑发或黑衣误认为附加画布。
    for x in range(width - 1, max(int(width * 0.70), 0), -1):
        samples = [pixels[x, y] for y in range(0, height, max(1, height // 80))]
        luminances = [(299 * r + 587 * g + 114 * b) / 1000 for r, g, b in samples]
        dark_ratio = sum(level < 28 for level in luminances) / max(len(luminances), 1)
        mean_luminance = sum(luminances) / max(len(luminances), 1)
        # 只容忍近黑 JPEG 噪声和极少亮点；暗灰区域宁可保留，避免误裁主体。
        if dark_ratio >= 0.94 and mean_luminance < 24:
            cut = x
        else:
            break
    return rgb.crop((0, 0, cut, height)) if cut < width else rgb


def export_single_page_pdf(
    markdown: str,
    output_path: Path,
    photo_path: Path | None = None,
) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas

    output_path.parent.mkdir(parents=True, exist_ok=True)
    font_name = "ResumeAgentCJK"
    bold_name = "ResumeAgentCJKBold"
    if font_name not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(font_name, _font_path()))
    bold_path = _font_path(bold=True)
    if bold_name not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(bold_name, bold_path))
    width, height = A4
    margin_x, margin_y = 45, 34
    rows = _plain_lines(markdown)
    photo_box = (width - margin_x - 64, height - margin_y - 82, 64, 82) if photo_path else None

    selected: tuple[float, list[tuple[str, str, float, float, str]]] | None = None
    # 不能靠极小字号硬塞一页。10.5pt 放不下时应精简低价值内容后重试。
    for body_size in (10.8, 10.5):
        layout: list[tuple[str, str, float, float, str]] = []
        total = 0.0
        for kind, text in rows:
            size = body_size + (7.2 if kind == "title" else 1.8 if kind == "heading" else 0)
            leading = size * (1.28 if kind == "body" else 1.24)
            usable_width = width - 2 * margin_x
            # 依据当前实际垂直占用判断是否处于照片区，不依赖固定逻辑行数。
            if photo_box and total < photo_box[3] + 8:
                usable_width -= photo_box[2] + 14
            row_font = bold_name if kind in {"title", "heading"} else font_name
            wrapped = _wrap_pdf(text, row_font, size, usable_width)
            if kind == "heading":
                total += body_size * 0.34
            for line in wrapped:
                layout.append((kind, line, size, leading, row_font))
                total += leading
        if total <= height - 2 * margin_y:
            selected = (body_size, layout)
            break
    if selected is None:
        raise ValueError("内容无法以不低于10.5pt排入单页 A4；请优先精简低相关内容")

    pdf = canvas.Canvas(str(output_path), pagesize=A4)
    pdf.setTitle("Resume Agent Delivery")
    pdf.setFillColor("#FFFFFF")
    pdf.rect(0, 0, width, height, fill=1, stroke=0)
    y = height - margin_y
    if photo_path:
        source = Path(photo_path)
        if not source.is_file():
            raise FileNotFoundError(f"找不到照片：{source}")
        with Image.open(source) as original:
            cleaned = _trim_dark_side_canvas(original)
            temporary_photo_path: str | None = None
            try:
                # Close the temporary file before ReportLab reopens it. Windows
                # otherwise keeps the NamedTemporaryFile handle locked.
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temporary_photo:
                    temporary_photo_path = temporary_photo.name
                cleaned.save(temporary_photo_path)
                x, py, box_w, box_h = photo_box
                iw, ih = cleaned.size
                scale = min(box_w / iw, box_h / ih)
                draw_w, draw_h = iw * scale, ih * scale
                pdf.drawImage(
                    temporary_photo_path,
                    x + (box_w - draw_w) / 2,
                    py + (box_h - draw_h) / 2,
                    width=draw_w,
                    height=draw_h,
                    preserveAspectRatio=True,
                    mask="auto",
                )
            finally:
                if temporary_photo_path:
                    Path(temporary_photo_path).unlink(missing_ok=True)
    for kind, text, size, leading, row_font in selected[1]:
        if kind == "heading":
            y -= selected[0] * 0.34
        pdf.setFont(row_font, size)
        pdf.setFillColor("#151515" if kind in {"title", "heading"} else "#252525")
        pdf.drawString(margin_x, y - size, text)
        if kind == "heading":
            rule_y = y - size - 2.5
            pdf.setStrokeColor("#B5B5B5")
            pdf.setLineWidth(0.45)
            pdf.line(margin_x, rule_y, width - margin_x, rule_y)
        y -= leading
    pdf.showPage()
    pdf.save()


def render_pdf_preview(pdf_path: Path, png_path: Path) -> None:
    import fitz

    document = fitz.open(pdf_path)
    try:
        if document.page_count != 1:
            raise ValueError(f"PDF 不是单页：{document.page_count}页")
        pixmap = document[0].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        pixmap.save(str(png_path))
    finally:
        document.close()


def _png_matches_pdf_preview(pdf_path: Path, png_path: Path) -> bool:
    """重新渲染 PDF 并逐像素核对，避免 PNG 被错配或替换。"""
    with tempfile.NamedTemporaryFile(suffix=".png") as expected_file:
        render_pdf_preview(pdf_path, Path(expected_file.name))
        with Image.open(expected_file.name) as expected, Image.open(png_path) as actual:
            return expected.convert("RGB").tobytes() == actual.convert("RGB").tobytes()


def _normalize_content(value: str) -> str:
    value = re.sub(r"(?m)^\s*>\s*\*\*真实性说明：.*$", "", value)
    value = re.sub(r"(?m)^\s*#{1,6}\s*", "", value)
    value = re.sub(r"\*\*(.*?)\*\*", r"\1", value)
    value = value.replace("•", "").replace("- ", "")
    value = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", value)
    value = re.sub(r"[\t\r\n]+", " ", value)
    return re.sub(r" {2,}", " ", value).strip()


def _compact_pdf_wrapping(value: str) -> str:
    """比较字符完整性时忽略 PDF 自动换行造成的空白。"""
    return re.sub(r"\s+", "", value)


def qa_delivery(markdown: str, pdf_path: Path, png_path: Path, ats_path: Path | None = None) -> dict[str, Any]:
    import fitz

    findings: list[str] = []
    document = fitz.open(pdf_path)
    try:
        page_count = document.page_count
        extracted = "\n".join(page.get_text() for page in document)
        blocks = document[0].get_text("blocks") if page_count else []
        words = document[0].get_text("dict") if page_count else {"blocks": []}
        page_rect = document[0].rect if page_count else None
    finally:
        document.close()
    if page_count != 1:
        findings.append(f"PDF页数异常：{page_count}")
    expected = _normalize_content(markdown)
    actual_raw = _normalize_content(extracted)
    actual = _compact_pdf_wrapping(actual_raw)
    if _compact_pdf_wrapping(expected) != actual:
        findings.append("PDF提取文本与最终简历不一致")
    # 字符完整性比较会忽略自动换行，但 ASCII 词组内部空格具有语义，需单独保留。
    ascii_phrases = re.findall(r"(?<![A-Za-z0-9])(?:[A-Za-z0-9_.+/#-]+\s+){1,4}[A-Za-z0-9_.+/#-]+", expected)
    if any(phrase not in actual_raw for phrase in ascii_phrases):
        findings.append("PDF丢失英文或数字词组中的必要空格")
    ats_bytes = ats_path.read_bytes() if ats_path else markdown.encode("utf-8")
    if ats_bytes != markdown.encode("utf-8"):
        findings.append("ATS Markdown 与最终简历不一致")
    # 自动换行可能让标点单独落行，但字符仍完整；交付门禁关注内容缺失、
    # 越界和字号，不把 PDF 提取器的换行位置误报为内容错误。
    if re.search(r"(?m)^\s*#{1,6}\s+", extracted) or "**" in extracted:
        findings.append("PDF存在 Markdown 标记残留")
    font_sizes = [
        span.get("size", 0)
        for block in words.get("blocks", []) if "lines" in block
        for line in block.get("lines", [])
        for span in line.get("spans", []) if span.get("text", "").strip()
    ]
    if font_sizes and min(font_sizes) < 10.45:
        findings.append(f"PDF正文字号过小：最小{min(font_sizes):.2f}pt")
    if page_rect:
        for block in blocks:
            x0, y0, x1, y1 = block[:4]
            if x0 < -0.5 or y0 < -0.5 or x1 > page_rect.width + 0.5 or y1 > page_rect.height + 0.5:
                findings.append("PDF存在越界文本块")
                break
    with Image.open(png_path) as image:
        png_size = list(image.size)
        if image.width < 1000 or image.height < 1400:
            findings.append("PNG预览分辨率过低")
    if not _png_matches_pdf_preview(pdf_path, png_path):
        findings.append("PNG预览与PDF内容不一致")
    return {
        "status": "pass" if not findings else "fail",
        "page_count": page_count,
        "text_chars": len(extracted),
        "png_size": png_size,
        "findings": findings,
        "sha256": {
            "source": hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
            "ats": hashlib.sha256(ats_bytes).hexdigest(),
            "pdf_text": hashlib.sha256(actual.encode("utf-8")).hexdigest(),
            "pdf": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
            "png": hashlib.sha256(png_path.read_bytes()).hexdigest(),
        },
    }


def export_delivery_package(
    final_resume: str,
    output_dir: Path,
    photo_path: Path | None = None,
) -> dict[str, Any]:
    if output_dir.is_symlink():
        raise ValueError("输出目录不能是符号链接")
    ats_path = output_dir / "final-resume-ats.md"
    pdf_path = output_dir / "final-resume.pdf"
    png_path = output_dir / "final-resume.png"
    # 先在同一文件系统的 staging 目录完成渲染与 QA，全部通过后再原子发布，
    # 避免失败运行留下 ATS/PDF/PNG 混合版本。
    with tempfile.TemporaryDirectory(prefix=".resume-delivery-", dir=output_dir) as staging:
        staging_dir = Path(staging)
        staged_ats = staging_dir / ats_path.name
        staged_pdf = staging_dir / pdf_path.name
        staged_png = staging_dir / png_path.name
        # Preserve the exact UTF-8 bytes across platforms; Windows text mode
        # would translate LF to CRLF and fail the byte-level ATS integrity gate.
        staged_ats.write_bytes(final_resume.encode("utf-8"))
        export_single_page_pdf(final_resume, staged_pdf, photo_path=photo_path)
        render_pdf_preview(staged_pdf, staged_png)
        qa = qa_delivery(final_resume, staged_pdf, staged_png, staged_ats)
        if qa["status"] != "pass":
            raise ValueError("交付 QA 失败：" + "、".join(qa["findings"]))
        staged_targets = (
            (staged_ats, ats_path), (staged_pdf, pdf_path), (staged_png, png_path)
        )
        backups: list[tuple[Path, Path | None]] = []
        try:
            for staged, target in staged_targets:
                staged.chmod(0o600)
                backup = staging_dir / f"old-{target.name}" if target.exists() else None
                if backup:
                    os.replace(target, backup)
                backups.append((target, backup))
                os.replace(staged, target)
            # 根目录三件套用于向后兼容；权威交付由不可变版本目录和单文件指针标识。
            # 指针切换失败时，下面的统一异常分支会恢复整套根目录旧版本。
            qa["release"] = _publish_release_snapshot(output_dir, output_dir)
        except BaseException as publish_error:
            # 多文件无法由文件系统提供单一事务；发布中途失败时回滚到整套旧版本。
            rollback_errors: list[OSError] = []
            for target, backup in reversed(backups):
                try:
                    if target.exists():
                        target.unlink()
                    if backup and backup.exists():
                        os.replace(backup, target)
                except OSError as rollback_error:
                    rollback_errors.append(rollback_error)
            if rollback_errors:
                raise RuntimeError("交付发布失败，且旧版本回滚不完整") from publish_error
            raise
    return qa
