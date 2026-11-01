#!/usr/bin/env python3
"""本地补齐 /api/pdf 所需字体（仅本机环境，不影响线上服务）。

背景：
- pdf_export.py 的字体目录硬编码为 /opt/demo/resume/fonts/（只读文件，禁改），
  需要 NotoSansSC-Regular.otf 与 NotoSansSC-Bold.otf；缺文件时 render_pdf
  的 font_buffers 为空，insert_text 抛错，/api/pdf 返回 500。
- 本机根本没有 /opt/demo/ 目录，线上服务在腾讯云服务器上，不在本机；
  因此这里新建 /opt/demo/resume/fonts/ 属于本地环境补齐，不影响线上。
- /opt/demo 不随仓库走、VM 换机后会丢：换机后重跑本脚本即可恢复。

做法：从系统 Noto Sans CJK TTC 中提取 SC 面（不是 JP/KR/TC/HK 面，
也不是 Mono 面），存为 .otf 扩展名；render_pdf 走 buffer，不认扩展名，
内容是标准 sfnt 字体数据即可。

选本脚本而不用 README 贴命令的理由：SC 面的选择逻辑（按 name 表精确
匹配 "Noto Sans CJK SC"，排除 Mono）用 shell 命令难以可靠表达，脚本
自带选择＋校验，换机后一键重跑。

用法：
    python3 scripts/setup_local_fonts.py            # 需写 /opt/demo 的权限（本机以 root 跑）
    python3 scripts/setup_local_fonts.py --force    # 覆盖已有文件
    python3 scripts/setup_local_fonts.py --target /tmp/fonts   # 写到别处（演练用）

依赖：fontTools。系统 python3 若缺失，本脚本自动装到 ~/workspace/pylibs
（home 持久化，换机不丢）后重试。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_TARGET = Path("/opt/demo/resume/fonts")
DEFAULT_REGULAR_TTC = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
DEFAULT_BOLD_TTC = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
SC_FAMILY = "Noto Sans CJK SC"


def _ensure_fonttools() -> None:
    """保证 fontTools 可 import；缺失时装到 home 下的 pylibs（换机不丢）。"""
    try:
        import fontTools  # noqa: F401
        return
    except ImportError:
        pass
    pylibs = Path.home() / "workspace" / "pylibs"
    sys.path.insert(0, str(pylibs))
    try:
        import fontTools  # noqa: F401
        return
    except ImportError:
        pass
    print("fontTools 缺失，正在安装到 ~/workspace/pylibs ...", file=sys.stderr)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--target", str(pylibs), "fonttools"]
    )
    sys.path.insert(0, str(pylibs))
    import fontTools  # noqa: F401


def _select_sc_face(ttc_path: Path, subfamily: str):
    """从 TTC 中精确选出 SC 面：family 必须等于 SC，排除 Mono/JP/KR/TC/HK。"""
    from fontTools.ttLib import TTCollection

    coll = TTCollection(str(ttc_path))
    matches = []
    for font in coll.fonts:
        name = font["name"]
        if name.getDebugName(1) == SC_FAMILY and name.getDebugName(2) == subfamily:
            matches.append(font)
    if len(matches) != 1:
        raise SystemExit(
            f"{ttc_path}：期望恰好 1 个 {SC_FAMILY}/{subfamily} 面，实际 {len(matches)} 个"
        )
    return matches[0]


def _verify_saved(path: Path, subfamily: str) -> tuple[str, str, int]:
    """重读输出文件，核对面正确且大小合理（SC 面约十几 MB，过小必为失败）。"""
    from fontTools.ttLib import TTFont

    font = TTFont(str(path))
    family = font["name"].getDebugName(1)
    sub = font["name"].getDebugName(2)
    size = path.stat().st_size
    assert family == SC_FAMILY, f"{path} 面错误：{family}"
    assert sub == subfamily, f"{path} 子族错误：{sub}"
    assert size > 1_000_000, f"{path} 过小（{size} 字节），可能提取失败"
    return family, sub, size


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="本地补齐 /api/pdf 所需字体（见文件头说明）。")
    ap.add_argument("--target", default=str(DEFAULT_TARGET), help="输出目录，默认 /opt/demo/resume/fonts")
    ap.add_argument("--regular-ttc", default=str(DEFAULT_REGULAR_TTC))
    ap.add_argument("--bold-ttc", default=str(DEFAULT_BOLD_TTC))
    ap.add_argument("--force", action="store_true", help="覆盖已有文件")
    args = ap.parse_args(argv)
    _ensure_fonttools()
    target = Path(args.target)
    try:
        target.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print(f"无权写 {target}：请以 root 跑本脚本（本机以 root 跑，无需 sudo）。", file=sys.stderr)
        return 2
    jobs = [
        ("Regular", Path(args.regular_ttc), "NotoSansSC-Regular.otf"),
        ("Bold", Path(args.bold_ttc), "NotoSansSC-Bold.otf"),
    ]
    for subfamily, ttc, out_name in jobs:
        if not ttc.is_file():
            print(f"找不到 TTC：{ttc}，请先确认系统 Noto CJK 字体包仍在。", file=sys.stderr)
            return 2
        out = target / out_name
        if out.exists() and not args.force:
            print(f"已存在，跳过：{out}（加 --force 覆盖）")
            continue
        face = _select_sc_face(ttc, subfamily)
        face.save(str(out))
        family, sub, size = _verify_saved(out, subfamily)
        print(f"已写入：{out}（{family} {sub}，{size} 字节）")
    print("完成。/api/pdf 渲染依赖此目录下的两个文件。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
