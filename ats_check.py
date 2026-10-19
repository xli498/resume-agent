"""ATS 兼容性检查清单（学自 ResumeCraftr）。

ResumeCraftr 的"ATS 优化"本质 = 干净可解析的版式 + JD 关键词覆盖。
这里做成规则检查，在分析阶段给出可操作的改进建议。
"""
from __future__ import annotations

import re

# ATS 可识别的标准章节标题
STANDARD_SECTIONS = {
    "个人信息", "个人简介", "求职意向",
    "教育背景", "教育经历",
    "工作经历", "实习经历", "项目经历", "项目经验",
    "技能", "专业技能", "技能证书",
    "自我评价", "荣誉", "获奖",
}

PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def check_ats(resume_text: str, jd_text: str = "", match_coverage: dict | None = None) -> list[dict]:
    """ATS 检查，返回 [{check, status, detail}]。

    status: pass / warn / fail
    """
    results: list[dict] = []
    lines = resume_text.splitlines()
    text = resume_text

    # 1. 标准章节标题
    found_sections = {s for s in STANDARD_SECTIONS if s in text}
    results.append({
        "check": "标准章节标题",
        "status": "pass" if len(found_sections) >= 3 else "warn",
        "detail": f"识别到 {len(found_sections)} 个标准章节：{'、'.join(sorted(found_sections)[:5]) or '无'}"
                  + ("；建议使用标准标题（教育背景/工作经历/项目经历/技能）" if len(found_sections) < 3 else ""),
    })

    # 2. 联系方式
    has_phone = bool(PHONE_RE.search(text))
    has_email = bool(EMAIL_RE.search(text))
    if has_phone and has_email:
        results.append({"check": "联系方式", "status": "pass", "detail": "电话+邮箱齐全"})
    elif has_phone or has_email:
        results.append({"check": "联系方式", "status": "warn", "detail": "只有电话或只有邮箱，建议两者都放"})
    else:
        results.append({"check": "联系方式", "status": "fail", "detail": "未识别到电话和邮箱，ATS 无法建立联系"})

    # 3. JD 关键词覆盖率（用已有的 match_coverage）
    if match_coverage and match_coverage.get("total"):
        cov = match_coverage["weighted_coverage"]
        counts = match_coverage["counts"]
        if cov >= 60:
            status = "pass"
        elif cov >= 30:
            status = "warn"
        else:
            status = "fail"
        results.append({
            "check": "JD 关键词覆盖",
            "status": status,
            "detail": f"加权覆盖率 {cov}%（强匹配 {counts['strong']} / 待补 {counts['weak']} / 缺口 {counts['gap']}）",
        })
    elif jd_text.strip():
        results.append({"check": "JD 关键词覆盖", "status": "warn", "detail": "JD 未解析出有效要求，无法评估"})

    # 4. 长度（单页建议）
    non_empty = [line for line in lines if line.strip()]
    if len(non_empty) <= 45:
        results.append({"check": "篇幅", "status": "pass", "detail": f"约 {len(non_empty)} 行，单页可容纳"})
    elif len(non_empty) <= 70:
        results.append({"check": "篇幅", "status": "warn", "detail": f"约 {len(non_empty)} 行，接近单页上限，注意精简"})
    else:
        results.append({"check": "篇幅", "status": "warn", "detail": f"约 {len(non_empty)} 行，超出一页，建议删减次要经历"})

    # 5. 反模式：特殊符号/表情（ATS 解析乱码风险）
    weird = re.findall(r"[�\U0001F300-\U0001FAFF★☆◆■●▲►❖✦✧]", text)
    if weird:
        results.append({"check": "特殊符号", "status": "warn",
                        "detail": f"发现 {len(set(weird))} 种特殊符号/表情，ATS 可能解析乱码，建议用纯文本"})
    else:
        results.append({"check": "特殊符号", "status": "pass", "detail": "无特殊符号，纯文本解析友好"})

    # 6. 日期格式一致性
    dates = re.findall(r"\d{4}[./-]\d{1,2}", text)
    fmts = {d[4] for d in dates if len(d) > 5}
    if len(fmts) <= 1:
        results.append({"check": "日期格式", "status": "pass", "detail": "日期格式一致"})
    else:
        results.append({"check": "日期格式", "status": "warn", "detail": "日期分隔符不统一，建议统一为 2023.09 格式"})

    return results
