"""ATS 兼容性评分（学自 Reactive-Resume 的 PDF_ATS_RULE_CATALOG_V1）。

三级严重度：
- blocker：硬封顶（如无文本层最高 10 分）
- warning：按类扣分（5/10/15）
- tip：不计分，纯建议

0-100 分制。
"""
from __future__ import annotations

import re

# 规则表：(严重度, 类别, 封顶/扣分)
RULES: dict[str, tuple[str, str, int]] = {
    # -- parseability：词能不能被还原 --
    "NO_TEXT_LAYER": ("blocker", "parseability", 10),
    "GARBLED_TEXT": ("blocker", "parseability", 25),
    "REPLACEMENT_CHARACTERS": ("warning", "parseability", 15),
    # -- layout：版式是否保序 --
    "MULTI_COLUMN_LAYOUT": ("blocker", "layout", 55),  # 我们单列，默认不触发
    "SMALL_BODY_TEXT": ("warning", "layout", 10),
    # -- sections：章节是否可识别 --
    "NO_EXPERIENCE_SECTION": ("blocker", "sections", 60),
    "NO_RECOGNIZED_HEADINGS": ("warning", "sections", 15),
    "FEW_SECTION_HEADINGS": ("warning", "sections", 10),
    "NO_EDUCATION_SECTION": ("warning", "sections", 10),
    "NO_SKILLS_SECTION": ("warning", "sections", 10),
    "NO_BULLET_POINTS": ("warning", "sections", 5),
    # -- contact：能联系上吗 --
    "NO_EMAIL": ("blocker", "contact", 50),
    "NO_PHONE": ("warning", "contact", 15),
    "NO_NAME_LINE": ("warning", "contact", 15),
    # -- dates：时间线可重建吗 --
    "NO_DATES_FOUND": ("blocker", "dates", 60),
    "FEW_DATES": ("warning", "dates", 15),
    "MIXED_DATE_FORMATS": ("warning", "dates", 5),
    # -- content：纯建议，不计分 --
    "NO_QUANTIFIED_IMPACT": ("tip", "content", 0),
    "WEAK_ACTION_VERBS": ("tip", "content", 0),
    "FIRST_PERSON_PRONOUNS": ("tip", "content", 0),
    "LONG_BULLETS": ("tip", "content", 0),
}

STANDARD_SECTIONS = {
    "个人信息", "个人简介", "求职意向",
    "教育背景", "教育经历",
    "工作经历", "实习经历", "项目经历", "项目经验",
    "技能", "专业技能", "技能证书",
    "自我评价", "荣誉", "获奖",
}
EXPERIENCE_SECTIONS = {"工作经历", "实习经历", "项目经历", "项目经验"}

PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
DATE_RE = re.compile(r"\d{4}[./-]\d{1,2}")
WEAK_VERBS = ["负责", "参与", "协助", "配合", "熟悉", "了解"]
FIRST_PERSON = ["我", "本人", "笔者"]


def _fired_rules(resume_text: str, pdf_bytes: bytes | None = None) -> list[tuple[str, str]]:
    """返回触发的规则 [(code, detail)]。"""
    fired: list[tuple[str, str]] = []
    text = resume_text

    # --- PDF 层（真实解析） ---
    if pdf_bytes is not None:
        try:
            import pymupdf
            doc = pymupdf.open(stream=pdf_bytes)
            extracted = "".join(page.get_text() for page in doc)
            doc.close()
            if not extracted.strip():
                fired.append(("NO_TEXT_LAYER", "PDF 无可提取文本层"))
            elif "�" in extracted:
                fired.append(("REPLACEMENT_CHARACTERS", "PDF 提取出现替换字符，可能乱码"))
            elif "GARBLED_TEXT" not in [f[0] for f in fired]:
                # 中文字符丢失检查：原文中文在 PDF 提取中大量丢失
                src_zh = len(re.findall(r"[\u4e00-\u9fff]", text))
                ext_zh = len(re.findall(r"[\u4e00-\u9fff]", extracted))
                if src_zh > 20 and ext_zh < src_zh * 0.5:
                    fired.append(("GARBLED_TEXT", f"中文丢失严重（原文{src_zh}字，PDF仅{ext_zh}字）"))
        except Exception as e:
            fired.append(("NO_TEXT_LAYER", f"PDF 解析失败：{e}"))

    # --- 章节 ---
    found = {s for s in STANDARD_SECTIONS if s in text}
    if not (EXPERIENCE_SECTIONS & found):
        fired.append(("NO_EXPERIENCE_SECTION", "未识别到工作/实习/项目经历章节"))
    if len(found) == 0:
        fired.append(("NO_RECOGNIZED_HEADINGS", "无标准章节标题"))
    elif len(found) < 3:
        fired.append(("FEW_SECTION_HEADINGS", f"仅识别到 {len(found)} 个标准章节"))
    if not ({"教育背景", "教育经历"} & found):
        fired.append(("NO_EDUCATION_SECTION", "无教育背景章节"))
    if not ({"技能", "专业技能", "技能证书"} & found):
        fired.append(("NO_SKILLS_SECTION", "无技能章节"))
    if not re.search(r"^[-*•]\s", text, re.M):
        fired.append(("NO_BULLET_POINTS", "无项目符号，经历描述建议用 bullet"))

    # --- 联系 ---
    if not EMAIL_RE.search(text):
        fired.append(("NO_EMAIL", "未识别到邮箱"))
    if not PHONE_RE.search(text):
        fired.append(("NO_PHONE", "未识别到手机号"))
    first_line = (text.splitlines() or [""])[0].strip()
    if len(first_line) > 20 or not first_line:
        fired.append(("NO_NAME_LINE", "首行疑似非姓名"))

    # --- 日期 ---
    dates = DATE_RE.findall(text)
    if not dates:
        fired.append(("NO_DATES_FOUND", "未识别到日期，时间线无法重建"))
    elif len(dates) < 2:
        fired.append(("FEW_DATES", "日期过少"))
    seps = {d[4] for d in dates if len(d) > 5}
    if len(seps) > 1:
        fired.append(("MIXED_DATE_FORMATS", "日期分隔符不统一"))

    # --- 内容（tip，不计分） ---
    if not re.search(r"\d+\s*[%％]|提升|增长|降低", text):
        fired.append(("NO_QUANTIFIED_IMPACT", "无量化成果，建议补充数字"))
    weak_count = sum(text.count(w) for w in WEAK_VERBS)
    if weak_count >= 5:
        fired.append(("WEAK_ACTION_VERBS", f"弱动词出现 {weak_count} 次，建议换强动词"))
    if any(p in text for p in FIRST_PERSON):
        fired.append(("FIRST_PERSON_PRONOUNS", "出现第一人称，简历建议省略主语"))
    long_bullets = [l for l in text.splitlines() if re.match(r"^[-*•]\s", l) and len(l) > 80]
    if long_bullets:
        fired.append(("LONG_BULLETS", f"{len(long_bullets)} 条 bullet 超 80 字，建议拆分"))

    return fired


def score_ats(resume_text: str, pdf_bytes: bytes | None = None) -> dict:
    """0-100 分 ATS 评分。返回 {score, fired, tips}。"""
    fired = _fired_rules(resume_text, pdf_bytes)
    score = 100
    cap = 100
    details: list[dict] = []
    tips: list[dict] = []

    for code, detail in fired:
        severity, category, value = RULES[code]
        if severity == "blocker":
            cap = min(cap, value)
            details.append({"code": code, "severity": "blocker",
                            "category": category, "detail": detail, "cap": value})
        elif severity == "warning":
            score -= value
            details.append({"code": code, "severity": "warning",
                            "category": category, "detail": detail, "deduction": value})
        else:
            tips.append({"code": code, "detail": detail})

    score = max(0, min(score, cap))
    return {
        "score": score,
        "fired": details,
        "tips": tips,
        "summary": f"ATS 评分 {score}/100"
                   + (f"（被 {len([d for d in details if d['severity']=='blocker'])} 个 blocker 封顶）" if cap < 100 else ""),
    }


# 向后兼容：旧 check_ats 接口
def check_ats(resume_text: str, jd_text: str = "", match_coverage: dict | None = None) -> list[dict]:
    result = score_ats(resume_text)
    out = [{"check": f"ATS 评分 {result['score']}/100",
            "status": "pass" if result["score"] >= 80 else "warn" if result["score"] >= 50 else "fail",
            "detail": result["summary"]}]
    for d in result["fired"]:
        out.append({"check": d["code"], "status": "fail" if d["severity"] == "blocker" else "warn",
                    "detail": d["detail"]})
    for t in result["tips"]:
        out.append({"check": t["code"], "status": "warn", "detail": t["detail"] + "（建议）"})
    return out
