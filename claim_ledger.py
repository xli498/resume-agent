"""Claim Ledger 原子事实层 + 语义角色门禁 + 面试防御。

学自 sushen-resume-maker，适配本项目规则引擎（无 LLM 依赖）：
- 原子 Claim：把行级事实拆成"动作/角色/数字/结果"原子单元，每个输出 bullet 可映射回 Claim
- 角色门禁：角色等级只允许降级，不允许升级（"参与"写成"主导"直接拦截）
- 面试防御：每个强表述自动生成面试追问 + 安全回答边界
"""
from __future__ import annotations

import re

# 角色等级：改写时只允许降级，不允许升级
ROLE_LEVELS: dict[str, int] = {
    # L5：最高决策 / ownership / 首创
    "主导": 5, "牵头": 5, "负责人": 5, "总负责": 5, "创始人": 5,
    # L4：职责 ownership
    "负责": 4, "管理": 4, "独立完成": 4, "独立负责": 4, "带领": 4,
    # L3：深度参与
    "参与": 3, "协同": 3, "推动": 3, "深度参与": 3,
    # L2：辅助
    "协助": 2, "配合": 2, "辅助": 2, "支持": 2,
    # L1：了解
    "熟悉": 1, "了解": 1, "学习": 1, "接触": 1,
}

# 强表述：出现即高被追问风险，需证据支撑
STRONG_WORDS = ["0到1", "0→1", "首个", "首次", "第一", "从无到有", "独立开发"]

NUMBER_RE = re.compile(r"(?<!\d)\d+(?:\.\d+)?(?:%|万|亿|年|个月|天|家|人|个|次|倍)?")


def role_level_of(text: str) -> tuple[int, list[str]]:
    """返回文本中出现的最高角色等级及命中的角色词。"""
    found = [word for word in ROLE_LEVELS if word in text]
    level = max((ROLE_LEVELS[word] for word in found), default=0)
    return level, found


def extract_atomic_claims(resume_text: str) -> list[dict]:
    """把简历拆成原子 Claim：每个 Claim = 一个可验证的事实单元。

    返回 [{claim_id, fact_id, text, numbers, role_level, role_words, strong_words}]
    """
    from main import build_fact_ledger  # 延迟导入，避免循环依赖

    claims: list[dict] = []
    for fact in build_fact_ledger(resume_text):
        clauses = [c.strip() for c in re.split(r"[。；;\n]+", fact["text"]) if c.strip()]
        for clause in clauses:
            numbers = NUMBER_RE.findall(clause)
            level, role_words = role_level_of(clause)
            strong = [w for w in STRONG_WORDS if w in clause]
            claims.append({
                "claim_id": f"C{len(claims) + 1:03d}",
                "fact_id": fact["fact_id"],
                "section": fact["section"],
                "text": clause,
                "numbers": numbers,
                "role_level": level,
                "role_words": role_words,
                "strong_words": strong,
            })
    return claims


def check_role_downgrade(content: str, claims: list[dict]) -> list[str]:
    """角色门禁：改写内容的角色等级不得高于证据 Claim 的最高等级。

    返回违规描述列表；空列表表示通过。
    """
    content_level, content_words = role_level_of(content)
    if content_level == 0:
        return []
    evidence_level = max((c["role_level"] for c in claims), default=0)
    if content_level > evidence_level:
        return [
            f"角色升级：原文最高角色{evidence_level}级，"
            f"改写用了「{'/'.join(content_words)}」({content_level}级)，只允许降级不允许升级"
        ]
    return []


def _is_year(number: str) -> bool:
    """年份（2023/2025）不算业务指标，不追问口径。"""
    return bool(re.fullmatch(r"(19|20)\d{2}", number))


# ---------------------------------------------------------------------------
# Claim 校验器（学自 validate_claim_ledger.py）
# ---------------------------------------------------------------------------

# 提升类词：出现但无数字支撑时标记 missing_baseline 风险
IMPROVE_WORDS = ["提升", "增长", "降低", "减少", "优化", "提高", "翻倍"]


def validate_claims(claims: list[dict], resume_text: str) -> list[str]:
    """校验 Claim Ledger：ID 唯一、引用有效、强角色词-等级一致、数字可追溯。

    返回错误列表；空列表表示通过。
    """
    errors: list[str] = []
    seen_ids: set[str] = set()
    source = resume_text.casefold()
    for claim in claims:
        cid = claim.get("claim_id", "")
        if not cid:
            errors.append("存在缺失 claim_id 的 Claim")
            continue
        if cid in seen_ids:
            errors.append(f"重复 claim_id：{cid}")
        seen_ids.add(cid)
        # 强角色词必须对应高等级（他们的 role_scope 检查）
        strong_in_text = [w for w in STRONG_WORDS if w in claim.get("text", "")]
        if strong_in_text and claim.get("role_level", 0) < 4:
            errors.append(
                f"{cid} 含强角色词「{'/'.join(strong_in_text)}」"
                f"但角色等级仅 {claim.get('role_level')} 级，需人工确认"
            )
        # 数字必须在原文可追溯（防幻觉）
        for number in claim.get("numbers", []):
            if number not in resume_text:
                errors.append(f"{cid} 的数字「{number}」在原文中找不到")
    return errors


def claim_risk_flags(claim: dict) -> list[str]:
    """风险标记（学自 RISK_FLAGS）：缺基线、缺口径等。"""
    flags: list[str] = []
    text = claim.get("text", "")
    if any(w in text for w in IMPROVE_WORDS) and not claim.get("numbers"):
        flags.append("missing_baseline")  # 有提升无数字：缺基线
    if claim.get("numbers") and "时间" not in text and not _is_year(claim["numbers"][0]):
        flags.append("missing_time_window")  # 有数字无时间窗
    if claim.get("role_level", 0) >= 4:
        flags.append("strong_role_term")  # 强角色词
    return flags


# ---------------------------------------------------------------------------
# 拷打深挖（学自 question-tree.md）
# ---------------------------------------------------------------------------

# 深挖问题模板：按 question-tree 三轮优先级
# 每类：(维度, 触发条件, 问题模板)
DIG_TEMPLATES: list[tuple[str, str, str]] = [
    ("主动发现", "has_metric", "「{hint}」这个结果，最初是谁发现的问题？你看到了什么别人没看到的信号？"),
    ("复杂问题", "has_metric", "做「{hint}」时最难的一步是什么？难点来自人、流程、数据还是时间？"),
    ("判断取舍", "has_method", "「{hint}」为什么选这个方法？当时有哪些备选和取舍依据？"),
    ("推动影响", "high_role", "「{hint}」需要协调哪些人？对方不配合时你怎么推动的？"),
    ("方法体系", "has_method", "做完「{hint}」后，你沉淀了什么可复用的方法/SOP/模板？"),
    ("结果影响", "has_metric", "除了「{hint}」这个数字，还有什么可验证的结果（覆盖范围/效率/复用）？"),
]

METHOD_WORDS = ["通过", "利用", "采用", "基于", "搭建", "设计", "建立", "优化"]


def build_dig_questions(claims: list[dict], max_q: int = 3) -> list[dict]:
    """按 question-tree 优先级生成深挖问题：每轮 1-3 个，不重复问已有信息。

    优先级：有数字的 Claim > 高角色 Claim > 有方法的 Claim。
    """
    def priority(claim: dict) -> int:
        score = 0
        if claim.get("numbers"):
            score += 3
        if claim.get("role_level", 0) >= 3:
            score += 2
        if any(w in claim.get("text", "") for w in METHOD_WORDS):
            score += 1
        return score

    ranked = sorted(claims, key=priority, reverse=True)
    questions: list[dict] = []
    used_dims: set[str] = set()
    for claim in ranked:
        if len(questions) >= max_q:
            break
        text = claim.get("text", "")
        hint = text[:24]
        has_metric = bool(claim.get("numbers"))
        high_role = claim.get("role_level", 0) >= 3
        has_method = any(w in text for w in METHOD_WORDS)
        for dim, cond, template in DIG_TEMPLATES:
            if dim in used_dims:
                continue
            if ((cond == "has_metric" and has_metric)
                    or (cond == "high_role" and high_role)
                    or (cond == "has_method" and has_method)):
                questions.append({
                    "question_id": f"Q{len(questions) + 1:02d}",
                    "claim_id": claim["claim_id"],
                    "dimension": dim,
                    "question": template.format(hint=hint),
                })
                used_dims.add(dim)
                break
    return questions


def add_interview_claims(claims: list[dict], qa_pairs: list[dict]) -> list[dict]:
    """用户回答 → 新 Claim（origin=interview），保留原话，不编造。

    qa_pairs: [{question_id, claim_id, answer}]
    """
    new_claims = list(claims)
    for qa in qa_pairs:
        answer = (qa.get("answer") or "").strip()
        if not answer or len(answer) < 5:
            continue  # 太短的回答跳过，不硬编
        # 去重：与已有 Claim 高度相似的跳过
        if any(answer[:20] in c.get("text", "") for c in new_claims):
            continue
        numbers = [n for n in NUMBER_RE.findall(answer) if not _is_year(n)]
        level, role_words = role_level_of(answer)
        new_claims.append({
            "claim_id": f"C{len(new_claims) + 1:03d}",
            "fact_id": qa.get("claim_id", ""),
            "section": "深挖补充",
            "text": answer[:200],
            "numbers": numbers,
            "role_level": level,
            "role_words": role_words,
            "strong_words": [w for w in STRONG_WORDS if w in answer],
            "origin": "interview",
        })
    return new_claims


# ---------------------------------------------------------------------------
# 面试防御覆盖校验（学自 validate_interview_defense.py）
# ---------------------------------------------------------------------------

def validate_defense_coverage(defense: list[dict], claims: list[dict]) -> list[str]:
    """每个高风险 Claim 必须被至少一个面试问题覆盖。

    高风险 = 有数字 / role≥4 / 有强表述词。
    """
    errors: list[str] = []
    covered: set[str] = set()
    for item in defense:
        covered.add(item.get("claim_id", ""))
    for claim in claims:
        is_risky = (
            bool(claim.get("numbers"))
            or claim.get("role_level", 0) >= 4
            or bool(claim.get("strong_words"))
        )
        if is_risky and claim["claim_id"] not in covered:
            errors.append(f"高风险 Claim {claim['claim_id']} 没有被面试防御覆盖")
    return errors


# ---------------------------------------------------------------------------
# JD Matrix 匹配类型 + 加权覆盖率（学自 validate_jd_matrix.py）
# ---------------------------------------------------------------------------

# 匹配类型：direct 直接 / transferable 可迁移 / credential_only 仅证书 / none 无
MATCH_FACTORS = {"strong": 1.0, "medium": 0.65, "weak": 0.30, "gap": 0.0}


def classify_match_type(item: dict) -> str:
    """给匹配项打匹配类型标签。"""
    level = item.get("level", "")
    evidence = item.get("evidence_ids") or []
    if level == "缺口" or not evidence:
        return "none"
    if level == "较强匹配":
        return "direct"
    # 待补证据：语义相关但非直接命中 → 可迁移
    return "transferable"


def match_coverage(matrix: list[dict]) -> dict:
    """加权覆盖率：strong=1.0, medium=0.65, weak=0.30, gap=0。"""
    level_map = {"较强匹配": "strong", "待补证据": "weak", "缺口": "gap"}
    total = 0.0
    counts: dict[str, int] = {"strong": 0, "medium": 0, "weak": 0, "gap": 0}
    for item in matrix:
        lv = level_map.get(item.get("level", ""), "gap")
        total += MATCH_FACTORS[lv]
        counts[lv] += 1
        item["match_type"] = classify_match_type(item)
    n = len(matrix) or 1
    return {
        "weighted_coverage": round(total / n * 100, 1),
        "counts": counts,
        "total": len(matrix),
    }


def build_interview_defense(claims: list[dict]) -> list[dict]:
    """面试防御：每个强表述 Claim 生成追问 + 安全回答边界。"""
    defense: list[dict] = []
    for claim in claims:
        questions: list[dict] = []
        for number in claim["numbers"]:
            if _is_year(number):
                continue
            questions.append({
                "question": f"「{number}」的统计口径是什么？",
                "followups": ["分母是什么？时间窗是多久？", "数据来源是哪里？能复现吗？"],
                "boundary": "只说自己经手的部分；团队结果不说成个人功劳",
            })
        if claim["role_level"] >= 4 and claim["role_words"]:
            word = claim["role_words"][0]
            questions.append({
                "question": f"你说的「{word}」具体指什么？",
                "followups": ["团队几个人？你负责哪一块？", "和谁协作？决策是谁拍的？"],
                "boundary": "按 Claim 原文的角色边界回答，不升格",
            })
        for strong in claim["strong_words"]:
            questions.append({
                "question": f"「{strong}」是真的吗？细讲一下过程。",
                "followups": ["当时还有谁参与？", "最大的困难是什么？怎么解决的？"],
                "boundary": '有证据说证据，没证据说尝试；不确定的标"未核实"',
            })
        if questions:
            defense.append({
                "claim_id": claim["claim_id"],
                "claim_text": claim["text"][:60],
                "questions": questions,
            })
    return defense
