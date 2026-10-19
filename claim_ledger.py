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
