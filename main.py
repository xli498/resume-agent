import argparse
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from llm_client import call_llm
from delivery import _image_font_path


# 读取 UTF-8 文本文件，并返回文件内容。
def read_text_file(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8")


# 根据标题，把简历分成几个可处理的模块。
def parse_resume(text: str) -> dict:
    sections = {
        "个人信息": [],
        "求职意向": [],
        "教育背景": [],
        "工作经历": [],
        "项目经历": [],
        "技能证书": [],
        "自我评价": [],
    }
    current_section = None

    aliases = {
        "实习经历": "工作经历", "工作经验": "工作经历", "实习经验": "工作经历",
        "项目经验": "项目经历", "校园经历": "项目经历", "社团经历": "项目经历",
        "荣誉奖项": "技能证书", "证书奖项": "技能证书", "专业技能": "技能证书",
        "联系方式": "个人信息", "基本信息": "个人信息",
        "教育": "教育背景", "education": "教育背景", "academic background": "教育背景",
        "经历": "工作经历", "experience": "工作经历", "work experience": "工作经历",
        "项目": "项目经历", "projects": "项目经历", "project experience": "项目经历",
        "技能": "技能证书", "skills": "技能证书", "certifications": "技能证书",
        "求职目标": "求职意向", "求职方向": "求职意向", "career objective": "求职意向",
        "summary": "自我评价", "个人概述": "自我评价", "profile": "自我评价",
        "自我介绍": "自我评价", "个人优势": "自我评价",
    }
    for line in text.splitlines():
        line = re.sub(r"^#{1,6}\s*", "", line.strip())
        line = re.sub(r"^\*\*(.+?)\*\*$", r"\1", line).strip()
        if not line:
            continue
        normalized = re.sub(r"^[一二三四五六七八九十]+[、.．\s]*", "", line)
        normalized = normalized.strip().rstrip("：:").casefold()
        section_key = next((key for key in sections if key.casefold() == normalized), None)
        alias_key = next((key for key in aliases if key.casefold() == normalized), None)
        if section_key or alias_key:
            current_section = aliases.get(alias_key, section_key)
            continue
        if current_section is not None:
            sections[current_section].append(line)
        else:
            sections["个人信息"].append(line)

    return sections


def build_fact_ledger(resume_text: str) -> list[dict]:
    """把原始简历拆成可追溯事实，供模型改写和最终校验引用。"""
    parsed = parse_resume(resume_text)
    facts: list[dict] = []
    for section, lines in parsed.items():
        for line in lines:
            clean = line.strip()
            if not clean:
                continue
            facts.append({
                "fact_id": f"F{len(facts) + 1:03d}",
                "section": section,
                "text": clean,
            })
    return facts


def validate_llm_revisions(resume_text: str, llm_result: dict) -> tuple[list[dict], list[str]]:
    """保守接纳改写：证据存在还不够，正文事实必须能在证据原文中逐字追溯。"""
    ledger = {item["fact_id"]: item for item in build_fact_ledger(resume_text)}
    accepted: list[dict] = []
    findings: list[str] = []
    for index, item in enumerate(llm_result.get("resume_revisions", []), 1):
        if not isinstance(item, dict):
            findings.append(f"第{index}条改写不是对象")
            continue
        content = item.get("content", "")
        if isinstance(content, list):
            content = "；".join(str(value) for value in content)
        content = str(content).strip()
        evidence_ids = item.get("evidence_ids", [])
        if not isinstance(evidence_ids, list) or not evidence_ids:
            findings.append(f"第{index}条改写缺少 evidence_ids")
            continue
        invalid_ids = [value for value in evidence_ids if value not in ledger]
        if invalid_ids:
            findings.append(f"第{index}条改写引用无效证据：{','.join(map(str, invalid_ids))}")
            continue
        evidence_text = " ".join(ledger[value]["text"] for value in evidence_ids)
        if not content:
            findings.append(f"第{index}条改写内容为空")
            continue
        numbers = re.findall(r"(?<!\d)\d+(?:\.\d+)?(?:年|个月|家|行|个|字段|科目)?", content)
        missing_numbers = [number for number in numbers if number not in evidence_text]
        if missing_numbers:
            findings.append(f"第{index}条改写包含证据中不存在的数字：{','.join(missing_numbers)}")
            continue
        assertion_words = (
            "精通", "熟练", "主导", "领导", "负责人", "负责", "推动", "优化", "提升",
            "建立", "搭建", "独立", "跨部门", "显著", "成功", "落地", "牵头", "管理",
        )
        unsupported_assertions = [word for word in assertion_words if word in content and word not in evidence_text]
        if unsupported_assertions:
            findings.append(f"第{index}条改写包含证据未支持的断言：{','.join(unsupported_assertions)}")
            continue
        normalize = lambda value: re.sub(r"[\s，。；：、,.!?！？（）()\-—_|/]+", "", value).casefold()
        normalized_content = normalize(content)
        normalized_evidence = normalize(evidence_text)
        # 自动写入采用默认拒绝策略：只允许证据原文的抽取、合并或标点整理。
        # 语义改写即使合理也进入“待人工核验”，避免 ID 存在却内容挂靠虚假事实。
        clauses = [normalize(value) for value in re.split(r"[。；;\n]+", content) if normalize(value)]
        if not normalized_content or any(clause not in normalized_evidence for clause in clauses):
            findings.append(f"第{index}条改写不是证据原文的可追溯抽取，需人工核验")
            continue
        strong_findings = validate_resume_draft(content)
        if strong_findings:
            findings.append(f"第{index}条改写触发强断言门禁：{'、'.join(strong_findings)}")
            continue
        accepted.append({
            "section": str(item.get("section", "项目经历")).strip() or "项目经历",
            "content": content,
            "evidence_ids": evidence_ids,
        })
    return accepted, findings


def prepare_output_dir(output_dir: Path) -> None:
    """清理本次运行会产出的旧文件，防止失败后被误当成最新结果。"""
    current = output_dir
    while current != current.parent:
        if current.is_symlink():
            raise ValueError("输出目录路径不能包含符号链接")
        current = current.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.chmod(0o700)
    artifacts = (
        "analysis.json", "match-report.md", "prompt.txt", "final-resume.md",
        "final-resume-with-photo.png", "llm-analysis.json", "targeted-resume-draft.md",
        "llm-alignment-report.md", "evidence-mapping-report.md", "run-manifest.json",
        "run-manifest.sha256",
        "final-resume-ats.md", "final-resume.pdf", "final-resume.png", "qa-report.json",
        "confirmation-questions.md", "revision-diff-report.md",
    )
    for name in artifacts:
        path = output_dir / name
        if path.is_file() or path.is_symlink():
            path.unlink()


def write_private_text(path: Path, content: str) -> None:
    """简历产物默认仅当前用户可读写。"""
    if path.is_symlink():
        raise ValueError("拒绝写入符号链接产物")
    # Write bytes explicitly so generated artifacts retain LF/UTF-8 identity
    # on Windows instead of receiving implicit CRLF translation.
    path.write_bytes(content.encode("utf-8"))
    path.chmod(0o600)


# 从岗位 JD 中检查关键词是否出现在简历文本里。
def match_keywords(resume_text: str, keywords: list[str]) -> dict:
    """返回每个关键词是否命中；英文关键词按词边界匹配，避免子串误报。"""
    normalized_text = resume_text.lower()
    result = {}

    for keyword in keywords:
        normalized_keyword = keyword.lower()
        if re.fullmatch(r"[a-z][a-z0-9+#./-]*", normalized_keyword):
            result[keyword] = bool(re.search(
                rf"(?<![a-z0-9]){re.escape(normalized_keyword)}(?![a-z0-9])",
                normalized_text,
            ))
        else:
            result[keyword] = normalized_keyword in normalized_text

    return result


SEMANTIC_CONCEPTS = {
    # 别名只表示弱语义关联，不等同于已证明岗位能力。
    "数据分析": {"数据处理", "数据整理", "统计分析", "报表", "可视化"},
    "客户运营": {"客户运营", "客户维护", "客户关系", "用户运营", "社群运营", "客户服务"},
    "项目管理": {"项目管理", "项目推进", "项目协调", "进度管理", "任务管理"},
    "AI Agent": {"ai agent", "智能体", "langgraph", "工作流编排", "agent"},
    "工作流建设": {"langgraph", "工作流编排", "工作流开发"},
    "风险控制": {"风控", "合规", "风险评估", "审核"},
    "机器学习": {"机器学习", "模型训练", "特征工程", "分类模型", "回归模型"},
}

JD_KNOWN_REQUIREMENTS = (
    "AI Agent", "机器学习", "特征工程", "数据分析", "数据挖掘", "项目管理",
    "客户运营", "风险控制", "用户挖掘", "工作流建设", "工作经验",
)

EDUCATION_LEVELS = {"高中": 1, "中专": 1, "大专": 2, "专科": 2, "本科": 3, "硕士": 4, "博士": 5}


def _extract_requirements_from_line(line: str) -> list[str]:
    """提取可审计要求，避免把连接词和标题碎片当成独立要求。"""
    stripped = line.strip()
    if re.match(r"^(职位名称|岗位名称|职位|岗位)\s*[：:]", stripped):
        return []
    found = [term for term in JD_KNOWN_REQUIREMENTS if term.casefold() in stripped.casefold()]
    # 保留常见经验对象，避免把“项目经验/销售经验”降级成无法解释的泛化要求。
    experience_terms = re.findall(r"(?:项目|销售|运营|研发|相关工作)经验", stripped)
    found.extend(experience_terms)
    if experience_terms:
        found.append("工作经验")
    # 否定句也必须进入矩阵，否则“不要求英语/无需销售经验”会被静默丢失。
    negative_matches = re.findall(
        r"(?:不要求|不需要|无需|不接受|不得|禁止|无须)\s*([\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9+#./-]{1,15})",
        stripped,
    )
    found.extend(negative_matches)
    for education in EDUCATION_LEVELS:
        if education in stripped:
            found.append(education)
    if re.search(r"(?:\d{2}\s*[—\-至到~～]\s*\d{2}\s*岁|年龄\s*(?:不超过|不高于|小于等于|≤|至多)?\s*\d{1,3}\s*岁?(?:及以上|以上|起|及以下|以下|以内|不超过)?)", stripped):
        found.append("年龄")
    # 英文技术、平台和证书名允许动态发现；多词概念中的单词不重复输出。
    # 先保留常见多词工具名，再提取单词；否则 Power BI 会被拆成两个无意义要求。
    multiword_english = re.findall(
        r"(?<![A-Za-z])(?:Power\s+BI|Google\s+Analytics|Microsoft\s+Excel|Amazon\s+Web\s+Services)(?![A-Za-z])",
        stripped,
        flags=re.IGNORECASE,
    )
    english = re.findall(r"(?<![A-Za-z])[A-Za-z][A-Za-z0-9+#./-]{1,}(?![A-Za-z])", stripped)
    normalized_multiword = [" ".join(value.split()) for value in multiword_english]
    found.extend(normalized_multiword)
    for phrase in normalized_multiword:
        words = {part.casefold() for part in phrase.split()}
        english = [token for token in english if token.casefold() not in words]
    # 某些边界组合会被正则拆开，二次合并并从单词结果中移除。
    for phrase in ("Power BI", "Google Analytics", "Microsoft Excel", "Amazon Web Services"):
        if re.search(re.escape(phrase).replace(r"\ ", r"\s+"), stripped, flags=re.IGNORECASE):
            if phrase not in found:
                found.append(phrase)
            found = [token for token in found if token.casefold() not in {part.casefold() for part in phrase.split()} or token.casefold() == phrase.casefold()]
    covered_words = {
        word.casefold()
        for phrase in found if " " in phrase
        for word in phrase.split()
    }
    ignored = {"and", "or", "the", "with", "using", "use", "experience"}
    ignored |= {
        "product", "team", "role", "owner", "familiar", "good", "communication",
        "ability", "skill", "skills", "responsibility", "responsibilities",
        "preferred", "required", "requirement", "requirements",
    }
    found.extend(
        token for token in english
        if token.casefold() not in covered_words and token.casefold() not in ignored
    )

    # 中文证书名采用明确后缀，避免把整句或连接词误当要求。
    certificate_pattern = re.compile(
        r"([\u4e00-\u9fff]{2,16}?(?:从业资格|职业资格|专业资格|资格证|等级证书|认证证书|证书))"
    )
    for clause in re.split(r"[，,；;。]", stripped):
        for part in re.split(r"[、/]|(?:和|及|或)", clause):
            match = certificate_pattern.search(part)
            if match:
                candidate = re.sub(r"^(?:要求|优先|具备|持有|需要|须有)", "", match.group(1))
                if candidate:
                    found.append(candidate)

    # 中文动态工具只从“使用/掌握/熟悉”后的短名词并列项提取。
    for match in re.finditer(r"(?:熟练)?(?:使用|掌握|熟悉)\s*([^，。；;]+)", stripped):
        tool_text = re.split(r"(?:完成|进行|负责|开展|实现|支持)", match.group(1), maxsplit=1)[0]
        for token in re.split(r"[、/]|(?:和|及|或)", tool_text):
            candidate = token.strip(" ：:，,。；;（）()")
            candidate = re.sub(r"^(?:相关|常用|主流)", "", candidate)
            candidate = re.sub(r"等(?:工具|软件|平台)?$", "", candidate)
            if (
                2 <= len(candidate) <= 16
                and re.fullmatch(r"[\u4e00-\u9fffA-Za-z0-9+#.-]+", candidate)
                and candidate not in {"经验", "能力", "工具", "软件", "平台"}
                and re.search(r"[\u4e00-\u9fff]", candidate)
            ):
                found.append(candidate)
    return list(dict.fromkeys(found))


def _requirement_type(line: str) -> str:
    lowered = line.casefold()
    if any(mark in lowered for mark in ("优先", "加分", "更佳")):
        return "优先条件"
    if any(mark in lowered for mark in ("必须", "需具备", "任职要求", "本科", "硕士", "学历", "证书", "资格")):
        return "硬性条件"
    return "岗位职责"


def _requirement_type_for_clause(clause: str, source_line: str) -> str:
    """优先采用子句标记；无显式标记时继承所在区块类型。"""
    result = _requirement_type(clause)
    if result != "岗位职责":
        return result
    if re.match(r"^(?:任职要求|岗位要求|招聘要求)\s*[：:]?", source_line.strip()):
        return "硬性条件"
    return result


def _requirement_clauses(line: str) -> list[str]:
    """按标点和常见要求标记拆分 JD，避免整行分类污染子要求。"""
    text = re.sub(r"^(?:任职要求|岗位要求|岗位职责|工作内容|招聘要求)\s*[：:]?", "", line.strip())
    parts = re.split(r"[，,；;。！？!?]+|(?<=优先)[：:]|(?<=要求)[：:]", text)
    return [part.strip(" ，,：:") for part in parts if part.strip(" ，,：:")]


def _clause_for_requirement(line: str, requirement: str) -> str:
    clauses = _requirement_clauses(line)
    folded = requirement.casefold()
    for clause in clauses:
        if folded in clause.casefold():
            return clause
    if requirement == "工作经验":
        for clause in clauses:
            if "经验" in clause and re.search(r"\d+(?:\.\d+)?\s*年", clause):
                return clause
        for clause in clauses:
            if "经验" in clause:
                return clause
    return line


def _logic_metadata(clause: str, requirement: str) -> tuple[str, str | None]:
    """返回要求自身的局部逻辑关系及分组键，支持常见 A 且（B 或 C）结构。"""
    folded = requirement.casefold()
    for match in re.finditer(r"[（(]([^）)]+)[）)]", clause):
        inner = match.group(1)
        if folded in inner.casefold():
            relation = "OR" if re.search(r"(?:或|任选其一|之一|\bor\b)", inner.casefold()) else "AND"
            return relation, inner.strip()
    outside = re.sub(r"[（(][^）)]+[）)]", "", clause)
    lowered = outside.casefold() if folded in outside.casefold() else clause.casefold()
    if re.search(r"(?:或|任选其一|二选一|至少一项|之一|\bor\b)", lowered):
        return "OR", clause.strip()
    if re.search(r"(?:和|及|且|并且|同时|以及|\band\b)", lowered):
        return "AND", clause.strip()
    return "单项", None


def _constraint_metadata(clause: str, requirement: str) -> dict:
    """提取年限、年龄等可验证约束，并保留否定语义。"""
    lowered = clause.casefold()
    negative = bool(re.search(r"(?:不要求|不需要|无需|不接受|不得|禁止|无须)", lowered))
    minimum_years = None
    maximum_years = None
    years_range = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*[—\-至到~～]\s*(\d+(?:\.\d+)?)\s*年", clause)
    if years_range:
        minimum_years, maximum_years = (float(years_range.group(1)), float(years_range.group(2)))
    else:
        minimum_years_match = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*年(?:以上|及以上)", clause)
        maximum_years_match = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*年(?:以内|及以下|以下|不超过|至多)", clause)
        if minimum_years_match:
            minimum_years = float(minimum_years_match.group(1))
        elif maximum_years_match:
            maximum_years = float(maximum_years_match.group(1))
    if minimum_years is not None and minimum_years.is_integer():
        minimum_years = int(minimum_years)
    if maximum_years is not None and maximum_years.is_integer():
        maximum_years = int(maximum_years)
    age_range = re.search(r"(?<!\d)(\d{2})\s*[—\-至到~～]\s*(\d{2})\s*岁", clause)
    minimum_age = None
    maximum_age = None
    if age_range:
        minimum_age, maximum_age = int(age_range.group(1)), int(age_range.group(2))
    else:
        minimum_age_match = re.search(r"年龄\s*(\d{1,3})\s*岁?(?:及以上|以上|起)", clause)
        maximum_age_match = re.search(r"年龄\s*(?:不超过|不高于|小于等于|≤|至多)?\s*(\d{1,3})\s*岁?(?:及以下|以下|以内|不超过)?", clause)
        if minimum_age_match:
            minimum_age = int(minimum_age_match.group(1))
        elif maximum_age_match and re.search(r"(?:不超过|不高于|及以下|以下|以内|小于等于|≤)", clause):
            maximum_age = int(maximum_age_match.group(1))
    relation, relation_group = _logic_metadata(clause, requirement)
    education_level = EDUCATION_LEVELS.get(requirement)
    return {
        "relation": relation,
        "relation_group": relation_group,
        "is_negative": negative,
        "operator": "between" if age_range or years_range else ">=" if minimum_years is not None or education_level is not None or minimum_age is not None else "<=" if maximum_years is not None or maximum_age is not None else None,
        "minimum_years": minimum_years,
        "maximum_years": maximum_years,
        "age_range": [minimum_age, maximum_age] if age_range else None,
        "minimum_age": minimum_age,
        "maximum_age": maximum_age,
        "minimum_education_level": education_level,
    }


def _semantic_terms(requirement: str) -> set[str]:
    terms: set[str] = set()
    for concept, aliases in SEMANTIC_CONCEPTS.items():
        folded = {alias.casefold() for alias in aliases}
        if requirement.casefold() == concept.casefold() or requirement.casefold() in folded:
            terms.update(folded)
    return {term for term in terms if len(term) > 1}


def _evidence_for_requirement(fact_ledger: list[dict], requirement: str) -> tuple[list[str], list[str], list[str]]:
    exact_term = requirement.casefold()
    semantic_terms = _semantic_terms(requirement)
    direct_ids: list[str] = []
    semantic_ids: list[str] = []
    matched_terms: list[str] = []
    for fact in fact_ledger:
        text = str(fact.get("text", "")).casefold()
        if exact_term in text:
            direct_ids.append(str(fact["fact_id"]))
            continue
        hits = sorted(term for term in semantic_terms if term in text)
        if hits:
            semantic_ids.append(str(fact["fact_id"]))
            matched_terms.extend(hits)
    return (
        list(dict.fromkeys(direct_ids)),
        list(dict.fromkeys(semantic_ids)),
        list(dict.fromkeys(matched_terms)),
    )


def _years_from_facts(fact_ledger: list[dict], evidence_ids: list[str]) -> float | int | None:
    """仅从直接证据行提取明确年限；不根据日期区间猜算。"""
    by_id = {str(item["fact_id"]): str(item.get("text", "")) for item in fact_ledger}
    values: list[float] = []
    for fact_id in evidence_ids:
        text = by_id.get(str(fact_id), "")
        values.extend(float(value) for value in re.findall(r"(?<!\d)(\d+(?:\.\d+)?)\s*年", text))
    if not values:
        return None
    value = max(values)
    return int(value) if value.is_integer() else value


def _education_from_facts(fact_ledger: list[dict]) -> tuple[str | None, int | None, list[str]]:
    """提取事实账本中明确写出的最高学历，不把学校或专业信息推断成学历。"""
    best_name = None
    best_level = None
    evidence_ids: list[str] = []
    for fact in fact_ledger:
        text = str(fact.get("text", ""))
        for name, level in EDUCATION_LEVELS.items():
            if name in text and (best_level is None or level > best_level):
                best_name, best_level = name, level
                evidence_ids = [str(fact["fact_id"])]
            elif name in text and level == best_level:
                evidence_ids.append(str(fact["fact_id"]))
    return best_name, best_level, list(dict.fromkeys(evidence_ids))


def _age_from_facts(fact_ledger: list[dict]) -> tuple[int | None, list[str]]:
    """仅提取事实账本中明确写出的年龄，不从出生日期自行推算。"""
    for fact in fact_ledger:
        text = str(fact.get("text", ""))
        match = re.search(r"(?<!\d)(\d{1,3})\s*岁", text)
        if match:
            return int(match.group(1)), [str(fact["fact_id"])]
    return None, []


def _evaluate_constraint(
    constraint: dict,
    *,
    exact_hit: bool,
    evidence_ids: list[str],
    evidenced_years: float | int | None,
    evidenced_education: str | None,
    evidenced_education_level: int | None,
    education_evidence_ids: list[str],
    evidenced_age: int | None,
    age_evidence_ids: list[str],
) -> dict:
    """统一评估可验证门槛；优先使用明确事实，不从相邻字段或日期推断。"""
    if constraint["is_negative"]:
        return {"status": "不适用", "level": "不构成门槛", "evidence_ids": evidence_ids}
    if constraint["minimum_years"] is not None or constraint["maximum_years"] is not None:
        if evidenced_years is None:
            return {"status": "待补证据", "level": "待补证据" if exact_hit else "缺口", "evidence_ids": evidence_ids}
        satisfied = (
            (constraint["minimum_years"] is None or evidenced_years >= constraint["minimum_years"])
            and (constraint["maximum_years"] is None or evidenced_years <= constraint["maximum_years"])
        )
        return {
            "status": "满足" if satisfied else "不满足",
            "level": "较强匹配" if satisfied else "缺口",
            "evidence_ids": evidence_ids,
        }
    if constraint["minimum_education_level"] is not None:
        if evidenced_education_level is None:
            return {"status": "待补证据", "level": "缺口", "evidence_ids": education_evidence_ids}
        satisfied = evidenced_education_level >= constraint["minimum_education_level"]
        return {"status": "满足" if satisfied else "不满足", "level": "较强匹配" if satisfied else "缺口", "evidence_ids": education_evidence_ids}
    if constraint["minimum_age"] is not None or constraint["maximum_age"] is not None:
        if evidenced_age is None:
            return {"status": "待补证据", "level": "缺口", "evidence_ids": age_evidence_ids}
        satisfied = (
            (constraint["minimum_age"] is None or evidenced_age >= constraint["minimum_age"])
            and (constraint["maximum_age"] is None or evidenced_age <= constraint["maximum_age"])
        )
        return {"status": "满足" if satisfied else "不满足", "level": "较强匹配" if satisfied else "缺口", "evidence_ids": age_evidence_ids}
    return {"status": None, "level": "较强匹配" if exact_hit else None, "evidence_ids": evidence_ids}


# 将岗位 JD 与简历内容整理成结构化结果。
def build_match_matrix(resume_text: str, jd_text: str = "") -> list[dict]:
    """将岗位要求映射到事实账本，给出可追溯的规则+概念语义判断。"""
    if not jd_text.strip():
        return []
    stop_phrases = {
        "岗位要求", "任职要求", "岗位职责", "职位描述", "工作内容", "招聘要求",
        "熟练", "熟悉", "负责", "参与", "能够", "具备", "优先", "要求",
        "以上", "以下", "相关", "良好", "较强", "协助", "完成", "工作",
    }
    invalid_short_phrases = stop_phrases | {
        "学历", "专业", "能力", "经验", "意识", "岗位", "职位", "团队", "客户",
        "业务", "问题", "要求", "任职", "描述", "内容", "负责的", "以及", "并且",
    }
    jd_lines = [re.sub(r"^[-•*\s]+", "", line).strip("：:") for line in jd_text.splitlines() if line.strip()]
    requirements = []
    seen = set()
    fact_ledger = build_fact_ledger(resume_text)
    evidenced_education, evidenced_education_level, education_evidence_ids = _education_from_facts(fact_ledger)
    evidenced_age, age_evidence_ids = _age_from_facts(fact_ledger)
    for line in jd_lines:
        phrases = _extract_requirements_from_line(line)
        # 斜杠证书/技能拆成候选项，同时保留同一 OR 分组的上下文。
        expanded_phrases = []
        for phrase in phrases:
            if "/" in phrase and re.fullmatch(r"[A-Za-z0-9+#./-]+", phrase):
                expanded_phrases.extend(part for part in phrase.split("/") if part)
            else:
                expanded_phrases.append(phrase)
        keywords = list(dict.fromkeys([
            p for p in expanded_phrases
            if len(p) > 1 and (p == "年龄" or p.casefold() not in {x.casefold() for x in invalid_short_phrases})
        ]))[:8]
        candidates = keywords
        for requirement in candidates:
            if not requirement or requirement in seen:
                continue
            seen.add(requirement)
            clause = _clause_for_requirement(line, requirement)
            item_keywords = [requirement] if requirement != line[:80] else []
            matches = match_keywords(resume_text, item_keywords)
            evidence_ids, semantic_evidence_ids, semantic_hits = _evidence_for_requirement(fact_ledger, requirement)
            exact_hit = any(matches.values())
            level = "较强匹配" if exact_hit else "待补证据" if semantic_evidence_ids else "缺口"
            constraint = _constraint_metadata(clause, requirement)
            evidenced_years = _years_from_facts(fact_ledger, evidence_ids) if constraint["minimum_years"] is not None or constraint["maximum_years"] is not None else None
            evaluation = _evaluate_constraint(
                constraint, exact_hit=exact_hit, evidence_ids=evidence_ids, evidenced_years=evidenced_years,
                evidenced_education=evidenced_education, evidenced_education_level=evidenced_education_level,
                education_evidence_ids=education_evidence_ids, evidenced_age=evidenced_age, age_evidence_ids=age_evidence_ids,
            )
            constraint_status = evaluation["status"]
            evidence_ids = evaluation["evidence_ids"]
            if evaluation["level"] is not None:
                level = evaluation["level"]
            requirements.append({
                "requirement": requirement, "keywords": item_keywords,
                "requirement_context": clause or line,
                "requirement_type": _requirement_type_for_clause(clause or line, line),
                **constraint,
                "evidenced_years": evidenced_years,
                "constraint_status": constraint_status,
                "evidenced_education": evidenced_education,
                "evidenced_education_level": evidenced_education_level,
                "evidenced_age": evidenced_age,
                "evidence_ids": evidence_ids,
                "evidence": "、".join(evidence_ids) if evidence_ids else "未发现可追溯简历证据",
                "semantic_evidence_ids": semantic_evidence_ids,
                "matched_terms": semantic_hits,
                "match_level": level,
                "gap": "" if constraint["is_negative"] or evidence_ids else "仅发现弱语义关联，仍需确认" if semantic_evidence_ids else "简历中未发现该要求的可追溯证据",
                "action": "该条件不构成门槛，无需补充" if constraint["is_negative"] else "保留并补充具体事实" if evidence_ids else "向候选人确认，取得直接证据后再补充",
                "keyword_matches": matches,
            })
            if constraint_status == "待补证据":
                if constraint["minimum_education_level"] is not None:
                    requirements[-1]["gap"] = f"未找到可追溯的{requirement}或更高学历证据"
                    requirements[-1]["action"] = "向候选人确认学校、专业、学历/学位及毕业时间"
                elif constraint["minimum_age"] is not None or constraint["maximum_age"] is not None:
                    age_requirement = f"{constraint['minimum_age']}岁及以上" if constraint["maximum_age"] is None else f"{constraint['maximum_age']}岁及以下" if constraint["minimum_age"] is None else f"{constraint['minimum_age']}—{constraint['maximum_age']}岁"
                    requirements[-1]["gap"] = f"未找到可追溯年龄证据，无法核验{age_requirement}要求"
                    requirements[-1]["action"] = "向候选人确认当前年龄"
                else:
                    years_requirement = f"{constraint['minimum_years']}年以上" if constraint["maximum_years"] is None else f"{constraint['maximum_years']}年以内" if constraint["minimum_years"] is None else f"{constraint['minimum_years']}—{constraint['maximum_years']}年"
                    requirements[-1]["gap"] = f"已发现相关经验，但未找到明确的{years_requirement}年限证据"
                    requirements[-1]["action"] = "向候选人确认明确起止时间或累计年限"
            elif constraint_status == "不满足":
                if constraint["minimum_education_level"] is not None:
                    requirements[-1]["gap"] = f"事实账本最高学历为{evidenced_education}，低于岗位要求的{requirement}"
                    requirements[-1]["action"] = "如实标记学历缺口，不得用在读、课程或学校信息替代学历"
                elif constraint["minimum_age"] is not None or constraint["maximum_age"] is not None:
                    age_requirement = f"{constraint['minimum_age']}岁及以上" if constraint["maximum_age"] is None else f"{constraint['maximum_age']}岁及以下" if constraint["minimum_age"] is None else f"{constraint['minimum_age']}—{constraint['maximum_age']}岁"
                    requirements[-1]["gap"] = f"事实账本年龄为{evidenced_age}岁，不满足{age_requirement}要求"
                    requirements[-1]["action"] = "如实标记年龄条件不匹配"
                else:
                    years_requirement = f"{constraint['minimum_years']}年以上" if constraint["maximum_years"] is None else f"{constraint['maximum_years']}年以内" if constraint["minimum_years"] is None else f"{constraint['minimum_years']}—{constraint['maximum_years']}年"
                    requirements[-1]["gap"] = f"直接证据仅支持{evidenced_years}年，不满足岗位要求的{years_requirement}"
                    requirements[-1]["action"] = "如实保留现有年限，不得改写为满足岗位要求"
    # OR 组按局部子句聚合：命中任一项时，其他候选不再被误报为独立硬缺口。
    groups: dict[str, list[dict]] = {}
    for item in requirements:
        if item.get("relation") == "OR":
            groups.setdefault(item.get("relation_group") or item["requirement_context"], []).append(item)
    for group_items in groups.values():
        group_id = f"G{requirements.index(group_items[0]) + 1:03d}"
        group_satisfied = any(item.get("evidence_ids") for item in group_items)
        for item in group_items:
            item["group_id"] = group_id
            item["group_satisfied"] = group_satisfied
            if group_satisfied and not item.get("evidence_ids"):
                item["match_level"] = "OR组已满足"
                item["gap"] = "同组已有其他候选要求具备直接证据"
                item["action"] = "无需单独补充；保留已命中的同组证据"
    # AND 组保留每个成员的独立证据，同时提供组级状态，便于调用方整体判定。
    and_groups: dict[str, list[dict]] = {}
    for item in requirements:
        if item.get("relation") == "AND":
            and_groups.setdefault(item.get("relation_group") or item["requirement_context"], []).append(item)
    for group_items in and_groups.values():
        group_id = f"G{requirements.index(group_items[0]) + 1:03d}"
        group_satisfied = all(item.get("evidence_ids") and item.get("match_level") == "较强匹配" for item in group_items)
        for item in group_items:
            item["group_id"] = group_id
            item["group_satisfied"] = group_satisfied
            item["group_status"] = "满足" if group_satisfied else "未满足"
    return requirements


def build_confirmation_questions(matrix: list[dict]) -> list[dict]:
    """把证据缺口转换为可回答的问题，而不是直接向简历补写事实。"""
    questions = []
    for index, item in enumerate(matrix, 1):
        if item.get("is_negative") or item.get("constraint_status") == "不适用":
            continue
        constraint_unresolved = item.get("constraint_status") in {"待补证据", "不满足"}
        if (item.get("evidence_ids") and not constraint_unresolved) or item.get("match_level") == "OR组已满足":
            continue
        requirement = str(item["requirement"])
        lowered = requirement.casefold()
        if item.get("minimum_years") is not None or item.get("maximum_years") is not None:
            years_requirement = f"{item['minimum_years']}年以上" if item.get("maximum_years") is None else f"{item['maximum_years']}年以内" if item.get("minimum_years") is None else f"{item['minimum_years']}—{item['maximum_years']}年"
            question = f"你是否满足“{requirement}”的{years_requirement}要求？请提供明确起止时间或累计年限。"
        elif item.get("minimum_age") is not None or item.get("maximum_age") is not None:
            age_requirement = f"{item['minimum_age']}岁及以上" if item.get("maximum_age") is None else f"{item['maximum_age']}岁及以下" if item.get("minimum_age") is None else f"{item['minimum_age']}—{item['maximum_age']}岁"
            question = f"你当前年龄是否满足{age_requirement}要求？请提供可核验证据。"
        elif any(mark in lowered for mark in ("本科", "硕士", "学历", "学位", "专业")):
            question = f"你是否满足“{requirement}”？请提供学校、专业、学历/学位及毕业时间。"
        elif any(mark in lowered for mark in ("证书", "资格", "认证")) or requirement.upper() in {"CFA", "ACCA", "CPA"}:
            question = f"你是否持有“{requirement}”相关证书或资格？请提供证书名称、发证机构及取得时间。"
        else:
            question = f"你是否真实具备“{requirement}”相关经历？如有，请提供具体场景、个人行动、结果及可核验证据。"
        weak_ids = item.get("semantic_evidence_ids", [])
        weak_text = "、".join(weak_ids)
        reason = (
            f"仅发现弱语义关联（{weak_text}），不足以证明该要求；确认前不会写入简历。"
            if weak_ids else "当前事实账本没有可追溯证据；确认前不会写入简历。"
        )
        questions.append({
            "question_id": f"Q{index:03d}",
            "requirement": requirement,
            "requirement_type": item.get("requirement_type", "岗位职责"),
            "question": question,
            "reason": reason,
        })
    return questions


def build_confirmation_questions_report(questions: list[dict]) -> str:
    lines = ["# 待确认事实问题", "", "> 回答后应先更新事实账本，再重新生成简历；未确认内容不会写入。", ""]
    if not questions:
        lines.append("当前岗位要求均已找到候选人证据，无新增待确认问题。")
    for item in questions:
        lines.extend([
            f"## {item['question_id']}｜{item['requirement_type']}｜{item['requirement']}", "",
            item["question"], "", f"- 原因：{item['reason']}", "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def build_analysis(resume_text: str, jd_text: str) -> dict:
    matrix = build_match_matrix(resume_text, jd_text)
    dynamic_keywords = [
        keyword for item in matrix for keyword in item.get("keywords", [])
    ]
    keywords = list(dict.fromkeys(dynamic_keywords))
    keyword_matches = match_keywords(resume_text, keywords)
    return {
        "fact_ledger": build_fact_ledger(resume_text),
        "resume_sections": parse_resume(resume_text),
        "jd_length": len(jd_text),
        "keyword_matches": keyword_matches,
        "matched_keywords": [
            keyword for keyword, matched in keyword_matches.items() if matched
        ],
        "missing_keywords": [
            keyword for keyword, matched in keyword_matches.items() if not matched
        ],
        "match_matrix": matrix,
        "confirmation_questions": build_confirmation_questions(matrix),
    }


def validate_resume_evidence(source_text: str, final_text: str) -> list[str]:
    """检查最终简历中的可疑新增事实是否能在原文中找到依据。

    这是轻量证据门禁，不替代人工核验：只拦截明显的新增数字、日期和强断言，
    避免把通用重排误判为“必须逐字相同”。
    """
    source = source_text.casefold()
    findings: list[str] = []
    for number in sorted(set(re.findall(r"(?<!\d)\d+(?:\.\d+)?(?:年|个月|家|行|个|字段|科目)?", final_text))):
        if number not in source and number not in {"0", "1"}:
            findings.append(f"最终简历出现原文未找到的数字：{number}")
    for phrase in re.findall(r"(?:负责|主导|独立完成|熟练掌握|具备)[^。；\n]{0,24}", final_text):
        if phrase.casefold() not in source and any(mark in phrase for mark in ("负责", "主导", "独立完成", "熟练掌握")):
            findings.append(f"新增强断言需人工核验：{phrase.strip()}")
    return findings


def build_evidence_mapping_report(source_text: str, final_text: str) -> str:
    """按最终稿非空行给出轻量原文证据映射，便于人工复核新增表述。"""
    source = source_text.casefold()

    def meaningful_fragments(text: str) -> list[str]:
        chunks = re.findall(r"[A-Za-z][A-Za-z0-9+#./-]{1,}|[一-龥]+", text.casefold())
        fragments: list[str] = []
        for chunk in chunks:
            if re.fullmatch(r"[一-龥]+", chunk):
                fragments.extend(chunk[index:index + 2] for index in range(len(chunk) - 1))
            else:
                fragments.append(chunk)
        return [fragment for fragment in set(fragments) if fragment not in {
            "个人信息", "求职意向", "教育背景", "工作经历", "项目经历", "技能证书", "自我评价",
            "本科", "工作", "经历", "实践", "岗位定向简历",
        }]
    rows = []
    for line in final_text.splitlines():
        clean = re.sub(r"[*#>`]", "", line).strip(" -")
        if not clean or clean.startswith("真实性说明"):
            continue
        fragments = meaningful_fragments(clean)
        source_hits = [fragment for fragment in fragments if fragment in source]
        if clean.casefold() in source:
            status = "直接命中原文"
        elif len(clean) <= 4 or clean in {"个人信息", "求职意向", "教育背景", "工作经历", "项目经历", "技能证书", "自我评价"}:
            status = "结构性内容"
        elif fragments and source_hits:
            status = "部分命中，需核对改写"
        else:
            status = "需人工核对"
        safe = clean.replace("|", "／")[:100]
        rows.append(f"| {len(rows) + 1} | {status} | {safe} |")
    return "\n".join([
        "# 最终简历逐条证据映射",
        "",
        "> 本报告是轻量自动筛查；“需人工核对”不等于错误，投递前必须核对事实、时间和个人贡献。",
        "",
        "| 序号 | 证据状态 | 最终简历片段 |",
        "|---:|---|---|",
        *rows,
        "",
        f"- 需人工核对或改写复核：{sum(('需人工核对' in row or '部分命中' in row) for row in rows)} 条。",
    ]) + "\n"


def build_report(analysis: dict) -> str:
    """把结构化分析转换为便于阅读的 Markdown 报告。"""
    matrix = analysis["match_matrix"]
    level_order = ["较强匹配", "部分匹配", "待补证据", "弱匹配", "缺口"]
    counts = {
        level: sum(item["match_level"] == level for item in matrix)
        for level in level_order
    }

    lines = [
        "# 岗位匹配分析",
        "",
        "## 一、匹配总览",
        "",
        "| 匹配等级 | 数量 |",
        "|---|---:|",
    ]
    lines.extend(f"| {level} | {counts[level]} |" for level in level_order)

    lines.extend(["", "## 二、逐项分析", ""])
    for index, item in enumerate(matrix, start=1):
        lines.extend([
            f"### {index}. {item['requirement']}",
            f"- 要求类型：{item.get('requirement_type', '岗位职责')}",
            f"- 匹配判断：{item['match_level']}",
            f"- 简历证据：{item['evidence']}",
            f"- 弱语义关联：{'、'.join(item.get('matched_terms', [])) or '无'}",
            f"- 弱关联证据：{'、'.join(item.get('semantic_evidence_ids', [])) or '无'}",
            f"- 当前缺口：{item['gap']}",
            f"- 修改建议：{item['action']}",
            "",
        ])

    lines.extend([
        "## 三、关键词结果",
        "",
        f"- 已命中：{'、'.join(analysis['matched_keywords'])}",
        f"- 暂未命中：{'、'.join(analysis['missing_keywords'])}",
        "",
        "## 四、真实性边界",
        "",
        "- 数据采集、结构化处理、字段校验和质量报告可以作为真实实践表述。",
        "- 未确认的推荐、风控、用户挖掘、机器学习模型和大数据框架经验不能直接写成已掌握。",
        "- 具体数字只使用已有数据文件能够核验的记录数和字段数。",
    ])
    return "\n".join(lines) + "\n"


def build_llm_prompt(resume_text: str, jd_text: str, analysis: dict) -> str:
    """构造供后续大模型调用的提示词；本步骤只生成，不发送。"""
    matrix_json = json.dumps(analysis["match_matrix"], ensure_ascii=False, indent=2)
    facts_json = json.dumps(analysis["fact_ledger"], ensure_ascii=False, indent=2)
    return f"""你是一名严谨的简历分析助手。

任务：基于岗位 JD 和候选人简历，输出岗位匹配分析与修改建议。

必须遵守：
1. 只能使用简历中已有事实，不得编造项目、模型、指标、数据规模或工作成果。
2. 区分“已证实”“部分匹配”“待补证据”“缺口”，不要为了 ATS 机械添加关键词。
3. 如果经历由 AI 辅助完成，使用“使用 AI 编程工具辅助”或“参与完成”等准确表述。
4. 对机器学习、推荐、风控、用户挖掘、Hadoop、Spark、Flink 等内容，没有证据就明确写缺口。
5. 先给结论，再给修改后的简历片段；输出中文。
6. 最终只返回合法 JSON，不要使用 Markdown 代码围栏或额外解释。
7. JSON 必须包含 conclusion、strengths、gaps、resume_revisions 四个字段；其中后三者为列表。
8. resume_revisions 每项必须包含 section、content、evidence_ids；evidence_ids 只能引用下方事实账本中的 fact_id。
9. 无法引用事实账本的建议只能放入 gaps，不得写入 resume_revisions。
10. 核心项目优先按“技术/架构、核心实现、安全边界、验证结果”展开；没有直接证据的层级必须省略，禁止为了显得完整而补写。
11. 版面空间不足时优先删除低相关内容，不得靠极小字号硬塞一页；岗位相关项目应写清问题、动作和可验证结果。
12. 修改岗位标题只影响简历顶部求职方向，不得误删公司职位、获奖名称等正文中真实存在的“实习生”字样。

岗位 JD：
{jd_text}

简历：
{resume_text}

规则分析结果：
{matrix_json}

候选人事实账本：
{facts_json}

请输出：
一、岗位匹配结论
二、优势证据
三、缺口与风险
四、建议保留的关键词
五、可直接替换的简历片段
"""


def validate_llm_result_schema(result: dict) -> dict:
    """校验已解析的 LLM 结果；Python 与 LangGraph 共用同一契约。"""
    if not isinstance(result, dict):
        raise ValueError("LLM 返回 JSON 顶层必须是对象")

    required_fields = {"conclusion", "strengths", "gaps", "resume_revisions"}
    missing_fields = required_fields - result.keys()
    if missing_fields:
        raise ValueError(f"LLM JSON 缺少字段：{', '.join(sorted(missing_fields))}")
    if not isinstance(result["strengths"], list) or not isinstance(result["gaps"], list):
        raise ValueError("LLM JSON 中 strengths 和 gaps 必须是列表")
    if not isinstance(result["resume_revisions"], list):
        raise ValueError("LLM JSON 中 resume_revisions 必须是列表")
    for index, item in enumerate(result["resume_revisions"], 1):
        if not isinstance(item, dict):
            raise ValueError(f"LLM JSON 第{index}条 resume_revisions 必须是对象")
        missing = {"section", "content", "evidence_ids"} - item.keys()
        if missing:
            raise ValueError(f"LLM JSON 第{index}条改写缺少字段：{', '.join(sorted(missing))}")
        if not isinstance(item["evidence_ids"], list):
            raise ValueError(f"LLM JSON 第{index}条 evidence_ids 必须是列表")
        if not item["evidence_ids"] or not all(isinstance(value, str) and value.strip() for value in item["evidence_ids"]):
            raise ValueError(f"LLM JSON 第{index}条 evidence_ids 必须是非空字符串列表")
        item["evidence_ids"] = list(dict.fromkeys(value.strip() for value in item["evidence_ids"]))
    return result


def parse_llm_json(text: str) -> dict:
    """解析模型返回的 JSON，兼容 Markdown 代码围栏。"""
    cleaned = text.strip()
    if cleaned.startswith("```json") and cleaned.endswith("```"):
        cleaned = cleaned[7:-3].strip()
    elif cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned[3:-3].strip()
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as error:
        raise ValueError("LLM 返回内容不是有效 JSON") from error
    return validate_llm_result_schema(result)


def validate_resume_draft(draft: str) -> list[str]:
    """检查定向简历是否把未证实能力写成既成事实。"""
    forbidden_claims = [
        "熟练R",
        "熟练 Hadoop",
        "熟练Spark",
        "熟练 Flink",
        "机器学习建模成果",
        "完成特征工程",
        "推荐系统开发",
        "风险控制模型",
        "用户挖掘项目",
    ]
    violations = [claim for claim in forbidden_claims if claim in draft]
    # 真实性门禁不仅拦固定短语，也拦常见的未经证实强断言组合。
    patterns = [
        r"(?:精通|熟练掌握|独立完成|负责)\s*(?:R|Hadoop|Spark|Flink|SVM|逻辑回归|神经网络)",
        r"(?:具备|拥有|完成过)\s*[^。；\n]{0,20}(?:推荐系统|风控模型|特征工程|用户画像)",
    ]
    for pattern in patterns:
        if re.search(pattern, draft, flags=re.IGNORECASE):
            violations.append(f"命中未证实强断言模式：{pattern}")
    return violations


def build_targeted_resume_draft(resume_text: str, llm_result: dict) -> str:
    """将模型返回的可替换片段整理为人工复核用的岗位定向简历草稿。"""
    revisions, rejected = validate_llm_revisions(resume_text, llm_result)
    lines = [
        "# 算法工程师-数据挖掘岗位定向简历草稿",
        "",
        "> 状态：候选稿，必须由本人逐条核对后再投递。未新增模型、指标、框架或业务成果。",
        "",
    ]
    for item in revisions:
        if not isinstance(item, dict):
            continue
        section = item.get("section", "未命名模块")
        if section == "不建议当前添加的内容":
            continue
        content = item.get("content", "")
        if isinstance(content, list):
            content = "、".join(str(value) for value in content)
        lines.extend([f"## {section}", "", str(content), ""])

    if not revisions:
        lines.extend(["## 原简历内容", "", resume_text.strip(), ""])
    if rejected:
        lines.extend(["## 未采用的改写", ""])
        lines.extend(f"- {finding}" for finding in rejected)
        lines.append("")
    lines.extend([
        "## 投递前真实性核对",
        "",
        "- 核对每个项目的个人贡献、代码或文档证据及项目时间。",
        "- 未实际完成的算法、框架、特征处理和业务建模不得添加。",
        "- 数字仅沿用已有简历证据，不将数据整理规模包装为海量数据。",
    ])
    draft = "\n".join(lines) + "\n"
    violations = validate_resume_draft(draft)
    if violations:
        raise ValueError("定向简历触发真实性门禁：" + "、".join(violations))
    return draft


def build_llm_alignment_report(
    llm_result: dict,
    final_resume: str,
    resume_text: str | None = None,
) -> str:
    """生成模型建议与最终稿的一致性报告，不把未经核验建议静默写入简历。"""
    revisions = llm_result.get("resume_revisions", [])
    accepted, rejected = validate_llm_revisions(resume_text, llm_result) if resume_text is not None else ([], [])
    accepted_keys = {
        (item["section"], item["content"], tuple(item["evidence_ids"])) for item in accepted
    }
    lines = [
        "# 模型建议与最终简历一致性核对",
        "",
        "> 最终简历只保留原始简历可核验事实；模型建议未自动视为事实。",
        "",
        "| 序号 | 模块 | 状态 | 说明 |",
        "|---:|---|---|---|",
    ]
    if not revisions:
        lines.append("| 1 | — | 未提供建议 | 没有可对照的模型修改片段。 |")
    else:
        for index, item in enumerate(revisions, 1):
            if not isinstance(item, dict):
                lines.append(f"| {index} | — | 已跳过 | 建议项不是结构化对象。 |")
                continue
            section = str(item.get("section", "未命名模块"))
            content = item.get("content", "")
            if isinstance(content, list):
                content = "、".join(str(value) for value in content)
            content = str(content).strip()
            key = (section, content, tuple(item.get("evidence_ids", [])))
            if key in accepted_keys and _revision_is_applied(resume_text or "", item, final_resume):
                status, note = "已采纳", "证据校验通过，且片段位于最终简历中。"
            elif section == "不建议当前添加的内容":
                status, note = "未采用", "根据真实性边界主动排除。"
            else:
                status, note = "已拒绝", "未通过自动证据校验，需人工核对。"
            safe_content = content.replace("|", "／").replace("\n", " ")[:80]
            lines.append(f"| {index} | {section} | {status} | {note} {safe_content} |")
    lines.extend([
        "",
        "## 使用规则",
        "",
        "- “已拒绝”不代表建议必然错误，只代表当前没有通过自动证据校验。",
        "- 只有本人确认事实、时间、个人贡献和证据后，才可手动修改最终稿。",
    ])
    return "\n".join(lines) + "\n"


def build_revision_diff_report(
    resume_text: str,
    llm_result: dict,
    final_resume: str,
) -> str:
    """逐条展示建议、证据和最终处理状态，供投递前人工批准。"""
    accepted, rejected = validate_llm_revisions(resume_text, llm_result)
    accepted_keys = {
        (item["section"], item["content"], tuple(item["evidence_ids"])) for item in accepted
    }
    lines = [
        "# 简历改写差异报告", "",
        "> 本报告只解释改写决策；已拒绝内容不会进入最终简历。", "",
        "| 序号 | 模块 | 建议内容 | 证据 | 处理结果 | 原因 |",
        "|---:|---|---|---|---|---|",
    ]
    revisions = llm_result.get("resume_revisions", [])
    for index, item in enumerate(revisions, 1):
        if not isinstance(item, dict):
            lines.append(f"| {index} | — | 非结构化建议 | — | 拒绝 | schema 不合法 |")
            continue
        section = str(item.get("section", "未命名模块")).replace("|", "／")
        content = str(item.get("content", "")).replace("|", "／").replace("\n", " ")[:120]
        evidence_ids = [str(value) for value in item.get("evidence_ids", [])]
        key = (str(item.get("section", "")), str(item.get("content", "")).strip(), tuple(evidence_ids))
        adopted = key in accepted_keys and _revision_is_applied(resume_text, item, final_resume)
        if adopted:
            status, reason = "采纳", "证据校验通过且已进入最终稿"
        else:
            status, reason = "拒绝", "未通过证据校验或未进入最终稿"
        evidence_text = "、".join(evidence_ids) or "无"
        lines.append(
            f"| {index} | {section} | {content} | {evidence_text} | {status} | {reason} |"
        )
    if not revisions:
        lines.append("| 1 | — | 没有模型改写建议 | — | 无变更 | 本次仅生成本地基础版本 |")
    if rejected:
        lines.extend(["", "## 自动拒绝明细", ""])
        lines.extend(f"- {finding}" for finding in rejected)
    return "\n".join(lines) + "\n"


WRITABLE_RESUME_SECTIONS = (
    "个人信息", "求职意向", "教育背景", "工作经历", "项目经历", "技能证书", "自我评价",
)


def _revision_is_applied(resume_text: str, revision: dict, final_resume: str) -> bool:
    """确认改写确实进入目标模块，避免原文同词造成误报。"""
    section = str(revision.get("section", ""))
    content = str(revision.get("content", "")).strip()
    if section not in WRITABLE_RESUME_SECTIONS or not content:
        return False
    parsed = parse_resume(resume_text)
    if section not in {"个人信息", "求职意向"} and not parsed.get(section):
        return False
    pattern = rf"(?ms)^## {re.escape(section)}\s*$\n(.*?)(?=^## |\Z)"
    match = re.search(pattern, final_resume)
    return bool(match and content in match.group(1))


def build_final_resume(
    resume_text: str | None = None,
    llm_result: dict | None = None,
    jd_text: str = "",
    accepted_revisions: list[dict] | None = None,
    job_title: str | None = None,
) -> str:
    """生成最终简历。

    当前岗位保留经过人工确认的事实模板；传入新的简历时，使用原简历内容
    构建通用最终稿，并把模型建议作为岗位定向模块插入，避免把模型输出当成
    未核验事实。最终稿始终经过真实性门禁。
    """
    if resume_text is not None and resume_text.strip():
        parsed = parse_resume(resume_text)
        title = job_title.strip() if job_title and job_title.strip() else "个人简历"
        if not (job_title and job_title.strip()) and jd_text.strip():
            title = "岗位定向简历"
            for jd_line in jd_text.splitlines():
                candidate = re.sub(r"^\s*(岗位名称|职位名称|职位|岗位)\s*[：:]\s*", "", jd_line).strip()
                if candidate and candidate != jd_line.strip():
                    title = candidate
                    break
        personal = parsed.get("个人信息", [])
        name = personal[0] if personal else "候选人"
        lines = [f"# {name}", "", f"**求职方向：{title}**", ""]
        accepted_by_section: dict[str, list[str]] = {}
        accepted = accepted_revisions
        if accepted is None and llm_result and llm_result.get("resume_revisions"):
            accepted, _ = validate_llm_revisions(resume_text, llm_result)
        for revision in accepted or []:
            accepted_by_section.setdefault(revision["section"], []).append(revision["content"])
        for section in WRITABLE_RESUME_SECTIONS:
            values = parsed.get(section, [])
            if not values and section not in {"个人信息", "求职意向"}:
                continue
            lines.extend([f"## {section}", ""])
            if values:
                lines.extend(values)
            elif section == "个人信息":
                lines.append(name)
            elif section == "求职意向":
                lines.append(title)
            for revision in accepted_by_section.get(section, []):
                if revision not in values:
                    lines.append(revision)
            lines.append("")
        resume = "\n".join(lines) + "\n"
        violations = validate_resume_draft(resume)
        if violations:
            raise ValueError("最终简历触发真实性门禁：" + "、".join(violations))
        return resume

    """无输入时使用脱敏演示版本，避免源码携带个人简历数据。"""
    resume = """# 示例候选人

**求职方向：示例岗位**

## 个人信息

姓名：示例候选人

## 求职意向

示例岗位

## 自我评价

具备经本人核验的教育、项目和工作经历；此处为脱敏演示内容，不能替代真实候选人简历。

## 教育背景

某大学｜示例专业｜本科｜2023.09—2027.06

## 项目经历

- 示例项目经历（请替换为本人已核验内容）

## 工作经历

- 示例工作经历（请替换为本人已核验内容）

## 技能证书

- 示例技能：Excel、Python

> **真实性说明：** 这是脱敏演示版本。投递前请替换为本人真实且可核验的内容。
"""
    violations = validate_resume_draft(resume)
    if violations:
        raise ValueError("最终简历触发真实性门禁：" + "、".join(violations))
    return resume


def render_resume_image(
    resume_text: str,
    photo_path: Path,
    output_path: Path,
    job_title: str | None = None,
    template: str = "sales",
) -> None:
    """生成自动高度的单栏带照片简历，头像与正文使用独立区域。"""
    if not photo_path.is_file():
        raise FileNotFoundError(f"找不到照片：{photo_path}")
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    font_path = _image_font_path()
    bold_path = _image_font_path(bold=True)
    W, margin, header_h = 1600, 105, 390
    if template == "ats":
        navy, blue, ink, muted, line, white = "#202020", "#202020", "#202020", "#555555", "#CFCFCF", "#FFFFFF"
    else:
        navy, blue, ink, muted, line, white = "#102A43", "#1F6FEB", "#243B53", "#627D98", "#D9E2EC", "#FFFFFF"
    name = ImageFont.truetype(bold_path, 62)
    subtitle = ImageFont.truetype(font_path, 30)
    section = ImageFont.truetype(bold_path, 34)
    job = ImageFont.truetype(bold_path, 28)
    body = ImageFont.truetype(font_path, 25)
    small = ImageFont.truetype(font_path, 24)
    # 先使用可扩展画布，正文结束后再裁切，避免固定高度导致长简历截断。
    img = Image.new("RGB", (W, 12000), white)
    draw = ImageDraw.Draw(img)

    # 顶部头像区：正文绝不进入 header_h 以内。
    draw.rectangle((0, 0, W, header_h), fill=navy)
    photo = ImageOps.fit(Image.open(photo_path).convert("RGB"), (220, 280), method=Image.Resampling.LANCZOS, centering=(.5, .25))
    px, py = W - margin - 220, 45
    draw.rectangle((px - 8, py - 8, px + 228, py + 288), fill="#D9EAF7")
    img.paste(photo, (px, py))
    photo_rect = (px - 8, py - 8, px + 228, py + 288)
    text_rects = []

    def draw_header_text(position, text, font, fill):
        bbox = draw.textbbox(position, text, font=font)
        text_rects.append(bbox)
        draw.text(position, text, font=font, fill=fill)

    parsed_header = parse_resume(resume_text)
    personal = parsed_header.get("个人信息", [])
    header_name = personal[0] if personal else "候选人"
    header_title = job_title or "岗位定向简历"
    header_meta = "｜".join(personal[1:3]) if len(personal) > 1 else ""
    header_max_width = px - 105 - 32

    def fit_header_text(text, font, max_width):
        if draw.textlength(text, font=font) <= max_width:
            return text
        suffix = "…"
        while text and draw.textlength(text + suffix, font=font) > max_width:
            text = text[:-1]
        return text + suffix if text else suffix

    header_title = fit_header_text(header_title, subtitle, header_max_width)
    header_meta = fit_header_text(header_meta, small, header_max_width)
    draw_header_text((105, 72), header_name, name, white)
    draw_header_text((105, 160), header_title, subtitle, "#B9D7F5")
    draw_header_text((105, 225), header_meta, small, white)

    def intersects(a, b):
        return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]

    collisions = [bbox for bbox in text_rects if intersects(bbox, photo_rect)]
    if collisions:
        raise ValueError(f"照片与文字区域重叠：{len(collisions)}处")

    def wrap(text, font, width):
        def measured_width(value):
            bbox = draw.textbbox((0, 0), value, font=font)
            return bbox[2] - bbox[0]

        rows, current, token = [], "", ""
        parts=[]
        for ch in text:
            if ch.isascii() and (ch.isalnum() or ch in "_./:-"):
                token += ch
            else:
                if token: parts.append(token); token=""
                parts.append(ch)
        if token: parts.append(token)
        for part in parts:
            # 超长英文/URL token 不能整段塞入，否则会越过右边界；按字符拆分。
            if measured_width(part) > width:
                if current:
                    rows.append(current); current = ""
                chunk = ""
                for char in part:
                    if measured_width(chunk + char) > width and chunk:
                        rows.append(chunk); chunk = char
                    else:
                        chunk += char
                current = chunk
            elif measured_width(current + part) > width and current:
                rows.append(current); current = part
            else:
                current += part
        if current: rows.append(current)
        return rows

    def write_wrapped(text, y, font=small, leading=36):
        # Leave a small deterministic cushion for font-side-bearing and
        # rounding differences across Pillow/font implementations.
        for row in wrap(text, font, W - 2 * margin - 8):
            bbox = draw.textbbox((margin, y), row, font=font)
            if intersects(bbox, photo_rect):
                raise ValueError("照片与正文文字区域重叠")
            if bbox[0] < margin or bbox[2] > W - margin:
                raise ValueError("正文文字超出左右边界")
            draw.text((margin, y), row, font=font, fill=ink)
            y += leading
        return y

    # 从最终 Markdown 读取真实正文，按标题/项目/工作经历重排；不渲染真实性说明。
    lines=[]
    for raw in resume_text.splitlines():
        raw=raw.strip()
        if raw.startswith("> **真实性说明："): break
        if raw: lines.append(raw.replace("**", ""))
    y=header_h+48
    first=True
    for raw in lines:
        if raw.startswith("# ") or raw.startswith("求职方向："):
            continue
        if raw.startswith("## "):
            y += 12
            heading_bbox = draw.textbbox((margin+25, y), raw[3:], font=section)
            if intersects(heading_bbox, photo_rect):
                raise ValueError("照片与正文标题区域重叠")
            if heading_bbox[0] < margin or heading_bbox[2] > W - margin:
                raise ValueError("正文标题超出左右边界")
            draw.rectangle((margin, y, margin+9, y+38), fill=blue)
            draw.text((margin+25, y), raw[3:], font=section, fill=navy)
            draw.line((margin+25, y+50, W-margin, y+50), fill=line, width=2)
            y += 78
        elif raw.startswith("-"):
            y = write_wrapped(raw, y, small, 36) + 6
        else:
            y = write_wrapped(raw, y, job, 39) + 5
    # 求职方向已放在顶部姓名下方，不在底部重复添加。
    final_h = y + 35
    img.crop((0, 0, W, final_h)).save(output_path)

def mock_llm(prompt: str) -> str:
    """本地模拟模型，仅用于测试 Agent 的输出链路，不访问网络。"""
    return json.dumps(
        {
            "conclusion": "具备 Python 数据处理、公开数据集构建和 AI Agent 实践证据；机器学习建模与大数据框架经验需要补证。",
            "strengths": ["Python 数据处理", "公开数据集质量校验", "AI Agent 项目实践"],
            "gaps": ["缺少已确认的机器学习建模指标", "未确认 Hadoop/Spark/Flink 实践"],
            "resume_revisions": [
                {"section": "技能证书", "content": "Excel、Python", "evidence_ids": ["F006"]},
                {"section": "工作经历", "content": "参与资料整理、客户沟通和Excel汇总。", "evidence_ids": ["F005"]},
            ],
        },
        ensure_ascii=False,
    )


def _run_cli() -> None:
    """读取简历和 JD，输出分析结果，并可选调用 LLM。"""
    parser = argparse.ArgumentParser(description="简历与岗位 JD 分析 Agent")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--call-llm",
        action="store_true",
        help="真正调用模型；需要外部提供 API_BASE_URL、API_KEY、MODEL",
    )
    mode_group.add_argument(
        "--mock-llm",
        action="store_true",
        help="使用本地模拟模型，不访问网络",
    )
    parser.add_argument(
        "--render-image",
        action="store_true",
        help="生成美化版带照片简历图片",
    )
    parser.add_argument(
        "--export-package",
        action="store_true",
        help="同步生成 ATS Markdown、单页 A4 PDF、PNG 预览和 QA 报告",
    )
    parser.add_argument(
        "--photo",
        type=Path,
        default=None,
        help="证件照路径，默认使用 input/photo.jpg",
    )
    parser.add_argument("--resume", type=Path, default=None, help="自定义简历文本路径")
    parser.add_argument("--jd", type=Path, default=None, help="自定义岗位 JD 文本路径")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录；默认使用项目目录下 output/<template>/",
    )
    parser.add_argument("--job-title", default=None, help="图片头部显示的岗位名称")
    parser.add_argument(
        "--template",
        choices=("sales", "ats"),
        default="sales",
        help="输出模板；sales 为商务稳重版，ats 为单栏机器筛选版",
    )
    parser.add_argument(
        "--workflow",
        choices=("python", "langgraph"),
        default="python",
        help="工作流实现；默认 python，langgraph 需要额外安装依赖",
    )
    args = parser.parse_args()
    started_at = datetime.now(timezone.utc).isoformat()

    project_dir = Path(__file__).parent
    resume_path = args.resume or (project_dir / "input" / "resume.txt")
    jd_path = args.jd or (project_dir / "input" / "jd.txt")

    try:
        resume_text = read_text_file(resume_path)
        jd_text = read_text_file(jd_path)
    except FileNotFoundError as error:
        _print_error(f"读取失败：找不到文件 {error.filename}", "Read failed: file not found")
        raise SystemExit(2)
    except UnicodeDecodeError:
        _print_error("读取失败：输入文件不是有效的 UTF-8 文本", "Read failed: input is not valid UTF-8")
        raise SystemExit(2)

    analysis = build_analysis(resume_text, jd_text)
    output_dir = args.output_dir or (project_dir / "output" / args.template)
    if not output_dir.is_absolute():
        output_dir = project_dir / output_dir
    prepare_output_dir(output_dir)
    write_private_text(output_dir / "analysis.json", json.dumps(analysis, ensure_ascii=False, indent=2) + "\n")
    write_private_text(output_dir / "match-report.md", build_report(analysis))
    write_private_text(
        output_dir / "confirmation-questions.md",
        build_confirmation_questions_report(analysis["confirmation_questions"]),
    )
    # 无 LLM 模式直接生成已核验固定版本；LLM 模式等解析完成后再生成最终版，
    # 避免中间稿覆盖最终稿或被误当成交付物。
    final_resume = None
    if not (args.call_llm or args.mock_llm):
        if args.workflow == "langgraph":
            try:
                from workflow import run_langgraph_workflow
                final_resume = run_langgraph_workflow(
                    resume_text, jd_text, job_title=args.job_title
                )["final_resume"]
            except (RuntimeError, ValueError) as error:
                _print_error(f"LangGraph工作流不可用：{error}", "LangGraph workflow unavailable")
                raise SystemExit(2)
        else:
            from workflow import run_python_workflow
            final_resume = run_python_workflow(resume_text, jd_text, job_title=args.job_title)["final_resume"]
        violations = validate_resume_draft(final_resume)
        violations.extend(validate_resume_evidence(resume_text, final_resume))
        if violations:
            _print_error("最终简历触发真实性门禁：" + "、".join(violations), "Resume failed truth validation")
            raise SystemExit(2)
        if args.render_image:
            photo_path = args.photo or (project_dir / "input" / "photo.jpg")
            if not photo_path.is_absolute() and not photo_path.is_file():
                photo_path = project_dir / photo_path
            try:
                render_resume_image(final_resume, photo_path, output_dir / "final-resume-with-photo.png", args.job_title, args.template)
            except (FileNotFoundError, OSError, subprocess.SubprocessError, ValueError) as error:
                _print_error(f"图片生成失败：{error}", "Image generation failed")
                raise SystemExit(2)

    if args.mock_llm:
        raw_result = mock_llm(build_llm_prompt(resume_text, jd_text, analysis))
        parsed_result = parse_llm_json(raw_result)
        write_private_text(
            output_dir / "llm-analysis.json",
            json.dumps(parsed_result, ensure_ascii=False, indent=2) + "\n",
        )
        draft = build_targeted_resume_draft(resume_text, parsed_result)
        write_private_text(output_dir / "targeted-resume-draft.md", draft)
        try:
            if args.workflow == "langgraph":
                from workflow import run_langgraph_workflow
                final_resume = run_langgraph_workflow(
                    resume_text, jd_text, parsed_result, job_title=args.job_title
                )["final_resume"]
            else:
                final_resume = build_final_resume(resume_text, parsed_result, jd_text, job_title=args.job_title)
        except ValueError as error:
            _print_error(f"最终简历生成失败：{error}", "Resume generation failed")
            raise SystemExit(2)
        violations = validate_resume_draft(final_resume)
        violations.extend(validate_resume_evidence(resume_text, final_resume))
        if violations:
            _print_error("最终简历触发真实性门禁：" + "、".join(violations), "Resume failed truth validation")
            raise SystemExit(2)
        write_private_text(
            output_dir / "llm-alignment-report.md",
            build_llm_alignment_report(parsed_result, final_resume, resume_text),
        )
        write_private_text(
            output_dir / "revision-diff-report.md",
            build_revision_diff_report(resume_text, parsed_result, final_resume),
        )
        if args.render_image:
            photo_path = args.photo or (project_dir / "input" / "photo.jpg")
            if not photo_path.is_absolute() and not photo_path.is_file():
                photo_path = project_dir / photo_path
            try:
                render_resume_image(final_resume, photo_path, output_dir / "final-resume-with-photo.png", args.job_title, args.template)
            except (FileNotFoundError, OSError, subprocess.SubprocessError, ValueError) as error:
                _print_error(f"图片生成失败：{error}", "Image generation failed")
                raise SystemExit(2)
    elif args.call_llm:
        try:
            raw_result = call_llm(build_llm_prompt(resume_text, jd_text, analysis))
            parsed_result = parse_llm_json(raw_result)
        except (RuntimeError, ValueError) as error:
            _print_error(f"LLM调用或解析失败：{error}", "LLM call or parsing failed")
            raise SystemExit(2)
        write_private_text(
            output_dir / "llm-analysis.json",
            json.dumps(parsed_result, ensure_ascii=False, indent=2) + "\n",
        )
        try:
            draft = build_targeted_resume_draft(resume_text, parsed_result)
        except ValueError as error:
            _print_error(f"定向简历生成失败：{error}", "Targeted resume generation failed")
            raise SystemExit(2)
        write_private_text(output_dir / "targeted-resume-draft.md", draft)
        try:
            if args.workflow == "langgraph":
                from workflow import run_langgraph_workflow
                final_resume = run_langgraph_workflow(
                    resume_text, jd_text, parsed_result, job_title=args.job_title
                )["final_resume"]
            else:
                final_resume = build_final_resume(resume_text, parsed_result, jd_text, job_title=args.job_title)
        except ValueError as error:
            _print_error(f"最终简历生成失败：{error}", "Resume generation failed")
            raise SystemExit(2)
        violations = validate_resume_draft(final_resume)
        violations.extend(validate_resume_evidence(resume_text, final_resume))
        if violations:
            _print_error("最终简历触发真实性门禁：" + "、".join(violations), "Resume failed truth validation")
            raise SystemExit(2)
        write_private_text(
            output_dir / "llm-alignment-report.md",
            build_llm_alignment_report(parsed_result, final_resume, resume_text),
        )
        write_private_text(
            output_dir / "revision-diff-report.md",
            build_revision_diff_report(resume_text, parsed_result, final_resume),
        )
        if args.render_image:
            photo_path = args.photo or (project_dir / "input" / "photo.jpg")
            if not photo_path.is_absolute() and not photo_path.is_file():
                photo_path = project_dir / photo_path
            try:
                render_resume_image(final_resume, photo_path, output_dir / "final-resume-with-photo.png", args.job_title, args.template)
            except (FileNotFoundError, OSError, subprocess.SubprocessError, ValueError) as error:
                _print_error(f"图片生成失败：{error}", "Image generation failed")
                raise SystemExit(2)

    qa_result = None
    if args.export_package:
        if final_resume is None:
            _print_error("交付包生成失败：没有可用的最终简历", "Delivery package failed: no resume")
            raise SystemExit(2)
        try:
            from delivery import export_delivery_package
            photo_path = args.photo or (project_dir / "input" / "photo.jpg")
            if not photo_path.is_absolute() and not photo_path.is_file():
                photo_path = project_dir / photo_path
            qa_result = export_delivery_package(
                final_resume,
                output_dir,
                photo_path=photo_path if photo_path.is_file() else None,
            )
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            _print_error(f"交付包生成失败：{error}", "Delivery package generation failed")
            raise SystemExit(2)
        write_private_text(
            output_dir / "qa-report.json",
            json.dumps(qa_result, ensure_ascii=False, indent=2) + "\n",
        )

    # 正式 Markdown 只在真实性门禁及可选交付 QA 全部通过后发布。
    if final_resume is None:
        _print_error("最终简历生成失败：没有可用的最终简历", "Resume generation failed: no resume")
        raise SystemExit(2)
    write_private_text(output_dir / "final-resume.md", final_resume)
    write_private_text(
        output_dir / "evidence-mapping-report.md",
        build_evidence_mapping_report(resume_text, final_resume),
    )

    managed_names = {
        "analysis.json", "match-report.md", "final-resume.md",
        "final-resume-with-photo.png", "llm-analysis.json", "targeted-resume-draft.md",
        "llm-alignment-report.md", "evidence-mapping-report.md",
        "confirmation-questions.md", "revision-diff-report.md",
        "final-resume-ats.md", "final-resume.pdf", "final-resume.png", "qa-report.json",
    }
    produced_paths = sorted(
        (output_dir / name for name in managed_names if (output_dir / name).is_file()),
        key=lambda path: path.name,
    )
    produced = [path.name for path in produced_paths]
    for path in output_dir.iterdir():
        if path.is_file():
            path.chmod(0o600)
    manifest = {
        "status": "success",
        "run_id": secrets.token_hex(12),
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "workflow": args.workflow,
        "template": args.template,
        "mode": "llm" if args.call_llm else "mock" if args.mock_llm else "local",
        "input_sha256": {
            "resume": hashlib.sha256(resume_text.encode("utf-8")).hexdigest(),
            "jd": hashlib.sha256(jd_text.encode("utf-8")).hexdigest(),
        },
        "qa": qa_result,
        "produced": sorted(set(produced)),
        "artifacts": {
            path.name: {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
                "mode": oct(path.stat().st_mode & 0o777),
            }
            for path in produced_paths
        },
    }
    write_private_text(
        output_dir / "run-manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    manifest_path = output_dir / "run-manifest.json"
    write_private_text(
        output_dir / "run-manifest.sha256",
        hashlib.sha256(manifest_path.read_bytes()).hexdigest() + "  run-manifest.json\n",
    )
    print("运行完成")
    print(f"最终简历：{output_dir / 'final-resume.md'}")
    print(f"运行清单：{manifest_path}")
    if args.export_package:
        print(f"单页PDF：{output_dir / 'final-resume.pdf'}")
        print(f"PNG预览：{output_dir / 'final-resume.png'}")
        print(f"ATS Markdown：{output_dir / 'final-resume-ats.md'}")


def _print_error(message: str, fallback: str) -> None:
    """Keep CLI diagnostics usable when stderr is a legacy non-UTF-8 stream."""
    try:
        print(message, file=sys.stderr)
    except UnicodeEncodeError:
        print(fallback, file=sys.stderr)


def _configure_console_streams() -> None:
    """Use UTF-8 for real console streams while tolerating test doubles."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            # Keep the platform encoding for parent-process compatibility;
            # backslash replacement prevents unsupported Unicode from aborting
            # a successful CLI run on legacy Windows consoles.
            reconfigure(errors="backslashreplace")
        except (AttributeError, OSError, ValueError):
            continue


def main() -> None:
    """统一 CLI 异常边界：已知失败保留退出码，未知失败不泄露堆栈或敏感正文。"""
    previous_umask = os.umask(0o077)
    try:
        _configure_console_streams()
        _run_cli()
    except SystemExit:
        raise
    except Exception as error:
        _print_error(f"运行失败：{type(error).__name__}", f"Run failed: {type(error).__name__}")
        raise SystemExit(2) from None
    finally:
        os.umask(previous_umask)


if __name__ == "__main__":
    main()
