"""拷打环节的 LangChain Agent 化（Demo 新增）。

把原来一次性的 3 轮 question-tree 深挖，改成可观测的多轮 Agent 循环：

    Thought（覆盖缺口是什么）→ Action（调工具）→ Observation（覆盖变化）

工具（langchain_core @tool，真实被循环调用）：
  - check_coverage：validate_defense_coverage，算高风险 Claim 覆盖缺口
  - next_question：build_dig_questions，按优先级取下一个未问过的深挖问题
  - apply_answer：apply_dig_answers，把用户回答转成新 Claim 并重建防御

模型适配：优先复用 llm_client.call_llm（环境变量 API_BASE_URL/API_KEY/MODEL）。
拿不到密钥或调用失败时，循环不中断、不编造：走规则回退（沿用 question-tree
原问题），并在每轮记录 mode=rule_fallback、llm_used=False。报告里必须如实写
"未验证 LLM 真调用"，不许把回退结果说成模型输出。

隐私：JSONL 日志只记 question_id、mode、answered、added 数量，不记简历原文与回答正文。
日志路径可用环境变量 RESUME_AGENT_DIG_LOG 覆盖，默认 output/agent_dig.jsonl。
诚实性补充：未匹配本轮问出 question_id 的回答列入 ignored，不静默丢弃；
停止原因 stopped_reason 区分问题池耗尽（question_pool_exhausted，
build_dig_questions 每快照最多出 max_q=3 个不同问题，故 max_rounds>3
也可能提前停止）与跑满轮次（max_rounds_reached）。
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from claim_ledger import (
    build_dig_questions,
    build_interview_defense,
    validate_defense_coverage,
)
from llm_client import call_llm
from workflow import apply_dig_answers

try:
    from langchain_core.tools import tool as _lc_tool

    LANGCHAIN_AVAILABLE = True
except ImportError:  # pragma: no cover - 仅无 langchain-core 的环境
    LANGCHAIN_AVAILABLE = False

    def _lc_tool(func):  # type: ignore[no-redef]
        """无 langchain-core 时的占位装饰器：循环逻辑不变，工具照常是函数。"""
        func.invoke = lambda payload: func(**payload)  # type: ignore[attr-defined]
        return func


@_lc_tool
def check_coverage(claims_json: str) -> str:
    """算高风险 Claim 的面试防御覆盖缺口。claims_json 为 claims 列表的 JSON。"""
    claims = json.loads(claims_json)
    defense = build_interview_defense(claims)
    errors = validate_defense_coverage(defense, claims)
    return json.dumps({"errors": errors}, ensure_ascii=False)


@_lc_tool
def next_question(claims_json: str, asked_json: str) -> str:
    """取下一个未问过的深挖问题。asked_json 为已问 (claim_id, dimension) 对的 JSON。"""
    claims = json.loads(claims_json)
    asked = {tuple(item) for item in json.loads(asked_json)}
    for question in build_dig_questions(claims, max_q=3):
        if (question["claim_id"], question["dimension"]) not in asked:
            return json.dumps(question, ensure_ascii=False)
    return json.dumps({}, ensure_ascii=False)


@_lc_tool
def apply_answer(
    resume_text: str, claims_json: str, question_id: str, claim_id: str, answer: str
) -> str:
    """把一条深挖回答转成新 Claim。返回新增数与最新账本。"""
    claims = json.loads(claims_json)
    result = apply_dig_answers(
        resume_text,
        claims,
        [{"question_id": question_id, "claim_id": claim_id, "answer": answer}],
    )
    return json.dumps(
        {"added": result["added"], "claims": result["claims"]},
        ensure_ascii=False,
    )


def _normalize_answers(
    answers: list[dict[str, Any]] | dict[str, str] | None,
) -> dict[str, dict[str, str]]:
    """回答归一化为 {question_id: {claim_id, answer}}。"""
    normalized: dict[str, dict[str, str]] = {}
    if isinstance(answers, dict):
        for question_id, answer in answers.items():
            normalized[str(question_id)] = {"claim_id": "", "answer": str(answer)}
    elif isinstance(answers, list):
        for item in answers:
            question_id = str(item.get("question_id", ""))
            if question_id:
                normalized[question_id] = {
                    "claim_id": str(item.get("claim_id", "")),
                    "answer": str(item.get("answer", "")),
                }
    return normalized


def _try_llm_question(claim_text: str, dimension: str, rule_question: str) -> tuple[str, bool, str]:
    """尝试用 LLM 改写深挖问题。返回 (问题, 是否用了 LLM, mode)。

    失败原因只记异常类型，不记异常正文（防上游回显进日志）。
    """
    prompt = json.dumps(
        {
            "task": "针对下面这条简历事实，写一个深挖问题，只返回 JSON：{\"question\": \"...\"}",
            "dimension": dimension,
            "claim_hint": claim_text[:120],
            "rule_question": rule_question,
        },
        ensure_ascii=False,
    )
    try:
        raw = call_llm(prompt)
        parsed = json.loads(raw)
        question = str(parsed.get("question", "")).strip()
        if question:
            return question, True, "llm"
    except Exception as exc:  # noqa: BLE001 - 回退路径必须接住一切调用失败
        return rule_question, False, f"rule_fallback:{type(exc).__name__}"
    return rule_question, False, "rule_fallback:empty_llm_output"


def _rule_challenge(claim_text: str) -> str:
    """规则回退质疑：不编造细节，只追问可验证口径。"""
    hint = claim_text[:60]
    return f"面试官可能追问：{hint} 里的数字/结果，能否给出可验证的口径、来源或前后对比？"


def _try_llm_challenge(claim_text: str, dimension: str) -> tuple[str, bool, str]:
    """尝试用 LLM 生成面试官质疑。返回 (质疑, 是否用了 LLM, mode)。

    失败原因只记异常类型，不记异常正文（防上游回显进日志）；
    质疑只记录、不判定防御成败（本轮不裁决 Claim 真假）。
    """
    prompt = json.dumps(
        {
            "task": "你是面试官，针对下面这条简历事实写一个尖锐的质疑，只返回 JSON：{\"challenge\": \"...\"}",
            "dimension": dimension,
            "claim_hint": claim_text[:120],
        },
        ensure_ascii=False,
    )
    try:
        raw = call_llm(prompt)
        parsed = json.loads(raw)
        challenge = str(parsed.get("challenge", "")).strip()
        if challenge:
            return challenge, True, "llm"
    except Exception as exc:  # noqa: BLE001 - 回退路径必须接住一切调用失败
        return _rule_challenge(claim_text), False, f"rule_fallback:{type(exc).__name__}"
    return _rule_challenge(claim_text), False, "rule_fallback:empty_llm_output"


def run_agent_dig(
    resume_text: str,
    claims: list[dict[str, Any]],
    answers: list[dict[str, Any]] | dict[str, str] | None = None,
    max_rounds: int = 3,
    task_id: str | None = None,
    challenge: bool = False,
    max_challenges: int = 2,
) -> dict[str, Any]:
    """多轮 Agent 拷打循环。返回 rounds（每轮 Thought/Action/Observation）与最终状态。

    诚实性补充（R5）：本轮问出过的 question_id 集合记为 asked；
    提供了回答但 question_id 从未被本轮问出的，列入 ignored（不静默丢弃）。
    停止原因 stopped_reason：question_pool_exhausted（问题池耗尽——
    build_dig_questions 每快照最多出 max_q=3 个不同问题，故 max_rounds>3
    也可能提前停止）或 max_rounds_reached（跑满指定轮次）。
    每轮另记 answered（是否提供了非空回答），回答原文永不进轮次与日志。
    质疑轮（challenge=True 时）：主循环结束后，对仍有覆盖缺口的高风险 Claim
    各跑一轮面试官质疑（对标多智能体辩论的反方视角），默认关闭——
    不开启时行为与旧版完全一致；质疑只记录、不判定防御成败，轮次放在
    result["challenge"]["rounds"]（不混入主 rounds），日志与响应均可查。
    """
    normalized_answers = _normalize_answers(answers)
    current: list[dict[str, Any]] = [dict(claim) for claim in claims]
    asked_keys: set[tuple[str, str]] = set()
    asked_ids: set[str] = set()
    rounds: list[dict[str, Any]] = []
    llm_used_any = False
    stopped_reason = "max_rounds_reached"

    for round_no in range(1, max(1, max_rounds) + 1):
        coverage_raw = check_coverage.invoke(
            {"claims_json": json.dumps(current, ensure_ascii=False)}
        )
        coverage_errors = json.loads(coverage_raw)["errors"]
        question_raw = next_question.invoke(
            {
                "claims_json": json.dumps(current, ensure_ascii=False),
                "asked_json": json.dumps(sorted(asked_keys), ensure_ascii=False),
            }
        )
        question = json.loads(question_raw)
        if not question:
            stopped_reason = "question_pool_exhausted"
            rounds.append(
                {
                    "round": round_no,
                    "thought": f"覆盖缺口：{coverage_errors}；但已无新问题可问",
                    "action": "stop",
                    "tool": None,
                    "observation": "问题池耗尽，循环结束",
                    "mode": "stop",
                    "llm_used": False,
                    "question_id": "",
                    "question": "",
                    "answer": "",
                    "answered": False,
                    "added": 0,
                }
            )
            break
        asked_keys.add((question["claim_id"], question["dimension"]))
        asked_ids.add(question["question_id"])
        claim_text = next(
            (c.get("text", "") for c in current if c.get("claim_id") == question["claim_id"]),
            "",
        )
        final_question, llm_used, mode = _try_llm_question(
            claim_text, question["dimension"], question["question"]
        )
        llm_used_any = llm_used_any or llm_used
        supplied = normalized_answers.get(question["question_id"])
        answered = bool(supplied and supplied["answer"].strip())
        added = 0
        if answered:
            applied_raw = apply_answer.invoke(
                {
                    "resume_text": resume_text,
                    "claims_json": json.dumps(current, ensure_ascii=False),
                    "question_id": question["question_id"],
                    "claim_id": question["claim_id"],
                    "answer": supplied["answer"],
                }
            )
            applied = json.loads(applied_raw)
            added = int(applied["added"])
            current = applied["claims"]
            observation = f"回答已转 Claim，新增 {added} 条，账本现 {len(current)} 条"
        else:
            observation = "未提供该题回答；问题记为待回答，账本不变"
        rounds.append(
            {
                "round": round_no,
                "thought": f"覆盖缺口：{coverage_errors}；下一问指向 {question['dimension']}",
                "action": "next_question" + ("+apply_answer" if added or supplied else ""),
                "tool": "next_question",
                "observation": observation,
                "mode": mode,
                "llm_used": llm_used,
                "question_id": question["question_id"],
                "question": final_question,
                "answer": "",
                "answered": answered,
                "added": added,
            }
        )

    ignored = [
        {"question_id": qid, "reason": "该 question_id 在本轮未被问出，回答未纳入账本"}
        for qid in normalized_answers
        if qid not in asked_ids
    ]
    final_defense = build_interview_defense(current)
    final_errors = validate_defense_coverage(final_defense, current)
    challenge_rounds: list[dict[str, Any]] = []
    if challenge:
        target_ids: list[str] = []
        for err in final_errors:
            m = re.match(r"高风险 Claim (\S+) 没有被面试防御覆盖", err)
            if m and m.group(1) not in target_ids:
                target_ids.append(m.group(1))
        for i, cid in enumerate(target_ids[: max(0, max_challenges)], start=1):
            claim_text = next(
                (c.get("text", "") for c in current if c.get("claim_id") == cid),
                "",
            )
            dimension = next(
                (d for (c, d) in asked_keys if c == cid),
                "",
            )
            challenge_text, c_used, c_mode = _try_llm_challenge(claim_text, dimension)
            llm_used_any = llm_used_any or c_used
            # P2-2：质疑依据是截断后的文本（LLM 120 字 / 规则 60 字），如实标注。
            truncated = len(claim_text) > (120 if c_used else 60)
            challenge_rounds.append(
                {
                    "round": i,
                    "thought": f"针对高风险 Claim {cid} 的防御缺口发起质疑",
                    "action": "challenge",
                    "tool": "challenge",
                    "observation": "质疑已记录，待面试准备中回应；本轮不判定防御成败",
                    "mode": c_mode,
                    "llm_used": c_used,
                    "question_id": "",
                    "question": challenge_text,
                    "answer": "",
                    "answered": False,
                    "added": 0,
                    "claim_id": cid,
                    "input_truncated": truncated,
                }
            )
    result: dict[str, Any] = {
        "rounds": rounds,
        "claims": current,
        "coverage_errors": final_errors,
        "llm_used": llm_used_any,
        "mode": "llm" if llm_used_any else "rule_fallback",
        "langchain_available": LANGCHAIN_AVAILABLE,
        "ignored": ignored,
        "stopped_reason": stopped_reason,
        "challenge": {"rounds": challenge_rounds, "challenged": len(challenge_rounds)},
    }
    _append_log(task_id, result)
    return result


def _log_path() -> Path:
    """日志路径：环境变量 RESUME_AGENT_DIG_LOG 可覆盖，默认 output/agent_dig.jsonl。"""
    env = os.environ.get("RESUME_AGENT_DIG_LOG", "").strip()
    if env:
        return Path(env)
    return Path(__file__).parent / "output" / "agent_dig.jsonl"


def _append_log(task_id: str | None, result: dict[str, Any]) -> None:
    """追加 JSONL 运行日志：只记轮次、mode、added、answered，不记原文与回答。"""
    log_path = _log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        {
            "ts": time.time(),
            "task_id": task_id,
            "rounds": [
                {
                    "round": r["round"],
                    "mode": r["mode"],
                    "llm_used": r["llm_used"],
                    "question_id": r["question_id"],
                    "answered": bool(r.get("answered")),
                    "added": r["added"],
                }
                for r in result["rounds"]
            ],
            "llm_used": result["llm_used"],
            "mode": result["mode"],
            "final_claims": len(result["claims"]),
            "challenge": {
                "challenged": len(result.get("challenge", {}).get("rounds", [])),
                "rounds": [
                    {
                        "round": r["round"],
                        "mode": r["mode"],
                        "llm_used": r["llm_used"],
                        "claim_id": r.get("claim_id", ""),
                    }
                    for r in result.get("challenge", {}).get("rounds", [])
                ],
            },
        },
        ensure_ascii=False,
    )
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
