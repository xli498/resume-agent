"""可选工作流编排层。

默认不依赖 LangGraph；核心节点仍由 main.py 提供，便于本地运行和回滚。
安装 LangGraph 后可通过 build_langgraph_workflow() 接入图编排，不改变核心业务门禁。
"""
from __future__ import annotations

from typing import Any, TypedDict

from main import (
    build_analysis,
    build_fact_ledger,
    build_final_resume,
    build_targeted_resume_draft,
    parse_llm_json,
    validate_llm_revisions,
    validate_llm_result_schema,
    validate_resume_draft,
    validate_resume_evidence,
)


class ResumeWorkflowState(TypedDict, total=False):
    resume_text: str
    jd_text: str
    llm_result: dict[str, Any]
    analysis: dict[str, Any]
    final_resume: str
    violations: list[str]
    fact_ledger: list[dict[str, Any]]
    accepted_revisions: list[dict[str, Any]]
    rejected_revisions: list[str]
    confirmation_questions: list[dict[str, Any]]
    targeted_draft: str
    job_title: str


def run_python_workflow(
    resume_text: str,
    jd_text: str,
    llm_raw: str | None = None,
    job_title: str | None = None,
) -> dict[str, Any]:
    """执行无框架的标准流程，作为默认和回滚路径。"""
    analysis = build_analysis(resume_text, jd_text)
    fact_ledger = build_fact_ledger(resume_text)
    analysis["fact_ledger"] = fact_ledger
    result: dict[str, Any] = {
        "analysis": analysis,
        "fact_ledger": fact_ledger,
        "confirmation_questions": analysis["confirmation_questions"],
    }
    if llm_raw is None:
        # 回退路径也必须使用调用方传入的事实，不得悄悄回到示例固定模板。
        final_resume = build_final_resume(resume_text, {}, jd_text, job_title=job_title)
        result.update({"accepted_revisions": [], "rejected_revisions": []})
    else:
        llm_result = parse_llm_json(llm_raw)
        accepted, rejected = validate_llm_revisions(resume_text, llm_result)
        draft = build_targeted_resume_draft(resume_text, llm_result)
        final_resume = build_final_resume(
            resume_text, None, jd_text, accepted_revisions=accepted, job_title=job_title
        )
        result.update({
            "llm_result": llm_result,
            "targeted_draft": draft,
            "accepted_revisions": accepted,
            "rejected_revisions": rejected,
        })
    violations = validate_resume_draft(final_resume)
    violations.extend(validate_resume_evidence(resume_text, final_resume))
    if violations:
        raise ValueError("最终简历触发真实性门禁：" + "、".join(violations))
    result["final_resume"] = final_resume
    return result


def build_langgraph_workflow():
    """可选构建 LangGraph；未安装时明确失败，不影响默认路径。"""
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as error:
        raise RuntimeError("未安装 LangGraph；当前仍可使用 run_python_workflow()") from error

    graph = StateGraph(ResumeWorkflowState)

    def extract_facts(state):
        return {"fact_ledger": build_fact_ledger(state["resume_text"])}

    def analyze(state):
        analysis = build_analysis(state["resume_text"], state["jd_text"])
        analysis["fact_ledger"] = state["fact_ledger"]
        return {
            "analysis": analysis,
            "confirmation_questions": analysis["confirmation_questions"],
        }

    def ground_revisions(state):
        accepted, rejected = validate_llm_revisions(
            state["resume_text"], {"resume_revisions": state.get("llm_result", {}).get("resume_revisions", [])}
        )
        result: dict[str, Any] = {
            "accepted_revisions": accepted,
            "rejected_revisions": rejected,
        }
        if state.get("llm_result"):
            result["targeted_draft"] = build_targeted_resume_draft(
                state["resume_text"], state["llm_result"]
            )
        return result

    def finalize(state):
        return {"final_resume": build_final_resume(
            state["resume_text"],
            None,
            state["jd_text"],
            state.get("accepted_revisions", []),
            job_title=state.get("job_title"),
        )}

    def validate(state):
        violations = validate_resume_draft(state["final_resume"])
        violations.extend(validate_resume_evidence(
            state["resume_text"], state["final_resume"]
        ))
        if violations:
            raise ValueError("最终简历触发真实性门禁：" + "、".join(violations))
        return {"violations": violations}

    graph.add_node("extract_facts", extract_facts)
    graph.add_node("analyze", analyze)
    graph.add_node("ground_revisions", ground_revisions)
    graph.add_node("finalize", finalize)
    graph.add_node("validate", validate)
    graph.set_entry_point("extract_facts")
    graph.add_edge("extract_facts", "analyze")
    graph.add_edge("analyze", "ground_revisions")
    graph.add_edge("ground_revisions", "finalize")
    graph.add_edge("finalize", "validate")
    graph.add_edge("validate", END)
    return graph.compile()


def run_langgraph_workflow(
    resume_text: str,
    jd_text: str,
    llm_result: dict | None = None,
    job_title: str | None = None,
) -> dict[str, Any]:
    """使用 LangGraph 编排同一套核心节点；依赖不存在时给出明确提示。"""
    if llm_result is not None:
        llm_result = validate_llm_result_schema(llm_result)
    workflow = build_langgraph_workflow()
    result = workflow.invoke({
        "resume_text": resume_text,
        "jd_text": jd_text,
        "llm_result": llm_result or {},
        "job_title": job_title or "",
    })
    return result
