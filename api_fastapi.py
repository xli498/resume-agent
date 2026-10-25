"""FastAPI 接口层（Demo 新增，与标准库 web_app.py 的端点一一对应）。

对应关系（请求/响应口径与 web_app.py 保持一致）：
  POST /api/analyze   POST /api/dig   POST /api/generate
  POST /api/pdf       GET  /api/pdf?task_id=   GET /api/tasks/{task_id}
新增：
  GET  /api/health          健康检查
  POST /api/agent_dig       LangChain 多轮 Agent 拷打（见 agent_dig.py）
  GET  /api/store/{task_id} SQLite 落盘查询（会话 + 账本 + 拷打记录）

安全机制（与 PR #4 原样一致，不重新发明）：
  - 任务存储直接复用 web_app.TASKS / _public_task：GET /api/tasks/{id} 仍剥离
    resume、jd、created_at；绑定仍走 _require_loopback（仅 127.0.0.1/::1）。
  - POST 统一校验 Content-Type 必须为 application/json、请求体不能为空、
    必须是 JSON 对象——与 web_app.Handler._body() 的校验一一对应。
  - 核心判定（claim_ledger / ATS / PDF）全部调用原函数，本层只做"接线+落盘"。

持久化：分析/深挖成功后写 SQLite（store.py）。写库失败只记 stderr 日志，
不改变请求结果——落盘是观测手段，不是新的门禁。
"""
from __future__ import annotations

import secrets
import sys
import time
from typing import Any

from fastapi import Body, FastAPI, Request
from fastapi.responses import JSONResponse, Response

import agent_dig
import store
from claim_ledger import build_interview_defense
from pdf_export import render_pdf
from web_app import (
    TASKS,
    TASKS_LOCK,
    _prune_tasks_locked,
    _public_task,
    _require_loopback,
)
from workflow import apply_dig_answers, run_python_workflow


def _persist(fn, *args, **kwargs) -> None:
    """落盘失败只记日志，不改请求结果（落盘不是门禁）。"""
    try:
        fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - 落盘失败不得阻断主流程
        print(f"[store] persist failed: {type(exc).__name__}", file=sys.stderr)


app = FastAPI(title="Resume Agent (FastAPI Demo Layer)")


@app.middleware("http")
async def _require_json_body(request: Request, call_next):  # noqa: ANN001,ANN201
    """镜像 web_app._body() 的 Content-Type 与空体校验（仅 POST）。"""
    if request.method == "POST":
        if request.headers.get("content-type", "").split(";")[0].strip() != "application/json":
            return JSONResponse({"error": "请求必须使用 application/json"}, status_code=400)
        if int(request.headers.get("content-length", "0") or "0") == 0:
            return JSONResponse({"error": "请求体不能为空"}, status_code=400)
    return await call_next(request)


def _bad(message: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


def _get_task(task_id: str) -> dict | None:
    with TASKS_LOCK:
        _prune_tasks_locked()
        return TASKS.get(task_id)


@app.get("/api/health")
def health() -> dict:
    with TASKS_LOCK:
        _prune_tasks_locked()
        count = len(TASKS)
    return {"status": "ok", "service": "resume-agent-fastapi", "tasks": count}


@app.post("/api/analyze")
def analyze(payload: Any = Body(...)) -> Any:
    if not isinstance(payload, dict):
        return _bad("请求体必须是 JSON 对象")
    resume, jd = str(payload.get("resume", "")).strip(), str(payload.get("jd", "")).strip()
    if not resume or not jd:
        return _bad("简历和 JD 不能为空")
    finance_flag = payload.get("finance_mode")
    finance_mode = True if finance_flag is True else None
    try:
        result = run_python_workflow(
            resume, jd, job_title=payload.get("job_title") or None, finance_mode=finance_mode
        )
    except ValueError as exc:
        return _bad(str(exc))
    task_id = secrets.token_urlsafe(10)
    task = {
        "task_id": task_id,
        "status": "awaiting_confirmation",
        "created_at": time.monotonic(),
        "resume": resume,
        "jd": jd,
        "job_title": payload.get("job_title") or None,
        **result,
    }
    with TASKS_LOCK:
        _prune_tasks_locked()
        if len(TASKS) >= 32:
            oldest = min(TASKS, key=lambda key: TASKS[key]["created_at"])
            TASKS.pop(oldest, None)
        TASKS[task_id] = task
    _persist(
        store.save_analysis,
        task_id,
        result["claims"],
        job_title=payload.get("job_title") or None,
        finance_mode=finance_mode,
    )
    return {
        "task_id": task_id,
        "status": task["status"],
        "analysis": result["analysis"],
        "fact_ledger": result["fact_ledger"],
        "interview_defense": result["interview_defense"],
        "dig_questions": result["dig_questions"],
        "match_coverage": result["match_coverage"],
        "ats_checks": result["ats_checks"],
        "ats_score": result["ats_score"],
    }


@app.post("/api/dig")
def dig(payload: Any = Body(...)) -> Any:
    if not isinstance(payload, dict):
        return _bad("请求体必须是 JSON 对象")
    task_id = str(payload.get("task_id", ""))
    qa_pairs = payload.get("answers", [])
    task = _get_task(task_id)
    if not task:
        return _bad("任务不存在", 404)
    dig_result = apply_dig_answers(task["resume"], task["claims"], qa_pairs)
    with TASKS_LOCK:
        task["claims"] = dig_result["claims"]
        task["interview_defense"] = dig_result["interview_defense"]
    _persist(store.save_dig, task_id, dig_result["claims"], qa_pairs)
    return {
        "task_id": task_id,
        "added": dig_result["added"],
        "claims": dig_result["claims"],
        "interview_defense": dig_result["interview_defense"],
        "claim_errors": dig_result["claim_errors"],
    }


@app.post("/api/generate")
def generate(payload: Any = Body(...)) -> Any:
    if not isinstance(payload, dict):
        return _bad("请求体必须是 JSON 对象")
    task_id = str(payload.get("task_id", ""))
    task = _get_task(task_id)
    if not task:
        return _bad("任务不存在", 404)
    with TASKS_LOCK:
        if task["status"] != "awaiting_confirmation":
            return _bad("任务状态不允许生成", 409)
        task["status"] = "generated"
    _persist(store.update_status, task_id, "generated")
    return {"task_id": task_id, "status": task["status"], "final_resume": task["final_resume"]}


def _pdf_bytes(task_id: str) -> bytes | None:
    task = _get_task(task_id)
    if task and task.get("final_resume"):
        return render_pdf(task["final_resume"])
    return None


@app.post("/api/pdf")
def pdf_post(payload: Any = Body(...)) -> Any:
    if not isinstance(payload, dict):
        return _bad("请求体必须是 JSON 对象")
    pdf_bytes = _pdf_bytes(str(payload.get("task_id", "")))
    if not pdf_bytes:
        return _bad("任务不存在或未生成", 404)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=resume.pdf"},
    )


@app.get("/api/pdf")
def pdf_get(task_id: str = "") -> Any:
    pdf_bytes = _pdf_bytes(task_id)
    if not pdf_bytes:
        return _bad("任务不存在或未生成", 404)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=resume.pdf"},
    )


@app.get("/api/tasks/{task_id}")
def get_task(task_id: str) -> Any:
    task = _get_task(task_id)
    if not task:
        return _bad("任务不存在", 404)
    return _public_task(task)


@app.post("/api/agent_dig")
def agent_dig_endpoint(payload: Any = Body(...)) -> Any:
    """多轮 Agent 拷打：Thought→Action→Observation，轮次日志可查。"""
    if not isinstance(payload, dict):
        return _bad("请求体必须是 JSON 对象")
    task_id = str(payload.get("task_id", ""))
    task = _get_task(task_id)
    if not task:
        return _bad("任务不存在", 404)
    try:
        max_rounds = int(payload.get("max_rounds", 3))
    except (TypeError, ValueError):
        return _bad("max_rounds 必须是整数")
    result = agent_dig.run_agent_dig(
        task["resume"],
        task["claims"],
        answers=payload.get("answers"),
        max_rounds=max_rounds,
        task_id=task_id,
    )
    with TASKS_LOCK:
        task["claims"] = result["claims"]
        task["interview_defense"] = build_interview_defense(result["claims"])
    _persist(store.save_dig, task_id, result["claims"], [])
    _persist(store.save_agent_rounds, task_id, result["rounds"])
    return {
        "task_id": task_id,
        "rounds": result["rounds"],
        "claims": result["claims"],
        "coverage_errors": result["coverage_errors"],
        "llm_used": result["llm_used"],
        "mode": result["mode"],
        "langchain_available": result["langchain_available"],
    }


@app.get("/api/store/{task_id}")
def get_store(task_id: str) -> Any:
    session = store.get_session(task_id)
    if session is None:
        return _bad("会话未落盘或不存在", 404)
    return session


def serve(host: str = "127.0.0.1", port: int = 8771) -> None:
    """仅允许 loopback 绑定（复用 web_app._require_loopback）。"""
    import uvicorn

    _require_loopback(host)
    print(f"Resume Agent (FastAPI) 已启动：http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")


def main() -> None:
    serve()


if __name__ == "__main__":
    main()
