"""SQLite 持久化层（Demo 新增，标准库 sqlite3，无新依赖）。

落盘内容：会话 + 主张账本 + 拷打记录。只做"写透"，不改变任何核心判定：
调用方（FastAPI 层）在分析/深挖成功后写库，写库失败只记日志、不改请求结果。

默认库文件：环境变量 RESUME_AGENT_DB，否则仓库 output/resume_agent.db
（output/ 已在 .gitignore 中；另补 *.db 忽略，双重保险）。
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any


def default_db_path() -> Path:
    """每次调用都重读环境变量，便于测试注入临时库。"""
    env = os.environ.get("RESUME_AGENT_DB", "").strip()
    if env:
        return Path(env)
    return Path(__file__).parent / "output" / "resume_agent.db"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    task_id TEXT PRIMARY KEY,
    job_title TEXT,
    finance_mode INTEGER,
    status TEXT,
    created_at REAL
);
CREATE TABLE IF NOT EXISTS claims (
    task_id TEXT,
    claim_id TEXT,
    claim_text TEXT,
    origin TEXT,
    risk_flags_json TEXT,
    PRIMARY KEY (task_id, claim_id)
);
CREATE TABLE IF NOT EXISTS dig_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT,
    round_no INTEGER,
    question_id TEXT,
    question TEXT,
    answer TEXT,
    mode TEXT,
    created_at REAL
);
"""


def init_db(path: Path | None = None) -> Path:
    """建库建表（幂等）。返回实际库路径。"""
    db = Path(path) if path is not None else default_db_path()
    db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as conn:
        conn.executescript(_SCHEMA)
    return db


def _connect(path: Path | None = None) -> sqlite3.Connection:
    db = init_db(path)
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    return conn


def save_analysis(
    task_id: str,
    claims: list[dict[str, Any]],
    job_title: str | None = None,
    finance_mode: bool | None = None,
    status: str = "awaiting_confirmation",
    path: Path | None = None,
) -> None:
    """分析成功后写会话 + 全量主张账本（同一 task_id 幂等覆盖）。"""
    with _connect(path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO sessions(task_id, job_title, finance_mode, status, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (task_id, job_title, 1 if finance_mode else 0, status, time.time()),
        )
        conn.execute("DELETE FROM claims WHERE task_id = ?", (task_id,))
        for claim in claims:
            conn.execute(
                "INSERT OR REPLACE INTO claims(task_id, claim_id, claim_text, origin, risk_flags_json)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    task_id,
                    str(claim.get("claim_id", "")),
                    str(claim.get("text", "")),
                    str(claim.get("origin", "")),
                    json.dumps(claim.get("risk_flags", []), ensure_ascii=False),
                ),
            )


def save_dig(
    task_id: str,
    claims: list[dict[str, Any]],
    qa_pairs: list[dict[str, Any]],
    path: Path | None = None,
) -> None:
    """深挖成功后：账本全量覆盖 + 每条回答记一条拷打记录。"""
    with _connect(path) as conn:
        conn.execute("DELETE FROM claims WHERE task_id = ?", (task_id,))
        for claim in claims:
            conn.execute(
                "INSERT OR REPLACE INTO claims(task_id, claim_id, claim_text, origin, risk_flags_json)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    task_id,
                    str(claim.get("claim_id", "")),
                    str(claim.get("text", "")),
                    str(claim.get("origin", "")),
                    json.dumps(claim.get("risk_flags", []), ensure_ascii=False),
                ),
            )
        for qa in qa_pairs:
            conn.execute(
                "INSERT INTO dig_records(task_id, round_no, question_id, question, answer, mode, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    task_id,
                    0,
                    str(qa.get("question_id", "")),
                    "",
                    str(qa.get("answer", "")),
                    "user_answer",
                    time.time(),
                ),
            )


def save_agent_rounds(
    task_id: str,
    rounds: list[dict[str, Any]],
    path: Path | None = None,
) -> None:
    """Agent 多轮循环的每一轮记一条记录（question/answer 均可为空=待回答）。"""
    with _connect(path) as conn:
        for rnd in rounds:
            conn.execute(
                "INSERT INTO dig_records(task_id, round_no, question_id, question, answer, mode, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    task_id,
                    int(rnd.get("round", 0)),
                    str(rnd.get("question_id", "")),
                    str(rnd.get("question", "")),
                    str(rnd.get("answer", "") or ""),
                    str(rnd.get("mode", "")),
                    time.time(),
                ),
            )


def update_status(task_id: str, status: str, path: Path | None = None) -> None:
    with _connect(path) as conn:
        conn.execute(
            "UPDATE sessions SET status = ? WHERE task_id = ?", (status, task_id)
        )


def get_session(task_id: str, path: Path | None = None) -> dict[str, Any] | None:
    """查一个会话：会话行 + 账本 + 全部拷打记录。查不到返回 None。"""
    db = default_db_path() if path is None else Path(path)
    if not db.exists():
        return None
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM sessions WHERE task_id = ?", (task_id,)
        ).fetchone()
        if row is None:
            return None
        claims = [
            dict(r)
            for r in conn.execute(
                "SELECT claim_id, claim_text, origin, risk_flags_json FROM claims"
                " WHERE task_id = ? ORDER BY claim_id",
                (task_id,),
            )
        ]
        records = [
            dict(r)
            for r in conn.execute(
                "SELECT round_no, question_id, question, answer, mode FROM dig_records"
                " WHERE task_id = ? ORDER BY id",
                (task_id,),
            )
        ]
    return {"session": dict(row), "claims": claims, "dig_records": records}
