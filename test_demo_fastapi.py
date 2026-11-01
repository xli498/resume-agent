"""Demo 层测试：FastAPI 端点 + SQLite 落盘 + Agent 多轮循环。

无 fastapi 的环境自动跳过（老套件在系统 python 下仍可全跑）。
测试数据均为虚构（与 test_web_app.py 同款），不含真实简历。
"""
import json
import os
import tempfile
import unittest

import web_app

try:
    from fastapi.testclient import TestClient

    import api_fastapi

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

RESUME = "张三\n工作经历\n某公司\n主导数据分析项目，2024年交付"
JD = "岗位要求：数据分析"
ANSWER = "主导某数据分析项目，2024年交付使用，覆盖3个业务部门"


@unittest.skipUnless(HAS_FASTAPI, "fastapi 未安装，跳过 Demo 层测试")
class DemoFastAPITests(unittest.TestCase):
    def setUp(self):
        with web_app.TASKS_LOCK:
            web_app.TASKS.clear()
        self._tmp = tempfile.TemporaryDirectory()
        self._db = os.path.join(self._tmp.name, "demo.db")
        self._old_db = os.environ.get("RESUME_AGENT_DB")
        os.environ["RESUME_AGENT_DB"] = self._db
        # R4：Agent 日志改走临时路径，不再写仓库 output/agent_dig.jsonl
        self._log = os.path.join(self._tmp.name, "agent_dig.jsonl")
        self._old_log = os.environ.get("RESUME_AGENT_DIG_LOG")
        os.environ["RESUME_AGENT_DIG_LOG"] = self._log
        # 强制 Agent 走规则回退：卸掉可能存在的 LLM 环境变量
        self._old_llm_env = {}
        for key in ("API_BASE_URL", "API_KEY", "MODEL"):
            if key in os.environ:
                self._old_llm_env[key] = os.environ.pop(key)
        self.client = TestClient(api_fastapi.app)
        status, created = self._post("/api/analyze", {"resume": RESUME, "jd": JD})
        self.assertEqual(status, 200, created)
        self.task_id = created["task_id"]
        self.created = created

    def tearDown(self):
        if self._old_db is None:
            os.environ.pop("RESUME_AGENT_DB", None)
        else:
            os.environ["RESUME_AGENT_DB"] = self._old_db
        if self._old_log is None:
            os.environ.pop("RESUME_AGENT_DIG_LOG", None)
        else:
            os.environ["RESUME_AGENT_DIG_LOG"] = self._old_log
        os.environ.update(self._old_llm_env)
        self._tmp.cleanup()
        with web_app.TASKS_LOCK:
            web_app.TASKS.clear()

    def _post(self, path, payload, content_type="application/json"):
        response = self.client.post(
            path,
            content=json.dumps(payload, ensure_ascii=False),
            headers={"Content-Type": content_type},
        )
        try:
            data = response.json()
        except json.JSONDecodeError:
            data = None
        return response.status_code, data

    def test_health_ok(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_analyze_shape_matches_web_app(self):
        created = self.created
        for field in (
            "task_id",
            "status",
            "analysis",
            "fact_ledger",
            "interview_defense",
            "dig_questions",
            "match_coverage",
            "ats_checks",
            "ats_score",
        ):
            self.assertIn(field, created, field)

    def test_task_query_strips_source_material(self):
        response = self.client.get(f"/api/tasks/{self.task_id}")
        self.assertEqual(response.status_code, 200)
        task = response.json()
        self.assertNotIn("resume", task)
        self.assertNotIn("jd", task)
        self.assertNotIn("created_at", task)

    def test_content_type_and_object_required(self):
        self.assertEqual(
            self._post("/api/analyze", {"resume": "a", "jd": "b"}, "text/plain")[0], 400
        )
        self.assertEqual(self._post("/api/analyze", ["bad"])[0], 400)

    def test_store_has_session_and_claims_after_analyze(self):
        response = self.client.get(f"/api/store/{self.task_id}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["session"]["task_id"], self.task_id)
        self.assertTrue(data["claims"], "账本落盘为空")

    def test_dig_and_generate_single_transition(self):
        first_q = self.created["dig_questions"][0]
        status, dig = self._post(
            "/api/dig",
            {
                "task_id": self.task_id,
                "answers": [
                    {
                        "question_id": first_q["question_id"],
                        "claim_id": first_q["claim_id"],
                        "answer": ANSWER,
                    }
                ],
            },
        )
        self.assertEqual(status, 200, dig)
        self.assertEqual(self._post("/api/generate", {"task_id": self.task_id})[0], 200)
        self.assertEqual(self._post("/api/generate", {"task_id": self.task_id})[0], 409)

    @unittest.skipUnless(
        __import__("pathlib").Path("/opt/demo/resume/fonts").is_dir(),
        "PDF 渲染依赖字体目录 /opt/demo/resume/fonts（未补齐时跳过；"
        "补齐办法见 scripts/setup_local_fonts.py）",
    )
    def test_pdf_after_generate(self):
        self._post("/api/generate", {"task_id": self.task_id})
        response = self.client.get(f"/api/pdf?task_id={self.task_id}")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("application/pdf"))
        self.assertGreater(len(response.content), 1000)

    def test_pdf_unknown_task_returns_404(self):
        self.assertEqual(self.client.get("/api/pdf?task_id=nope").status_code, 404)
        self.assertEqual(
            self._post("/api/pdf", {"task_id": "nope"})[0], 404
        )

    def test_agent_dig_rule_fallback_rounds_are_observable(self):
        status, result = self._post(
            "/api/agent_dig", {"task_id": self.task_id, "max_rounds": 2}
        )
        self.assertEqual(status, 200, result)
        self.assertFalse(result["llm_used"], "无 LLM 环境变量时 llm_used 必须为 False")
        self.assertTrue(result["mode"].startswith("rule_fallback"))
        self.assertTrue(result["rounds"], "Agent 轮次为空")
        for rnd in result["rounds"]:
            for field in ("round", "thought", "action", "observation", "mode"):
                self.assertIn(field, rnd, field)
        # 轮次已落盘可查
        response = self.client.get(f"/api/store/{self.task_id}")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["dig_records"], "Agent 轮次未落盘")

    def test_agent_dig_with_answer_adds_claim(self):
        first_q = self.created["dig_questions"][0]
        status, result = self._post(
            "/api/agent_dig",
            {
                "task_id": self.task_id,
                "max_rounds": 3,
                "answers": [
                    {
                        "question_id": first_q["question_id"],
                        "claim_id": first_q["claim_id"],
                        "answer": ANSWER,
                    }
                ],
            },
        )
        self.assertEqual(status, 200, result)
        self.assertTrue(
            any(rnd["added"] >= 1 for rnd in result["rounds"]),
            f"提供回答后应新增 Claim，rounds={result['rounds']}",
        )

    # ---- 本轮修复的新测试（R1/R2/R3/R5/R4/risk_flags/连续流程）----

    def test_body_over_2mb_returns_400(self):
        big = json.dumps({"resume": "x" * 2_000_001, "jd": "y"}, ensure_ascii=False)
        response = self.client.post(
            "/api/analyze",
            content=big.encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(response.status_code, 400)

    def test_body_exactly_2mb_passes_size_check(self):
        # 注意：不用 /api/analyze 测边界——2MB 简历会跑完整工作流，
        # 其中 ATS 规则引擎对超大单 token 文本极慢（既有核心行为，禁动）。
        # 改走 /api/pdf：只验中间件放行（404 = 通过了尺寸校验、任务不存在）。
        base = json.dumps({"task_id": "nope", "pad": ""}, ensure_ascii=False)
        pad = 2_000_000 - len(base.encode("utf-8"))
        payload = json.dumps({"task_id": "nope", "pad": "x" * pad}, ensure_ascii=False)
        self.assertEqual(len(payload.encode("utf-8")), 2_000_000)
        response = self.client.post(
            "/api/pdf",
            content=payload.encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(response.status_code, 404)

    def test_malformed_json_returns_400_not_422(self):
        response = self.client.post(
            "/api/analyze",
            content=b"{not json",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(response.status_code, 400)

    def test_store_strips_original_text_and_structures_flags(self):
        first_q = self.created["dig_questions"][0]
        self._post(
            "/api/dig",
            {
                "task_id": self.task_id,
                "answers": [
                    {
                        "question_id": first_q["question_id"],
                        "claim_id": first_q["claim_id"],
                        "answer": ANSWER,
                    }
                ],
            },
        )
        response = self.client.get(f"/api/store/{self.task_id}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        for claim in data["claims"]:
            self.assertNotIn("claim_text", claim)
            self.assertNotIn("risk_flags_json", claim)
            self.assertIsInstance(claim["risk_flags"], list)
        for rec in data["dig_records"]:
            self.assertNotIn("question", rec)
            self.assertNotIn("answer", rec)
            self.assertIn("answered", rec)
            self.assertIn("added", rec)

    def test_dig_records_keep_full_text_in_db(self):
        first_q = self.created["dig_questions"][0]
        self._post(
            "/api/dig",
            {
                "task_id": self.task_id,
                "answers": [
                    {
                        "question_id": first_q["question_id"],
                        "claim_id": first_q["claim_id"],
                        "answer": ANSWER,
                    }
                ],
            },
        )
        import sqlite3

        with sqlite3.connect(self._db) as conn:
            rows = conn.execute(
                "SELECT question, answer FROM dig_records"
                " WHERE task_id = ? AND mode = 'user_answer'",
                (self.task_id,),
            ).fetchall()
        self.assertTrue(rows, "库内无深挖记录")
        self.assertEqual(rows[0][0], first_q["question"])
        self.assertEqual(rows[0][1], ANSWER)

    def test_agent_dig_ignored_and_stopped_reason(self):
        status, result = self._post(
            "/api/agent_dig",
            {
                "task_id": self.task_id,
                "max_rounds": 5,
                "answers": [
                    {"question_id": "Q999", "claim_id": "C001", "answer": ANSWER}
                ],
            },
        )
        self.assertEqual(status, 200, result)
        self.assertTrue(
            any(item["question_id"] == "Q999" for item in result["ignored"]),
            f"未匹配回答必须明列 ignored，got={result.get('ignored')}",
        )
        self.assertIn(
            result["stopped_reason"],
            ("question_pool_exhausted", "max_rounds_reached"),
        )

    def test_mixed_pool_outside_ids_rejected_as_batch(self):
        first_q = self.created["dig_questions"][0]
        status, result = self._post(
            "/api/dig",
            {
                "task_id": self.task_id,
                "answers": [
                    {
                        "question_id": first_q["question_id"],
                        "claim_id": first_q["claim_id"],
                        "answer": ANSWER,
                    },
                    {"question_id": "Q999", "answer": ANSWER},
                ],
            },
        )
        self.assertEqual(status, 400, result)
        self.assertIn("Q999", json.dumps(result, ensure_ascii=False))
        import sqlite3

        with sqlite3.connect(self._db) as conn:
            rows = conn.execute(
                "SELECT question, answer FROM dig_records"
                " WHERE task_id = ? AND mode = 'user_answer'",
                (self.task_id,),
            ).fetchall()
        self.assertEqual(rows, [], "整批拒绝时不得落盘任何记录")

    def test_continuous_analyze_dig_keeps_claims(self):
        before = self.client.get(f"/api/store/{self.task_id}").json()
        ids_before = {c["claim_id"] for c in before["claims"]}
        first_q = self.created["dig_questions"][0]
        status, dig = self._post(
            "/api/dig",
            {
                "task_id": self.task_id,
                "answers": [
                    {
                        "question_id": first_q["question_id"],
                        "claim_id": first_q["claim_id"],
                        "answer": ANSWER,
                    }
                ],
            },
        )
        self.assertEqual(status, 200, dig)
        after = self.client.get(f"/api/store/{self.task_id}").json()
        ids_after = {c["claim_id"] for c in after["claims"]}
        self.assertTrue(ids_before <= ids_after, f"深挖丢了原有 Claim：{ids_before - ids_after}")
        self.assertGreater(len(ids_after), len(ids_before))

    def test_agent_log_goes_to_env_path_not_repo(self):
        import pathlib

        repo_log = pathlib.Path(__file__).parent / "output" / "agent_dig.jsonl"
        before = repo_log.read_bytes() if repo_log.exists() else None
        status, _ = self._post("/api/agent_dig", {"task_id": self.task_id, "max_rounds": 1})
        self.assertEqual(status, 200)
        with open(self._log, encoding="utf-8") as f:
            self.assertIn(self.task_id, f.read())
        after = repo_log.read_bytes() if repo_log.exists() else None
        self.assertEqual(before, after, "测试写了仓库 output/agent_dig.jsonl")

    def _dig_record_count(self, task_id):
        import sqlite3

        with sqlite3.connect(self._db) as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM dig_records WHERE task_id = ?", (task_id,)
            ).fetchone()[0]

    def _store_claim_ids(self, task_id):
        store = self.client.get(f"/api/store/{task_id}").json()
        return {c["claim_id"] for c in store["claims"]}

    def test_dig_rejects_pool_outside_id(self):
        records_before = self._dig_record_count(self.task_id)
        claims_before = self._store_claim_ids(self.task_id)
        status, result = self._post(
            "/api/dig",
            {
                "task_id": self.task_id,
                "answers": [
                    {"question_id": "Q999", "claim_id": "C001", "answer": ANSWER}
                ],
            },
        )
        self.assertEqual(status, 400, result)
        self.assertIn("Q999", result.get("error", ""))
        self.assertIn("问题池", result.get("error", ""))
        self.assertEqual(
            self._dig_record_count(self.task_id), records_before, "池外 ID 不得入库"
        )
        self.assertEqual(
            self._store_claim_ids(self.task_id), claims_before, "池外 ID 不得生成 Claim"
        )

    def test_dig_pool_inside_id_persists(self):
        first_q = self.created["dig_questions"][0]
        records_before = self._dig_record_count(self.task_id)
        status, dig = self._post(
            "/api/dig",
            {
                "task_id": self.task_id,
                "answers": [
                    {
                        "question_id": first_q["question_id"],
                        "claim_id": first_q["claim_id"],
                        "answer": ANSWER,
                    }
                ],
            },
        )
        self.assertEqual(status, 200, dig)
        self.assertEqual(
            self._dig_record_count(self.task_id), records_before + 1, "池内 ID 应正常入库"
        )

    def test_dig_empty_pool_rejects_any_id(self):
        status, created = self._post("/api/analyze", {"resume": "你好", "jd": "无"})
        self.assertEqual(status, 200, created)
        self.assertEqual(created["dig_questions"], [])
        status, result = self._post(
            "/api/dig",
            {
                "task_id": created["task_id"],
                "answers": [{"question_id": "Q1", "answer": ANSWER}],
            },
        )
        self.assertEqual(status, 400, result)
        self.assertIn("问题池", result.get("error", ""))


if __name__ == "__main__":
    unittest.main()
