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
        "PDF 渲染依赖服务器字体目录 /opt/demo/resume/fonts（本机无此目录，"
        "且任务禁动 /opt/demo，故跳过；端点接线由 404 测试覆盖）",
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


if __name__ == "__main__":
    unittest.main()
