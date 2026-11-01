"""质疑轮测试：默认关闭不改变旧行为；开启后对仍有覆盖缺口的高风险 Claim 出质疑轮。

只测 agent_dig.run_agent_dig（不走 HTTP），LLM 环境变量卸掉以强制规则回退，
日志走临时 RESUME_AGENT_DIG_LOG，不写仓库 output/agent_dig.jsonl。
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, "/home/hatch/workspace/repos/resume-agent")

import agent_dig

# 仅含年份数字的 Claim：高风险（有 numbers）但防御零问题 -> 覆盖缺口，质疑靶子。
YEAR_ONLY = {"claim_id": "C001", "text": "主导数据分析项目，2024年交付",
             "numbers": ["2024"], "role_level": 1, "role_words": [],
             "strong_words": []}
# 非年份数字 -> 防御有问题 -> 已覆盖，质疑应跳过。
COVERED = {"claim_id": "C002", "text": "覆盖 3 个业务部门",
           "numbers": ["3"], "role_level": 1, "role_words": [],
           "strong_words": []}


class ChallengeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old_log = os.environ.get("RESUME_AGENT_DIG_LOG")
        os.environ["RESUME_AGENT_DIG_LOG"] = os.path.join(
            self._tmp.name, "agent_dig.jsonl")
        self._old_llm_env = {}
        for key in ("API_BASE_URL", "API_KEY", "MODEL"):
            if key in os.environ:
                self._old_llm_env[key] = os.environ.pop(key)

    def tearDown(self):
        if self._old_log is None:
            os.environ.pop("RESUME_AGENT_DIG_LOG", None)
        else:
            os.environ["RESUME_AGENT_DIG_LOG"] = self._old_log
        os.environ.update(self._old_llm_env)
        self._tmp.cleanup()

    def test_default_off_changes_nothing(self):
        result = agent_dig.run_agent_dig("简历", [YEAR_ONLY], max_rounds=2)
        self.assertEqual(result["challenge"], {"rounds": [], "challenged": 0})
        self.assertIn("rounds", result)
        self.assertIn("coverage_errors", result)

    def test_challenge_targets_uncovered_claim(self):
        result = agent_dig.run_agent_dig(
            "简历", [YEAR_ONLY], max_rounds=2, challenge=True)
        self.assertEqual(result["challenge"]["challenged"], 1)
        rnd = result["challenge"]["rounds"][0]
        for field in ("round", "thought", "action", "observation", "mode",
                      "llm_used", "question_id", "question", "answer",
                      "answered", "added", "claim_id"):
            self.assertIn(field, rnd, field)
        self.assertEqual(rnd["action"], "challenge")
        self.assertEqual(rnd["claim_id"], "C001")
        self.assertTrue(rnd["question"], "质疑文本为空")
        self.assertFalse(rnd["llm_used"], "无 LLM 环境时必须走规则回退")
        self.assertTrue(rnd["mode"].startswith("rule_fallback"))
        # 截断标注：短文本（13 字）未超规则回退 60 字阈值。
        self.assertIn("input_truncated", rnd)
        self.assertFalse(rnd["input_truncated"])
        # 主 rounds 不受影响：仍有覆盖缺口记录。
        self.assertTrue(any("C001" in e for e in result["coverage_errors"]))

    def test_challenge_input_truncated_flagged(self):
        long_claim = {"claim_id": "C003",
                      "text": "主导数据分析项目，2024年交付，覆盖多个业务部门且指标提升显著" * 2,
                      "numbers": ["2024"], "role_level": 1, "role_words": [],
                      "strong_words": []}
        result = agent_dig.run_agent_dig(
            "简历", [long_claim], max_rounds=2, challenge=True)
        self.assertEqual(result["challenge"]["challenged"], 1)
        self.assertTrue(result["challenge"]["rounds"][0]["input_truncated"])

    def test_challenge_skipped_when_covered(self):
        result = agent_dig.run_agent_dig(
            "简历", [COVERED], max_rounds=2, challenge=True)
        self.assertEqual(result["challenge"]["challenged"], 0)
        self.assertEqual(result["challenge"]["rounds"], [])


try:
    from fastapi.testclient import TestClient

    import web_app

    import api_fastapi
    import store

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


@unittest.skipUnless(HAS_FASTAPI, "fastapi 未安装，跳过端点落盘测试")
class ChallengePersistTests(unittest.TestCase):
    """端点级：challenge=true 时质疑轮必须进 store 落盘（P1-3），
    而不只存在响应里。直接注入 TASKS 任务（绕开 analyze 的 Claim 提取），
    用仅含年份数字的 Claim 强制造出覆盖缺口。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old_db = os.environ.get("RESUME_AGENT_DB")
        os.environ["RESUME_AGENT_DB"] = os.path.join(self._tmp.name, "demo.db")
        self._old_log = os.environ.get("RESUME_AGENT_DIG_LOG")
        os.environ["RESUME_AGENT_DIG_LOG"] = os.path.join(
            self._tmp.name, "agent_dig.jsonl")
        self._old_llm_env = {}
        for key in ("API_BASE_URL", "API_KEY", "MODEL"):
            if key in os.environ:
                self._old_llm_env[key] = os.environ.pop(key)
        import time as _time
        with web_app.TASKS_LOCK:
            web_app.TASKS.clear()
            web_app.TASKS["t-challenge"] = {
                "task_id": "t-challenge", "status": "awaiting_confirmation",
                "created_at": _time.monotonic(), "resume": "简历",
                "claims": [dict(YEAR_ONLY)],
            }
        # 真实流程里任务总是由 /api/analyze 建会话行；直接注入 TASKS 时
        # 需补这一步，否则 get_session 因无会话行返回 None（与落盘逻辑无关）。
        store.save_analysis("t-challenge", [dict(YEAR_ONLY)])
        self.client = TestClient(api_fastapi.app)

    def tearDown(self):
        with web_app.TASKS_LOCK:
            web_app.TASKS.clear()
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

    def test_endpoint_persists_challenge_rounds(self):
        r = self.client.post(
            "/api/agent_dig",
            json={"task_id": "t-challenge", "max_rounds": 2, "challenge": True})
        self.assertEqual(r.status_code, 200, r.text)
        j = r.json()
        self.assertEqual(j["challenge"]["challenged"], 1)
        session = store.get_session("t-challenge")
        self.assertIsNotNone(session)
        challenge_rows = [
            rec for rec in session["dig_records"]
            if rec["mode"].startswith("challenge:")
        ]
        self.assertEqual(len(challenge_rows), 1)
        self.assertTrue(challenge_rows[0]["question"], "质疑文本未落盘")
        self.assertEqual(
            challenge_rows[0]["question"],
            j["challenge"]["rounds"][0]["question"])


if __name__ == "__main__":
    unittest.main()
