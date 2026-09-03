import json
import threading
import unittest
from http.client import HTTPConnection

import web_app


class WebAppTests(unittest.TestCase):
    def setUp(self):
        with web_app.TASKS_LOCK:
            web_app.TASKS.clear()
        self.server = web_app.ThreadingHTTPServer(("127.0.0.1", 0), web_app.Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, method, path, payload=None, content_type="application/json"):
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {} if body is None else {"Content-Type": content_type}
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        data = json.loads(response.read().decode("utf-8")) if response.getheader("Content-Type", "").startswith("application/json") else None
        connection.close()
        return response.status, data

    def test_non_loopback_bind_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "仅允许绑定"):
            web_app._require_loopback("0.0.0.0")
        web_app._require_loopback("127.0.0.1")
        web_app._require_loopback("::1")

    def test_task_query_does_not_expose_source_material(self):
        status, created = self.request("POST", "/api/analyze", {"resume": "张三\n工作经历\n某公司", "jd": "岗位要求：数据分析"})
        self.assertEqual(status, 200)
        status, task = self.request("GET", f"/api/tasks/{created['task_id']}")
        self.assertEqual(status, 200)
        self.assertNotIn("resume", task)
        self.assertNotIn("jd", task)
        self.assertNotIn("created_at", task)

    def test_generate_is_single_transition(self):
        status, created = self.request("POST", "/api/analyze", {"resume": "张三\n工作经历\n某公司", "jd": "岗位要求：数据分析"})
        self.assertEqual(status, 200)
        payload = {"task_id": created["task_id"]}
        self.assertEqual(self.request("POST", "/api/generate", payload)[0], 200)
        self.assertEqual(self.request("POST", "/api/generate", payload)[0], 409)

    def test_json_object_and_content_type_are_required(self):
        self.assertEqual(self.request("POST", "/api/analyze", ["bad"])[0], 400)
        self.assertEqual(self.request("POST", "/api/analyze", {"resume": "a", "jd": "b"}, "text/plain")[0], 400)


if __name__ == "__main__":
    unittest.main()
