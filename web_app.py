"""Resume Agent 本地 MVP Web 入口。

只使用 Python 标准库，避免第一阶段引入前端构建链：
- 粘贴简历与 JD
- 分析并展示事实账本、匹配结论、待确认问题
- 用户确认后生成基于证据的简历
- 提供 Markdown 下载

这是单用户本地工作台，不是公网部署版本；任务状态保存在进程内存中。
"""
from __future__ import annotations

import json
import ipaddress
import secrets
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from workflow import run_python_workflow


TASKS: dict[str, dict] = {}
TASKS_LOCK = threading.Lock()
TASK_TTL_SECONDS = 60 * 60
MAX_TASKS = 32


INDEX_HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Resume Agent · 可信定向简历工作台</title>
<style>
:root{--ink:#172033;--muted:#667085;--line:#e6e8ee;--bg:#f6f7fb;--brand:#315efb;--brand2:#6c4df6;--ok:#087443;--warn:#a15c00}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 Inter,-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif}.shell{max-width:1180px;margin:auto;padding:38px 22px 72px}.hero{display:flex;justify-content:space-between;align-items:end;gap:20px;margin-bottom:28px}.eyebrow{color:var(--brand);font-weight:700;letter-spacing:.08em;font-size:12px}.hero h1{font-size:34px;line-height:1.15;margin:8px 0}.hero p{color:var(--muted);margin:0}.badge{background:#e9edff;color:#294bd2;border-radius:999px;padding:7px 12px;font-weight:700;white-space:nowrap}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.card{background:#fff;border:1px solid var(--line);border-radius:18px;padding:22px;box-shadow:0 8px 28px #27315c0b}.card h2{font-size:18px;margin:0 0 14px}.wide{grid-column:1/-1}label{display:block;font-weight:700;margin:12px 0 6px}textarea,input{width:100%;border:1px solid #d9dce5;border-radius:11px;padding:12px;font:inherit;color:var(--ink);background:#fcfcfe}textarea{min-height:220px;resize:vertical}input{height:44px}button{border:0;border-radius:11px;padding:11px 17px;font:inherit;font-weight:700;cursor:pointer;background:var(--brand);color:#fff}button.secondary{background:#eef1f7;color:#26324a}button:disabled{opacity:.5;cursor:not-allowed}.actions{display:flex;gap:10px;align-items:center;margin-top:16px}.hint{color:var(--muted);font-size:13px}.hidden{display:none}.status{font-weight:700;margin-left:6px}.status.ok{color:var(--ok)}.status.warn{color:var(--warn)}.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:9px}.fact{border:1px solid var(--line);border-radius:10px;padding:10px}.fact b{color:var(--brand);font-size:12px}.fact small{color:var(--muted)}.question{padding:10px 12px;border-left:3px solid #f0a33a;background:#fff8ec;margin:8px 0;border-radius:5px}.result{white-space:pre-wrap;background:#111827;color:#e8edf7;border-radius:12px;padding:16px;max-height:620px;overflow:auto;font:13px/1.65 ui-monospace,SFMono-Regular,Menlo,monospace}.metric{display:inline-flex;gap:7px;background:#f3f5fa;border-radius:9px;padding:7px 10px;margin:0 7px 7px 0;font-size:13px}.metric b{color:var(--brand)}@media(max-width:760px){.grid{grid-template-columns:1fr}.wide{grid-column:auto}.hero{display:block}.badge{display:inline-block;margin-top:14px}.hero h1{font-size:28px}}
</style></head><body><main class="shell"><header class="hero"><div><div class="eyebrow">EVIDENCE-AWARE RESUME AGENT</div><h1>把岗位要求，变成可信的定向简历</h1><p>先分析证据，再让你确认；未确认的事实不会进入最终稿。</p></div><span class="badge">本地 MVP · 不上传材料</span></header>
<section class="grid"><div class="card"><h2>① 输入材料</h2><label for="resume">基础简历</label><textarea id="resume" placeholder="粘贴简历文本，建议保留模块标题"></textarea><label for="jd">岗位 JD</label><textarea id="jd" placeholder="粘贴岗位描述"></textarea><label for="title">求职方向（可选）</label><input id="title" placeholder="例如：大模型应用研发实习生"><div class="actions"><button id="analyze">开始分析</button><span id="status" class="status"></span></div><p class="hint">材料只在当前本地进程中处理；真实模型调用尚未接入此页面。</p></div>
<div id="analysisCard" class="card hidden"><h2>② 证据与匹配</h2><div id="metrics"></div><h3>事实账本</h3><div id="facts" class="facts"></div><h3>待确认问题</h3><div id="questions"></div><div class="actions"><button id="generate" disabled>确认事实并生成</button><span class="hint">当前按钮表示确认已有事实；未确认问题不会被模型擅自补写。</span></div></div>
<div id="resultCard" class="card wide hidden"><h2>③ 结果与交付</h2><div id="resultMeta"></div><pre id="result" class="result"></pre><div class="actions"><button id="download" class="secondary">下载 Markdown</button><span class="hint">下一步将加入 PDF/PNG 交付包、版本历史和 DOCX/PDF 导入。</span></div></div></section></main>
<script>
let taskId='',finalResume='';const $=id=>document.getElementById(id);function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
$('analyze').onclick=async()=>{const resume=$('resume').value.trim(),jd=$('jd').value.trim();if(!resume||!jd){$('status').textContent='请先填写简历和 JD';$('status').className='status warn';return}$('analyze').disabled=true;$('status').textContent='分析中…';try{const r=await fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({resume,jd,job_title:$('title').value.trim()})});const d=await r.json();if(!r.ok)throw Error(d.error||'分析失败');taskId=d.task_id;const a=d.analysis;$('metrics').innerHTML=`<span class="metric">岗位要求 <b>${a.requirements.length}</b></span><span class="metric">事实 <b>${d.fact_ledger.length}</b></span><span class="metric">待确认 <b>${a.confirmation_questions.length}</b></span>`;$('facts').innerHTML=d.fact_ledger.map(f=>`<div class="fact"><b>${esc(f.fact_id)} · ${esc(f.section)}</b><br>${esc(f.text)}</div>`).join('')||'<span class="hint">未解析到事实</span>';$('questions').innerHTML=a.confirmation_questions.map(q=>`<div class="question"><b>${esc(q.category||'待确认')}</b>：${esc(q.question||q.text||'')}</div>`).join('')||'<span class="status ok">当前没有待确认问题</span>';$('analysisCard').classList.remove('hidden');$('generate').disabled=false;$('status').textContent='分析完成';$('status').className='status ok';$('analysisCard').scrollIntoView({behavior:'smooth',block:'start'})}catch(e){$('status').textContent=e.message;$('status').className='status warn'}finally{$('analyze').disabled=false}};
$('generate').onclick=async()=>{if(!taskId)return;$('generate').disabled=true;$('status').textContent='生成与真实性校验中…';try{const r=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task_id:taskId})});const d=await r.json();if(!r.ok)throw Error(d.error||'生成失败');finalResume=d.final_resume;$('result').textContent=finalResume;$('resultMeta').innerHTML='<span class="metric">状态 <b>QA 前置校验通过</b></span><span class="metric">任务 <b>'+esc(taskId)+'</b></span>';$('resultCard').classList.remove('hidden');$('status').textContent='生成完成';$('status').className='status ok';$('resultCard').scrollIntoView({behavior:'smooth',block:'start'})}catch(e){$('status').textContent=e.message;$('status').className='status warn'}finally{$('generate').disabled=false}};
$('download').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([finalResume],{type:'text/markdown;charset=utf-8'}));a.download='targeted-resume.md';a.click();URL.revokeObjectURL(a.href)};
</script></body></html>'''


def _json(handler: BaseHTTPRequestHandler, payload: dict, status=HTTPStatus.OK) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("X-Content-Type-Options", "nosniff")
    handler.end_headers()
    handler.wfile.write(body)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A003
        return

    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path
        if path == "/":
            body = INDEX_HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()
            self.wfile.write(body)
            return
        if path.startswith("/api/tasks/"):
            task_id = path.rsplit("/", 1)[-1]
            with TASKS_LOCK:
                _prune_tasks_locked()
                task = TASKS.get(task_id)
            if not task:
                _json(self, {"error": "任务不存在"}, HTTPStatus.NOT_FOUND)
            else:
                _json(self, _public_task(task))
            return
        _json(self, {"error": "找不到页面"}, HTTPStatus.NOT_FOUND)

    def _body(self):
        if self.headers.get_content_type() != "application/json":
            raise ValueError("请求必须使用 application/json")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("Content-Length 无效") from error
        if length <= 0:
            raise ValueError("请求体不能为空")
        if length > 2_000_000:
            raise ValueError("请求体过大")
        data = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("请求体必须是 JSON 对象")
        return data

    def do_POST(self):  # noqa: N802
        try:
            data = self._body()
            path = urlparse(self.path).path
            if path == "/api/analyze":
                resume, jd = str(data.get("resume", "")).strip(), str(data.get("jd", "")).strip()
                if not resume or not jd:
                    raise ValueError("简历和 JD 不能为空")
                result = run_python_workflow(resume, jd, job_title=data.get("job_title") or None)
                task_id = secrets.token_urlsafe(10)
                task = {"task_id": task_id, "status": "awaiting_confirmation", "created_at": time.monotonic(), "resume": resume, "jd": jd, "job_title": data.get("job_title") or None, **result}
                with TASKS_LOCK:
                    _prune_tasks_locked()
                    if len(TASKS) >= MAX_TASKS:
                        oldest = min(TASKS, key=lambda key: TASKS[key]["created_at"])
                        TASKS.pop(oldest, None)
                    TASKS[task_id] = task
                _json(self, {"task_id": task_id, "status": task["status"], "analysis": result["analysis"], "fact_ledger": result["fact_ledger"]})
                return
            if path == "/api/generate":
                task_id = str(data.get("task_id", ""))
                with TASKS_LOCK:
                    _prune_tasks_locked()
                    task = TASKS.get(task_id)
                    if not task:
                        response = None
                    elif task["status"] != "awaiting_confirmation":
                        response = {"error": "任务状态不允许生成"}
                    else:
                        task["status"] = "generated"
                        response = {"task_id": task_id, "status": task["status"], "final_resume": task["final_resume"]}
                if response is None:
                    _json(self, {"error": "任务不存在"}, HTTPStatus.NOT_FOUND); return
                if "error" in response:
                    _json(self, response, HTTPStatus.CONFLICT); return
                _json(self, response)
                return
            _json(self, {"error": "找不到接口"}, HTTPStatus.NOT_FOUND)
        except (ValueError, json.JSONDecodeError) as error:
            _json(self, {"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception:
            _json(self, {"error": "处理失败，请检查输入或查看本地终端"}, HTTPStatus.INTERNAL_SERVER_ERROR)


def _prune_tasks_locked(now: float | None = None) -> None:
    now = time.monotonic() if now is None else now
    expired = [key for key, task in TASKS.items() if now - task["created_at"] > TASK_TTL_SECONDS]
    for key in expired:
        TASKS.pop(key, None)


def _public_task(task: dict) -> dict:
    return {key: value for key, value in task.items() if key not in {"resume", "jd", "created_at"}}


def _require_loopback(host: str) -> None:
    try:
        if not ipaddress.ip_address(host).is_loopback:
            raise ValueError
    except ValueError as error:
        raise ValueError("本地 MVP 仅允许绑定 127.0.0.1 或 ::1") from error


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    _require_loopback(host)
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Resume Agent MVP 已启动：http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="启动 Resume Agent 本地 MVP Web 工作台")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    serve(args.host, args.port)
