"""Resume Agent Web 入口（Arco 设计语言）。

- 粘贴简历与 JD
- 分析并展示事实账本、Claim Ledger、匹配结论、待确认问题、面试防御、深挖
- 用户确认后生成基于证据的简历
- 提供 Markdown 下载
"""
from __future__ import annotations

import json
import secrets
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from workflow import apply_dig_answers, run_python_workflow
from pdf_export import render_pdf


TASKS: dict[str, dict] = {}
TASKS_LOCK = threading.Lock()
TASK_TTL_SECONDS = 60 * 60
MAX_TASKS = 32


INDEX_HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Resume Agent · 可信定向简历工作台</title>
<style>
:root{--brand:#165DFF;--brand-hover:#4080FF;--brand-active:#0E42D2;--brand-light:#E8F3FF;--brand-border:#BEDAFF;--title:#1D2129;--body:#4E5969;--secondary:#86909C;--border:#E5E6EB;--fill1:#F7F8FA;--fill2:#F2F3F5;--ok:#00B42A;--warn:#FF7D00;--warn-bg:#FFF7E8;--warn-bd:#FFD591;--err:#F53F3F}
*{box-sizing:border-box}html{color-scheme:light}body{margin:0;background:var(--fill2);color:var(--title);font:14px/1.7 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;-webkit-font-smoothing:antialiased}
.shell{max-width:1200px;margin:auto;padding:0 20px 64px}
.hero{background:#fff;border:1px solid var(--border);border-radius:4px;padding:28px 32px;margin:20px 0 16px}
.hero .eyebrow{display:inline-block;background:var(--brand-light);color:var(--brand);border-radius:2px;padding:3px 10px;font-size:12px;font-weight:600;letter-spacing:.04em;margin-bottom:10px}
.hero h1{font-size:24px;line-height:1.4;margin:0 0 8px;font-weight:600}
.hero h1 .accent{color:var(--brand)}
.hero p{color:var(--body);margin:0;max-width:680px;font-size:14px}
.steps{display:flex;gap:26px;margin-top:18px;flex-wrap:wrap}
.step{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--body)}
.step .n{width:22px;height:22px;border-radius:50%;background:var(--brand);color:#fff;font-size:12px;font-weight:600;display:flex;align-items:center;justify-content:center}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.card{background:#fff;border:1px solid var(--border);border-radius:4px;padding:24px}
.card h2{font-size:16px;margin:0 0 4px;display:flex;align-items:center;gap:10px;font-weight:600}
.card h2 .n{width:24px;height:24px;border-radius:50%;background:var(--brand);color:#fff;font-size:13px;font-weight:600;display:inline-flex;align-items:center;justify-content:center;flex:none}
.card h3{font-size:14px;margin:20px 0 10px;color:var(--title);font-weight:600}
.card .desc{color:var(--body);font-size:13px;margin:0 0 6px}
.wide{grid-column:1/-1}
label.f-label{display:block;font-weight:600;margin:14px 0 6px;font-size:13px}
textarea,input[type=text]{width:100%;border:1px solid var(--border);border-radius:2px;padding:10px 12px;font:inherit;color:var(--title);background:#fff;transition:border-color .15s,box-shadow .15s}
textarea{min-height:170px;resize:vertical;line-height:1.7}
textarea:focus,input[type=text]:focus{outline:none;border-color:var(--brand);box-shadow:0 0 0 2px rgba(22,93,255,.12)}
input[type=text]{height:38px}
.finance-row{display:flex;align-items:center;gap:10px;margin:14px 0 4px;background:var(--fill1);border:1px solid var(--border);border-radius:2px;padding:10px 14px;cursor:pointer}
.finance-row input{width:18px;height:18px;accent-color:var(--brand);flex:none}
.finance-row b{font-size:14px;font-weight:600}
.finance-row small{display:block;color:var(--body);font-weight:400;font-size:12px}
.actions{display:flex;gap:10px;align-items:center;margin-top:18px;flex-wrap:wrap}
button{border:0;border-radius:2px;padding:0 20px;height:38px;font:inherit;font-size:14px;font-weight:500;cursor:pointer;background:var(--brand);color:#fff;transition:background .15s}
button:hover:not(:disabled){background:var(--brand-hover)}button:active:not(:disabled){background:var(--brand-active)}
button.secondary{background:var(--fill2);color:var(--title)}
button.secondary:hover:not(:disabled){background:#E5E6EB}
button:disabled{opacity:.5;cursor:not-allowed}
.hint{color:var(--secondary);font-size:12px}
.hidden{display:none!important}
.status{font-weight:600;font-size:13px}
.status.ok{color:var(--ok)}.status.warn{color:var(--err)}
.metrics{display:flex;gap:8px;flex-wrap:wrap;margin:2px 0 4px}
.metric{display:inline-flex;gap:6px;align-items:center;background:var(--fill1);border:1px solid var(--border);border-radius:2px;padding:4px 10px;font-size:12px;color:var(--body)}
.metric b{color:var(--brand);font-size:13px;font-weight:600}
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:8px}
.fact{border:1px solid var(--border);border-left:3px solid var(--brand);border-radius:2px;padding:10px 12px;background:#fff;font-size:13px;line-height:1.65}
.fact b{color:var(--brand);font-size:12px;display:block;margin-bottom:2px;font-weight:600}
.question{padding:10px 12px;border-left:3px solid var(--warn);background:var(--warn-bg);border-radius:0 2px 2px 0;margin:8px 0;font-size:13px;line-height:1.65;color:var(--title)}
.question b{color:#B25E09}
.question textarea{min-height:60px;margin-top:8px}
.result{white-space:pre-wrap;background:#fff;color:var(--title);border:1px solid var(--border);border-radius:2px;padding:20px;max-height:600px;overflow:auto;font:13px/1.75 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
@media(max-width:760px){.grid{grid-template-columns:1fr}.wide{grid-column:auto}.hero{padding:20px}}
</style></head>
<body><main class="shell">
<header class="hero"><span class="eyebrow">EVIDENCE-AWARE · 证据优先</span>
<h1>把岗位要求，变成<span class="accent">可信的</span>定向简历</h1>
<p>先建事实账本，再让你确认；未确认的事实不会进入最终稿。含 Claim Ledger 原子校验、角色门禁、面试防御与能力深挖。</p>
<div class="steps"><span class="step"><span class="n">1</span>输入简历与 JD</span><span class="step"><span class="n">2</span>确认事实与深挖</span><span class="step"><span class="n">3</span>生成可信简历</span></div>
</header>
<section class="grid">
<div class="card"><h2><span class="n">1</span>输入材料</h2>
<label class="f-label" for="resume">基础简历</label><textarea id="resume" placeholder="粘贴简历文本，建议保留模块标题"></textarea>
<label class="f-label" for="jd">岗位 JD</label><textarea id="jd" placeholder="粘贴岗位描述"></textarea>
<label class="f-label" for="title">求职方向（可选）</label><input type="text" id="title" placeholder="例如：大模型应用研发实习生">
<label class="finance-row"><input type="checkbox" id="finance"><span><b>🏦 金融岗位模式</b><small>勾选则强制开启（金融能力词典+金融视角改写）；不勾选时按 JD 自动检测。只重述已有经历，不编造。</small></span></label>
<div class="actions"><button id="analyze">开始分析</button><span id="status" class="status"></span></div>
<p class="hint">材料只在当前进程中处理；真实模型调用尚未接入此页面。</p></div>
<div id="analysisCard" class="card hidden"><h2><span class="n">2</span>证据与匹配</h2><div id="metrics" class="metrics"></div>
<h3>📋 事实账本</h3><div id="facts" class="facts"></div>
<h3>❓ 待确认问题</h3><div id="questions"></div>
<h3>🛡️ 面试防御</h3><div id="defense"></div>
<h3>🔍 深挖</h3><div id="dig"></div><h3>✅ ATS 检查</h3><div id="ats"></div>
<div class="actions"><button id="submitDig" class="secondary hidden">提交深挖回答</button><button id="generate" disabled>确认事实并生成</button></div>
<p class="hint">确认即表示认可当前事实账本；待确认问题不会被擅自补写进简历。</p></div>
<div id="resultCard" class="card wide hidden"><h2><span class="n">3</span>结果与交付</h2><div id="resultMeta" class="metrics"></div><pre id="result" class="result"></pre>
<div class="actions"><button id="download" class="secondary">下载 Markdown</button><button id="downloadPdf" class="secondary">下载 PDF</button></div></div>
</section></main>
<script>
const $=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let taskId=null,finalResume='';
$('analyze').onclick=async()=>{const resume=$('resume').value.trim(),jd=$('jd').value.trim();if(!resume||!jd){$('status').textContent='请先填写简历和 JD';$('status').className='status warn';return}$('analyze').disabled=true;$('status').textContent='分析中…';try{const r=await fetch('api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({resume,jd,job_title:$('title').value.trim(),finance_mode:$('finance').checked?true:null})});const d=await r.json();if(!r.ok)throw Error(d.error||'分析失败');taskId=d.task_id;const a=d.analysis;const finBadge=a.finance_mode?'<span class="metric">🏦 金融模式 <b>已开启</b></span>':'';const atsScore=d.ats_score!=null?`<span class="metric">ATS <b>${d.ats_score}</b></span>`:'';const cov=d.match_coverage?`<span class="metric">匹配覆盖率 <b>${d.match_coverage.weighted_coverage}%</b></span>`:'';$('metrics').innerHTML=finBadge+atsScore+cov+`<span class="metric">岗位要求 <b>${a.requirements.length}</b></span><span class="metric">事实 <b>${d.fact_ledger.length}</b></span><span class="metric">待确认 <b>${a.confirmation_questions.length}</b></span>`;$('facts').innerHTML=d.fact_ledger.map(f=>`<div class="fact"><b>${esc(f.fact_id)} · ${esc(f.section)}</b>${esc(f.text)}</div>`).join('')||'<span class="hint">未解析到事实</span>';$('questions').innerHTML=a.confirmation_questions.map(q=>`<div class="question"><b>${esc(q.category||'待确认')}</b>：${esc(q.question||q.text||'')}</div>`).join('')||'<span class="status ok">当前没有待确认问题</span>';$('defense').innerHTML=(d.interview_defense||[]).map(x=>`<div class="question"><b>${esc(x.claim_id)}</b> ${esc(x.claim_text)}${x.questions.map(q=>`<br>❓ ${esc(q.question)}<br><span class="hint">追问：${esc(q.followups.join(' / '))}<br>边界：${esc(q.boundary)}</span>`).join('')}</div>`).join('')||'<span class="hint">无强表述，无需防御</span>';window._digQs=d.dig_questions||[];$('dig').innerHTML=(d.dig_questions||[]).map(q=>`<div class="question"><b>${esc(q.question_id)} · ${esc(q.dimension)}</b>：${esc(q.question)}<br><textarea id="dig_${esc(q.question_id)}" rows="2" style="width:100%;margin-top:6px" placeholder="写下你的回答（具体事实，不要编）"></textarea></div>`).join('')||'<span class="hint">暂无深挖问题</span>';$('ats').innerHTML=(d.ats_checks||[]).map(c=>`<div class="question" style="border-left-color:${c.status==='pass'?'var(--ok)':c.status==='warn'?'var(--warn)':'var(--err)'}"><b>${esc(c.check)}</b>：${esc(c.detail)}</div>`).join('');if((d.dig_questions||[]).length)$('submitDig').classList.remove('hidden');$('analysisCard').classList.remove('hidden');$('generate').disabled=false;$('status').textContent='分析完成';$('status').className='status ok';$('analysisCard').scrollIntoView({behavior:'smooth',block:'start'})}catch(e){$('status').textContent=e.message;$('status').className='status warn'}finally{$('analyze').disabled=false}};
$('submitDig').onclick=async()=>{if(!taskId)return;const answers=(window._digQs||[]).map(q=>({question_id:q.question_id,claim_id:q.claim_id,answer:($('dig_'+q.question_id)||{}).value||''})).filter(a=>a.answer.trim());if(!answers.length){$('status').textContent='请先填写至少一个回答';$('status').className='status warn';return}$('submitDig').disabled=true;$('status').textContent='深挖中…';try{const r=await fetch('api/dig',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task_id:taskId,answers})});const d=await r.json();if(!r.ok)throw Error(d.error||'深挖失败');$('status').textContent='深挖完成，新增 '+d.added+' 条事实';$('status').className='status ok';$('defense').innerHTML=(d.interview_defense||[]).map(x=>`<div class="question"><b>${esc(x.claim_id)}</b> ${esc(x.claim_text)}${x.questions.map(q=>`<br>❓ ${esc(q.question)}`).join('')}</div>`).join('')}catch(e){$('status').textContent=e.message;$('status').className='status warn'}finally{$('submitDig').disabled=false}};
$('generate').onclick=async()=>{if(!taskId)return;$('generate').disabled=true;$('status').textContent='生成与真实性校验中…';try{const r=await fetch('api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task_id:taskId})});const d=await r.json();if(!r.ok)throw Error(d.error||'生成失败');finalResume=d.final_resume;$('result').textContent=finalResume;$('resultMeta').innerHTML='<span class="metric">状态 <b>QA 前置校验通过</b></span><span class="metric">任务 <b>'+esc(taskId)+'</b></span>';$('resultCard').classList.remove('hidden');$('status').textContent='生成完成';$('status').className='status ok';$('resultCard').scrollIntoView({behavior:'smooth',block:'start'})}catch(e){$('status').textContent=e.message;$('status').className='status warn'}finally{$('generate').disabled=false}};
$('downloadPdf').onclick=()=>{if(!taskId)return;const a=document.createElement('a');a.href='api/pdf?task_id='+encodeURIComponent(taskId);a.download='resume.pdf';a.click()};
$('download').onclick=()=>{if(!finalResume)return;const b=new Blob([finalResume],{type:'text/markdown'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='resume.md';a.click()};
</script>
</body></html>
'''


def _prune_tasks_locked() -> None:
    now = time.monotonic()
    for key in [k for k, v in TASKS.items() if now - v["created_at"] > TASK_TTL_SECONDS]:
        TASKS.pop(key, None)


def _json(handler: BaseHTTPRequestHandler, obj: dict, status: int = HTTPStatus.OK) -> None:
    body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # noqa: ANN002,ANN003
        pass

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/pdf":
            q = parse_qs(parsed.query)
            task_id = str((q.get("task_id") or [""])[0])
            with TASKS_LOCK:
                _prune_tasks_locked()
                task = TASKS.get(task_id)
                pdf_bytes = render_pdf(task["final_resume"]) if task and task.get("final_resume") else None
            if not pdf_bytes:
                _json(self, {"error": "任务不存在或未生成"}, HTTPStatus.NOT_FOUND); return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/pdf")
            self.send_header("Content-Disposition", "attachment; filename=resume.pdf")
            self.send_header("Content-Length", str(len(pdf_bytes)))
            self.end_headers()
            self.wfile.write(pdf_bytes)
            return
        if path in ("/", "/resume", "/resume/"):
            body = INDEX_HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if not length:
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
                finance_flag = data.get("finance_mode")
                finance_mode = True if finance_flag is True else None
                result = run_python_workflow(resume, jd, job_title=data.get("job_title") or None, finance_mode=finance_mode)
                task_id = secrets.token_urlsafe(10)
                task = {"task_id": task_id, "status": "awaiting_confirmation", "created_at": time.monotonic(), "resume": resume, "jd": jd, "job_title": data.get("job_title") or None, **result}
                with TASKS_LOCK:
                    _prune_tasks_locked()
                    if len(TASKS) >= MAX_TASKS:
                        oldest = min(TASKS, key=lambda key: TASKS[key]["created_at"])
                        TASKS.pop(oldest, None)
                    TASKS[task_id] = task
                _json(self, {"task_id": task_id, "status": task["status"], "analysis": result["analysis"], "fact_ledger": result["fact_ledger"], "interview_defense": result["interview_defense"], "dig_questions": result["dig_questions"], "match_coverage": result["match_coverage"], "ats_checks": result["ats_checks"], "ats_score": result["ats_score"]})
                return
            if path == "/api/dig":
                task_id = str(data.get("task_id", ""))
                qa_pairs = data.get("answers", [])
                with TASKS_LOCK:
                    _prune_tasks_locked()
                    task = TASKS.get(task_id)
                    if not task:
                        response = None
                    else:
                        dig = apply_dig_answers(task["resume"], task["claims"], qa_pairs)
                        task["claims"] = dig["claims"]
                        task["interview_defense"] = dig["interview_defense"]
                        response = {"task_id": task_id, "added": dig["added"],
                                    "claims": dig["claims"], "interview_defense": dig["interview_defense"],
                                    "claim_errors": dig["claim_errors"]}
                if response is None:
                    _json(self, {"error": "任务不存在"}, HTTPStatus.NOT_FOUND); return
                _json(self, response)
                return
            if path == "/api/pdf":
                task_id = str(data.get("task_id", ""))
                with TASKS_LOCK:
                    _prune_tasks_locked()
                    task = TASKS.get(task_id)
                    pdf_bytes = render_pdf(task["final_resume"]) if task and task.get("final_resume") else None
                if not pdf_bytes:
                    _json(self, {"error": "任务不存在或未生成"}, HTTPStatus.NOT_FOUND); return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/pdf")
                self.send_header("Content-Disposition", "attachment; filename=resume.pdf")
                self.send_header("Content-Length", str(len(pdf_bytes)))
                self.end_headers()
                self.wfile.write(pdf_bytes)
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
            _json(self, {"error": "服务器内部错误"}, HTTPStatus.INTERNAL_SERVER_ERROR)


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
