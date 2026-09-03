"""最小 OpenAI-compatible LLM 客户端。

默认不会读取具体密钥，也不会自动发起网络请求。
真正调用时由外部环境提供 API_BASE_URL、API_KEY 和 MODEL。
"""

import json
import os
from pathlib import Path
from urllib import error, request


def _write_log(message: str) -> None:
    """写入不含密钥和提示词正文的调用日志。"""
    log_path = Path(__file__).parent / "output" / "llm-call.log"
    if log_path.parent.is_symlink() or log_path.is_symlink():
        raise RuntimeError("拒绝通过符号链接写入 LLM 日志")
    log_path.parent.mkdir(exist_ok=True, mode=0o700)
    log_path.parent.chmod(0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(log_path, flags, 0o600)
    os.fchmod(descriptor, 0o600)
    with os.fdopen(descriptor, "a", encoding="utf-8") as log_file:
        log_file.write(message + "\n")


def call_llm(prompt: str) -> str:
    """向 OpenAI-compatible /chat/completions 端点发送一次请求。"""
    base_url = os.environ.get("API_BASE_URL", "").rstrip("/")
    api_key = os.environ.get("API_KEY", "")
    model = os.environ.get("MODEL", "")

    if not base_url or not api_key or not model:
        raise RuntimeError(
            "缺少 API_BASE_URL、API_KEY 或 MODEL 环境变量；默认只运行本地预览。"
        )

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 2000,
        "response_format": {"type": "json_object"},
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    http_request = request.Request(
        f"{base_url}/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    _write_log("request started=true")
    try:
        with request.urlopen(http_request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        _write_log(f"error type=http status={exc.code}")
        # 不把上游响应正文（可能包含提示词回显、账号信息或其他敏感内容）带到终端。
        exc.read()
        raise RuntimeError(f"模型接口返回 HTTP {exc.code}") from exc
    except error.URLError as exc:
        _write_log("error type=connection")
        raise RuntimeError("模型接口连接失败") from exc

    _write_log("success response_received=true")
    try:
        return result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("模型响应格式不符合预期，缺少 choices.message.content") from exc
