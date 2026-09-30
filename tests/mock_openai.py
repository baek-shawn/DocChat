"""OpenAI 호환 mock 서버 — 실제 `openai` SDK 경로(HTTP)를 그대로 태워 테스트한다.

handler(body) 반환값:
  - str                                   → assistant 본문
  - {"status": 400, "body": {...}}        → HTTP 오류
  - {"text": "...", "finish_reason": "length", "tool_calls": [{"name": "...", "arguments": {...}}]}
  - {"text": "", "reasoning": "...", "finish_reason": "length"}   → 추론을 본문과 따로 주는 서버(vLLM의 reasoning_content)
"""
from __future__ import annotations

import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

from PIL import Image

Handler = Callable[[dict[str, Any]], Any]


class MockOpenAIServer:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.handler: Handler = lambda _body: "mock reply"
        self.models: list[str] = ["mock-vlm"]
        self._lock = threading.Lock()
        outer = self

        class RequestHandler(BaseHTTPRequestHandler):
            def log_message(self, *_args: Any) -> None:  # 테스트 출력 조용히
                return

            def _send(self, status: int, payload: dict[str, Any]) -> None:
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                if self.path.rstrip("/") == "/v1/models":
                    self._send(200, {"object": "list", "data": [{"id": name, "object": "model"} for name in outer.models]})
                else:
                    self._send(404, {"error": {"message": "not found"}})

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                if self.path.rstrip("/") != "/v1/chat/completions":
                    self._send(404, {"error": {"message": "not found"}})
                    return
                with outer._lock:
                    outer.requests.append(body)
                    handler = outer.handler
                try:
                    outcome = handler(body)
                except Exception as error:  # 핸들러 버그를 테스트에서 바로 보이게
                    self._send(500, {"error": {"message": f"mock handler failed: {error}"}})
                    return
                if isinstance(outcome, dict) and "status" in outcome:
                    self._send(int(outcome["status"]), outcome.get("body") or {"error": {"message": "mock error"}})
                    return
                spec = outcome if isinstance(outcome, dict) else {"text": str(outcome)}
                message: dict[str, Any] = {"role": "assistant", "content": spec.get("text") or ""}
                if spec.get("reasoning"):
                    message["reasoning_content"] = spec["reasoning"]
                if spec.get("tool_calls"):
                    message["tool_calls"] = [
                        {"id": f"call_{index}", "type": "function",
                         "function": {"name": call["name"], "arguments": json.dumps(call.get("arguments") or {})}}
                        for index, call in enumerate(spec["tool_calls"])
                    ]
                finish = spec.get("finish_reason") or ("tool_calls" if spec.get("tool_calls") else "stop")
                self._send(200, {
                    "id": "chatcmpl-mock", "object": "chat.completion", "created": 0, "model": body.get("model", "mock"),
                    "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                })

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), RequestHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def start(self) -> "MockOpenAIServer":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def reset(self, handler: Handler | None = None) -> None:
        with self._lock:
            self.requests.clear()
            if handler is not None:
                self.handler = handler


# --------------------------------------------------------------------------- 요청 본문 검사 도우미
def system_text(body: dict[str, Any]) -> str:
    return "\n".join(str(m.get("content") or "") for m in body.get("messages", []) if m.get("role") == "system")


def all_text(body: dict[str, Any]) -> str:
    """메시지의 모든 텍스트(문자열 content + text 파트)를 이어 붙인다. 이미지 base64는 제외."""
    chunks: list[str] = []
    for message in body.get("messages", []):
        content = message.get("content")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            chunks += [str(part.get("text") or "") for part in content if part.get("type") == "text"]
    return "\n".join(chunks)


def image_count(body: dict[str, Any]) -> int:
    return sum(
        1
        for message in body.get("messages", [])
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if part.get("type") == "image_url"
    )


def request_images(body: dict[str, Any]) -> list[Image.Image]:
    """요청에 실린 이미지를 실제로 디코드한다 — 모델이 "무엇을 봤는지"(크기·내용)를 검사할 때 쓴다."""
    images = []
    for message in body.get("messages", []):
        if not isinstance(message.get("content"), list):
            continue
        for part in message["content"]:
            if part.get("type") == "image_url":
                encoded = part["image_url"]["url"].split(",", 1)[1]
                image = Image.open(io.BytesIO(base64.b64decode(encoded)))
                image.load()
                images.append(image)
    return images


def thinking_disabled(body: dict[str, Any]) -> bool:
    """이 요청이 추론을 끄고 왔는가(`chat_template_kwargs.enable_thinking == false`)."""
    return (body.get("chat_template_kwargs") or {}).get("enable_thinking") is False


def is_ocr_call(body: dict[str, Any]) -> bool:
    return "transcription engine" in system_text(body)


def is_grounding_call(body: dict[str, Any]) -> bool:
    return "visual grounding engine" in system_text(body)
