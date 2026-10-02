"""OpenAI 호환 mock 서버 — 실제 `openai` SDK 경로(HTTP)를 그대로 태워 테스트한다.

handler(body) 반환값:
  - str                                   → assistant 본문
  - {"status": 400, "body": {...}}        → HTTP 오류
  - {"text": "...", "finish_reason": "length", "tool_calls": [{"name": "...", "arguments": {...}}]}
  - {"text": "", "reasoning": "...", "finish_reason": "length"}   → 추론을 본문과 따로 주는 서버(vLLM의 reasoning_content)

요청이 `stream: true`면(Step 6, 추론을 켠 호출) 같은 spec을 SSE 조각으로 흘려보낸다: 추론 → 본문 → 도구 호출 → 종료 사유
→ (`stream_options.include_usage`면) usage → `[DONE]`. spec에 더 쓸 수 있는 것:
  - "chunk_chars": 조각 하나의 글자 수(기본 4 — vLLM의 토큰 하나 ≈ 3.8자)
  - "inline_think": True → 추론을 따로 주지 않고 본문 속 `<think>…</think>`로 섞어 보내는 서버 흉내
  - "reasoning_lines": [...], "repeat": N → 그 줄들을 N번 되풀이하는 긴 추론(앱이 중간에 끊는 경우를 위해)
"""
from __future__ import annotations

import base64
import io
import json
import sys
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
                if spec.get("reasoning_lines"):
                    spec = {**spec, "reasoning": "\n".join(list(spec["reasoning_lines"]) * int(spec.get("repeat") or 1))}
                tool_calls = [
                    {"id": f"call_{index}", "type": "function",
                     "function": {"name": call["name"], "arguments": json.dumps(call.get("arguments") or {})}}
                    for index, call in enumerate(spec.get("tool_calls") or [])
                ]
                finish = spec.get("finish_reason") or ("tool_calls" if tool_calls else "stop")
                if body.get("stream"):
                    self._send_stream(body, spec, tool_calls, finish)
                    return
                message: dict[str, Any] = {"role": "assistant", "content": spec.get("text") or ""}
                if spec.get("reasoning"):
                    message["reasoning_content"] = spec["reasoning"]
                if tool_calls:
                    message["tool_calls"] = tool_calls
                self._send(200, {
                    "id": "chatcmpl-mock", "object": "chat.completion", "created": 0, "model": body.get("model", "mock"),
                    "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                })

            def _send_stream(self, body: dict[str, Any], spec: dict[str, Any], tool_calls: list[dict[str, Any]],
                             finish: str) -> None:
                """SSE 조각. 클라이언트가 중간에 연결을 끊으면 쓰기가 실패해 여기서 끝난다(QuietServer가 조용히 삼킨다)."""
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                base = {"id": "chatcmpl-mock", "object": "chat.completion.chunk", "created": 0, "model": body.get("model", "mock")}
                size = max(1, int(spec.get("chunk_chars") or 4))
                sent = 0

                def emit(payload: dict[str, Any]) -> None:
                    self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode("utf-8"))
                    self.wfile.flush()

                def pieces(text: str):
                    for start in range(0, len(text), size):
                        yield text[start:start + size]

                reasoning = spec.get("reasoning") or ""
                content = spec.get("text") or ""
                if reasoning and spec.get("inline_think"):
                    content = f"<think>\n{reasoning}\n</think>\n{content}"
                    reasoning = ""
                for piece in pieces(reasoning):
                    sent += 1
                    emit({**base, "choices": [{"index": 0, "delta": {"reasoning_content": piece}, "finish_reason": None}]})
                for piece in pieces(content):
                    sent += 1
                    emit({**base, "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]})
                if tool_calls:
                    emit({**base, "choices": [{"index": 0, "delta": {"tool_calls": [
                        {"index": index, **call} for index, call in enumerate(tool_calls)]}, "finish_reason": None}]})
                emit({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
                if (body.get("stream_options") or {}).get("include_usage"):
                    emit({**base, "choices": [], "usage": {"prompt_tokens": int(spec.get("prompt_tokens") or 0),
                                                           "completion_tokens": sent, "total_tokens": sent}})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        class QuietServer(ThreadingHTTPServer):
            def handle_error(self, request: Any, client_address: Any) -> None:
                # 앱이 요청을 취소하면(사용자 "중지") 핸들러가 닫힌 소켓에 쓰다 실패한다 — 테스트 출력에 트레이스백을 찍지 않는다.
                if isinstance(sys.exc_info()[1], (ConnectionError, BrokenPipeError)):
                    return
                super().handle_error(request, client_address)

        self._server = QuietServer(("127.0.0.1", 0), RequestHandler)
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
