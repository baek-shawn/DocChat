"""Step 2 — provider 팩토리와 메시지 변환(네트워크 없이 검증할 수 있는 부분).

Anthropic/Gemini는 실제 API key가 없어 **라이브 호출은 검증하지 못했다.** 여기서는 SDK에 넘기는
요청 형태와 응답 해석만 확인한다.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app import config
from app.pipeline.images import ModelImage
from app.providers import ProviderError, create_provider
from app.providers.anthropic_provider import (AnthropicProvider, parse_anthropic_response, split_system,
                                              to_anthropic_messages, to_anthropic_tools)
from app.providers.base import (ToolCall, ToolSpec, compact_messages, image_anchor_index, is_context_window_error,
                                looks_like_tools_unsupported)
from app.providers.gemini_provider import GeminiProvider
from app.providers.openai_compat import OpenAICompatProvider, to_wire_messages, to_wire_tools

IMAGE = ModelImage(name="a.png", mime="image/png", data=b"\x89PNG-bytes")
TOOL = ToolSpec(name="inspect_visual", description="d", parameters={"type": "object", "properties": {}})
CONVERSATION = [
    {"role": "system", "content": "SYS"},
    {"role": "user", "content": "first"},
    {"role": "assistant", "content": "", "tool_calls": [ToolCall(name="inspect_visual", arguments={"name": "a.png"}, id="c1")]},
    {"role": "tool", "tool_call_id": "c1", "name": "inspect_visual", "content": '{"regions":[]}'},
]


# --------------------------------------------------------------------------- 팩토리
def test_factory_validates_inputs_with_helpful_messages():
    with pytest.raises(ProviderError, match="지원하지 않는"):
        create_provider("skynet", model="m")
    with pytest.raises(ProviderError, match="서버 주소"):
        create_provider("openaiCompatible", model="m")
    with pytest.raises(ProviderError, match="http://"):
        create_provider("openaiCompatible", model="m", base_url="localhost:11434/v1")
    for name in ("openai", "anthropic", "gemini"):
        with pytest.raises(ProviderError, match="API key"):
            create_provider(name, model="m")


def test_one_implementation_serves_local_runtimes_and_openai():
    local = create_provider("openaiCompatible", model="gemma3", base_url="http://127.0.0.1:11434/v1/")
    cloud = create_provider("openai", model="gpt", api_key="sk-test")
    assert isinstance(local, OpenAICompatProvider) and isinstance(cloud, OpenAICompatProvider)
    assert local.is_local and local.base_url == "http://127.0.0.1:11434/v1" and local.timeout == config.LOCAL_TIMEOUT_SECONDS
    assert not cloud.is_local and cloud.base_url == "https://api.openai.com/v1"
    assert local.cache_namespace == "openaiCompatible:http://127.0.0.1:11434/v1:gemma3"
    assert isinstance(create_provider("anthropic", model="claude", api_key="k"), AnthropicProvider)
    assert isinstance(create_provider("gemini", model="gemini", api_key="k"), GeminiProvider)


# --------------------------------------------------------------------------- 공통 도우미
def test_error_classification():
    assert is_context_window_error("the request exceeds the available context size")
    assert is_context_window_error("This model's maximum context length is 8192 tokens")
    assert not is_context_window_error("invalid api key")
    assert looks_like_tools_unsupported(400, "gemma3 does not support tools")
    assert looks_like_tools_unsupported(None, "chat template error")
    assert not looks_like_tools_unsupported(401, "unauthorized")


def test_images_attach_to_the_anchored_question_not_the_latest_user_message():
    messages = [{"role": "user", "content": "question", "images_anchor": True},
                {"role": "assistant", "content": "{}"}, {"role": "user", "content": "TOOL RESULT"}]
    assert image_anchor_index(messages) == 0
    assert image_anchor_index([{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]) == 1


def test_compaction_keeps_system_and_never_orphans_tool_results():
    long = [{"role": "system", "content": "S" * 9000}, *[{"role": "user", "content": f"u{i}" * 3000} for i in range(6)],
            {"role": "assistant", "content": "", "tool_calls": [ToolCall(name="t", arguments={})]},
            {"role": "tool", "tool_call_id": "x", "content": "R" * 9000}]
    level1, level2 = compact_messages(long, 1), compact_messages(long, 2)
    assert level1[0]["role"] == "system" and len(level1[0]["content"]) < 3400
    assert len(level2) < len(level1) and sum(len(m["content"]) for m in level2) < sum(len(m["content"]) for m in level1)
    only_tool_tail = compact_messages([{"role": "user", "content": "q"}] * 9 + [{"role": "tool", "content": "r"}] * 3, 2)
    assert all(message["role"] != "tool" for message in only_tool_tail)


# --------------------------------------------------------------------------- OpenAI 호환
def test_openai_wire_format():
    wire = to_wire_messages(CONVERSATION, None)
    assert wire[0] == {"role": "system", "content": "SYS"}
    assert wire[2]["content"] is None and wire[2]["tool_calls"][0] == {
        "id": "c1", "type": "function", "function": {"name": "inspect_visual", "arguments": '{"name": "a.png"}'}}
    assert wire[3] == {"role": "tool", "tool_call_id": "c1", "content": '{"regions":[]}'}
    with_image = to_wire_messages(CONVERSATION[:2], [IMAGE])[1]["content"]
    assert with_image[0] == {"type": "text", "text": "first"}
    assert with_image[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert to_wire_tools([TOOL])[0]["function"]["name"] == "inspect_visual"


# --------------------------------------------------------------------------- Anthropic
def test_anthropic_request_shape():
    system, turns = split_system(CONVERSATION)
    messages = to_anthropic_messages(turns, [IMAGE])
    assert system == "SYS" and [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert [block["type"] for block in messages[0]["content"]] == ["text", "image"]
    assert messages[0]["content"][1]["source"]["media_type"] == "image/png"
    assert messages[1]["content"] == [{"type": "tool_use", "id": "c1", "name": "inspect_visual", "input": {"name": "a.png"}}]
    assert messages[2]["content"][0] == {"type": "tool_result", "tool_use_id": "c1", "content": '{"regions":[]}'}
    assert to_anthropic_tools([TOOL]) == [{"name": "inspect_visual", "description": "d", "input_schema": TOOL.parameters}]
    # 같은 역할이 연달아 오면 한 턴으로 합친다(역할 교대 요구).
    merged = to_anthropic_messages([{"role": "user", "content": "a"}, {"role": "user", "content": "b"}], None)
    assert len(merged) == 1 and [block["text"] for block in merged[0]["content"]] == ["a", "b"]


def test_anthropic_response_parsing():
    response = SimpleNamespace(stop_reason="tool_use", content=[
        SimpleNamespace(type="text", text="살펴보겠습니다."),
        SimpleNamespace(type="tool_use", id="toolu_1", name="inspect_visual", input={"name": "a.png", "task": "t"})])
    parsed = parse_anthropic_response(response)
    assert parsed.text == "살펴보겠습니다." and parsed.finish_reason == "tool_use"
    assert [(c.id, c.name, c.arguments) for c in parsed.tool_calls] == [("toolu_1", "inspect_visual", {"name": "a.png", "task": "t"})]


# --------------------------------------------------------------------------- Gemini
def test_gemini_request_and_response_shape():
    provider = create_provider("gemini", model="gemini-test", api_key="k")
    contents = provider.to_contents(CONVERSATION, [IMAGE])
    assert [content.role for content in contents] == ["user", "model", "user"]          # system은 별도 인자로 간다
    assert contents[0].parts[0].text == "first" and contents[0].parts[1].inline_data.mime_type == "image/png"
    assert contents[1].parts[0].function_call.name == "inspect_visual"
    assert contents[2].parts[0].function_response.response == {"result": '{"regions":[]}'}

    types = provider._types
    raw = types.Content(role="model", parts=[types.Part(thought=True, text="thinking"),
                                             types.Part.from_function_call(name="inspect_visual", args={"name": "a.png"})])
    parsed = provider.parse_response(SimpleNamespace(candidates=[SimpleNamespace(content=raw, finish_reason=SimpleNamespace(name="STOP"))]))
    assert parsed.text == "" and parsed.tool_calls[0].arguments == {"name": "a.png"}
    # 함수 호출 턴은 원본 Content를 그대로 되돌려 보낸다(thought_signature 보존).
    replay = provider.to_contents([{"role": "assistant", "content": "", "tool_calls": parsed.tool_calls,
                                    "raw": parsed.raw_assistant}], None)
    assert replay[0] is raw
    truncated = provider.parse_response(SimpleNamespace(candidates=[SimpleNamespace(
        content=types.Content(role="model", parts=[types.Part.from_text(text="잘림")]), finish_reason=SimpleNamespace(name="MAX_TOKENS"))]))
    assert truncated.text == "잘림" and truncated.finish_reason == "length" and truncated.raw_assistant is None
