"""provider 팩토리. 요청에 실려 온 (provider, apiKey, baseUrl, model)로 매번 새 인스턴스를 만든다."""
from __future__ import annotations

from urllib.parse import urlparse

from .. import config
from .base import (ContextWindowError, Message, ModelResponse, Provider, ProviderError, ToolCall, ToolSpec,
                   ToolsUnsupportedError)

__all__ = [
    "ContextWindowError", "Message", "ModelResponse", "Provider", "ProviderError", "ToolCall", "ToolSpec",
    "ToolsUnsupportedError", "create_provider",
]


def _validate_base_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ProviderError("서버 주소는 http:// 또는 https://로 시작해야 합니다. 예: http://127.0.0.1:11434/v1")
    return base_url.rstrip("/")


def create_provider(name: str, *, model: str = "", api_key: str = "", base_url: str = "",
                    disable_thinking: bool = False) -> Provider:
    name, model, api_key, base_url = str(name or ""), str(model or "").strip(), str(api_key or "").strip(), str(base_url or "").strip()
    if name not in config.ALL_PROVIDERS:
        raise ProviderError(f"지원하지 않는 provider입니다: {name or '(없음)'}")

    if name == "openaiCompatible":
        if not base_url:
            raise ProviderError("로컬/자체 호스팅 API를 쓰려면 서버 주소(base URL)가 필요합니다.")
        from .openai_compat import OpenAICompatProvider
        return OpenAICompatProvider(name=name, model=model, api_key=api_key, base_url=_validate_base_url(base_url),
                                    timeout=config.LOCAL_TIMEOUT_SECONDS, is_local=True,
                                    disable_thinking=disable_thinking)

    if not api_key:
        raise ProviderError("이 provider는 API key가 필요합니다. 설정에서 입력하세요.")
    if base_url:
        base_url = _validate_base_url(base_url)
    if name == "openai":
        from .openai_compat import OpenAICompatProvider
        return OpenAICompatProvider(name=name, model=model, api_key=api_key, base_url=base_url,
                                    timeout=config.CLOUD_TIMEOUT_SECONDS, is_local=False)
    if name == "anthropic":
        from .anthropic_provider import AnthropicProvider
        return AnthropicProvider(model=model, api_key=api_key, base_url=base_url, timeout=config.CLOUD_TIMEOUT_SECONDS)
    from .gemini_provider import GeminiProvider
    return GeminiProvider(model=model, api_key=api_key, base_url=base_url, timeout=config.CLOUD_TIMEOUT_SECONDS)
