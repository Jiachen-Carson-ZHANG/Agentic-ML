from __future__ import annotations
from typing import Optional, runtime_checkable, Protocol


@runtime_checkable
class LLMBackend(Protocol):
    def complete(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        response_format: Optional[dict] = None,
    ) -> str: ...


def create_backend(provider: str, model: str, api_key: str) -> LLMBackend:
    """Factory — returns the correct LLMBackend for the given provider."""
    if provider == "anthropic":
        from src.llm.providers.anthropic import AnthropicBackend
        return AnthropicBackend(model=model, api_key=api_key)
    elif provider == "openai":
        from src.llm.providers.openai import OpenAIBackend
        return OpenAIBackend(model=model, api_key=api_key)
    else:
        raise ValueError(f"Unknown provider: {provider!r}. Choose 'anthropic' or 'openai'.")
