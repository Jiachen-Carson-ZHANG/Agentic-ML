from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel


class Message(BaseModel):
    role: str   # "system" | "user" | "assistant"
    content: str


class LLMBackend:
    """Base class for LLM backends. Subclass and implement complete()."""

    def complete(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        raise NotImplementedError


def create_backend(provider: str, model: str, api_key: str) -> LLMBackend:
    """Factory: returns the correct LLMBackend for the given provider."""
    if provider == "anthropic":
        from src.llm.providers.anthropic import AnthropicBackend
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        return AnthropicBackend(model=model, client=client)
    elif provider == "openai":
        from src.llm.providers.openai import OpenAIBackend
        import openai
        client = openai.OpenAI(api_key=api_key)
        return OpenAIBackend(model=model, client=client)
    else:
        raise ValueError(f"Unknown provider: {provider!r}. Choose 'anthropic' or 'openai'.")
