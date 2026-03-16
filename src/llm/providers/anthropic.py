from __future__ import annotations
from typing import Optional
import anthropic


class AnthropicBackend:
    def __init__(self, model: str, api_key: str):
        self._model = model
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        response_format: Optional[dict] = None,
    ) -> str:
        # Anthropic requires system message separate from messages list
        system = None
        filtered = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                filtered.append(m)

        kwargs = dict(
            model=self._model,
            max_tokens=4096,
            messages=filtered,
            temperature=temperature,
        )
        if system:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)
        return response.content[0].text
