import pytest
from unittest.mock import MagicMock
from src.llm.backend import create_backend, LLMBackend
from src.llm.providers.anthropic import AnthropicBackend


def test_anthropic_backend_implements_protocol():
    backend = AnthropicBackend(model="claude-haiku-4-5-20251001", api_key="test-key")
    assert isinstance(backend, LLMBackend)


def test_create_backend_anthropic():
    backend = create_backend(provider="anthropic", model="claude-haiku-4-5-20251001", api_key="test-key")
    assert isinstance(backend, AnthropicBackend)


def test_create_backend_unknown_raises():
    with pytest.raises(ValueError, match="Unknown provider"):
        create_backend(provider="unknown_llm", model="model", api_key="key")


def test_anthropic_backend_complete(monkeypatch):
    """Test that AnthropicBackend calls the API and returns content."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text='{"eval_metric": "roc_auc"}')]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    backend = AnthropicBackend(model="claude-haiku-4-5-20251001", api_key="test-key")
    backend._client = mock_client

    result = backend.complete(messages=[{"role": "user", "content": "test"}])
    assert result == '{"eval_metric": "roc_auc"}'
    mock_client.messages.create.assert_called_once()
