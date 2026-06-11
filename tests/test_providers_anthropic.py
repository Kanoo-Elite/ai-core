"""X3 proof gate — Anthropic provider unit tests.

Mocks ``anthropic.Anthropic`` throughout; no real API calls are made.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from kanoo_ai_core.providers.anthropic import AnthropicProvider


def _build_mock_response(text: str = "Hi.") -> SimpleNamespace:
    """Build a stand-in for an ``anthropic.types.Message`` response."""
    return SimpleNamespace(
        id="msg_test",
        model="claude-haiku-4-5",
        role="assistant",
        content=[SimpleNamespace(text=text, type="text")],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


def _build_mock_client(response=None) -> MagicMock:
    client = MagicMock()
    client.messages.create.return_value = response or _build_mock_response()
    return client


# --- Constructor + reject paths -------------------------------------------


def test_provider_rejects_empty_api_key():
    with pytest.raises(ValueError):
        AnthropicProvider(api_key="")


def test_provider_rejects_streaming():
    client = _build_mock_client()
    provider = AnthropicProvider(api_key="k", client=client)
    with pytest.raises(ValueError, match="Streaming"):
        provider.call({"model": "m", "messages": [], "stream": True})
    client.messages.create.assert_not_called()


# --- Request translation --------------------------------------------------


def test_request_translation_lifts_system_role():
    client = _build_mock_client()
    provider = AnthropicProvider(api_key="k", client=client)
    provider.call(
        {
            "model": "claude-haiku-4-5",
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hi"},
            ],
            "max_tokens": 100,
        }
    )
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["system"] == "You are helpful"
    assert kwargs["messages"] == [{"role": "user", "content": "Hi"}]
    assert kwargs["max_tokens"] == 100


def test_request_translation_concatenates_multiple_system_messages():
    client = _build_mock_client()
    provider = AnthropicProvider(api_key="k", client=client)
    provider.call(
        {
            "model": "m",
            "messages": [
                {"role": "system", "content": "Part one."},
                {"role": "system", "content": "Part two."},
                {"role": "user", "content": "go"},
            ],
        }
    )
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["system"] == "Part one.\n\nPart two."


def test_request_translation_strips_openai_prefix_from_model():
    client = _build_mock_client()
    provider = AnthropicProvider(api_key="k", client=client)
    provider.call({"model": "openai/claude-haiku-4-5", "messages": []})
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-haiku-4-5"


def test_request_translation_passes_temperature_top_p():
    client = _build_mock_client()
    provider = AnthropicProvider(api_key="k", client=client)
    provider.call(
        {"model": "m", "messages": [], "temperature": 0.3, "top_p": 0.9}
    )
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["temperature"] == 0.3
    assert kwargs["top_p"] == 0.9


def test_request_translation_stop_to_stop_sequences_string():
    client = _build_mock_client()
    provider = AnthropicProvider(api_key="k", client=client)
    provider.call({"model": "m", "messages": [], "stop": "DONE"})
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["stop_sequences"] == ["DONE"]


def test_request_translation_stop_to_stop_sequences_list():
    client = _build_mock_client()
    provider = AnthropicProvider(api_key="k", client=client)
    provider.call({"model": "m", "messages": [], "stop": ["A", "B"]})
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["stop_sequences"] == ["A", "B"]


def test_request_translation_default_max_tokens():
    client = _build_mock_client()
    provider = AnthropicProvider(api_key="k", client=client)
    provider.call({"model": "m", "messages": []})
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["max_tokens"] == 1024


# --- Response translation -------------------------------------------------


def test_response_translation_concatenates_content_blocks():
    response = SimpleNamespace(
        id="msg_x",
        model="m",
        role="assistant",
        content=[
            SimpleNamespace(text="Hello ", type="text"),
            SimpleNamespace(text="world.", type="text"),
        ],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=3, output_tokens=2),
    )
    client = _build_mock_client(response=response)
    provider = AnthropicProvider(api_key="k", client=client)
    result = provider.call({"model": "m", "messages": []})
    assert result["choices"][0]["message"]["content"] == "Hello world."


def test_response_translation_maps_finish_reasons():
    mappings = {
        "end_turn": "stop",
        "max_tokens": "length",
        "stop_sequence": "stop",
        "tool_use": "stop",
    }
    for anthropic_reason, openai_reason in mappings.items():
        response = SimpleNamespace(
            id="msg",
            model="m",
            role="assistant",
            content=[SimpleNamespace(text="x", type="text")],
            stop_reason=anthropic_reason,
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )
        client = _build_mock_client(response=response)
        provider = AnthropicProvider(api_key="k", client=client)
        result = provider.call({"model": "m", "messages": []})
        assert result["choices"][0]["finish_reason"] == openai_reason


def test_response_translation_maps_token_counts():
    response = SimpleNamespace(
        id="msg",
        model="m",
        role="assistant",
        content=[SimpleNamespace(text="x", type="text")],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=42, output_tokens=7),
    )
    client = _build_mock_client(response=response)
    provider = AnthropicProvider(api_key="k", client=client)
    result = provider.call({"model": "m", "messages": []})
    assert result["usage"]["prompt_tokens"] == 42
    assert result["usage"]["completion_tokens"] == 7
    assert result["usage"]["total_tokens"] == 49


def test_response_carries_chat_completion_object_marker():
    client = _build_mock_client()
    provider = AnthropicProvider(api_key="k", client=client)
    result = provider.call({"model": "m", "messages": []})
    assert result["object"] == "chat.completion"


# --- Guardrails ordering --------------------------------------------------


def test_guardrails_invoked_before_llm_call():
    """The redaction sentinel must appear in the args passed to anthropic.create."""
    client = _build_mock_client()
    provider = AnthropicProvider(api_key="k", client=client)
    provider.call(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "Email a@b.com"}],
        }
    )
    kwargs = client.messages.create.call_args.kwargs
    sent_content = kwargs["messages"][0]["content"]
    assert "[EMAIL-REDACTED]" in sent_content
    assert "a@b.com" not in sent_content


# --- Failure modes --------------------------------------------------------


def test_provider_propagates_anthropic_api_error():
    client = MagicMock()
    client.messages.create.side_effect = RuntimeError("anthropic blew up")
    provider = AnthropicProvider(api_key="k", client=client)
    with pytest.raises(RuntimeError, match="anthropic blew up"):
        provider.call({"model": "m", "messages": []})
