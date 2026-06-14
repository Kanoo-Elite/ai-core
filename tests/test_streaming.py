"""Streaming tests for the Anthropic provider (KAIROS-032 v0.2.0).

Mocks ``anthropic.Anthropic.messages.create(stream=True, ...)`` with
sequences of ``SimpleNamespace`` event objects that mimic the shapes
captured during K032 Probe 3 (``message_start``,
``content_block_delta``, ``message_delta``, etc.).

No real API calls are made.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import anthropic
import pytest

from kanoo_ai_core.providers.anthropic import AnthropicProvider

# --- Mock builders --------------------------------------------------------


class _MockUpstreamStream:
    """Iterable mock for anthropic's streaming response.

    Tracks ``close()`` calls so tests can verify upstream cancellation
    on client disconnect.
    """

    def __init__(self, events):
        self._events = list(events)
        self._iter = iter(self._events)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iter)

    def close(self):
        self.closed = True


class _MockAPIError(anthropic.APIError):
    """Test-only APIError subclass with a single-arg init.

    Real ``anthropic.APIError`` requires httpx.Request scaffolding we
    do not need in unit tests; this subclass keeps ``isinstance`` checks
    against ``anthropic.APIError`` true while accepting just a message.
    """

    def __init__(self, message: str):
        Exception.__init__(self, message)


def _build_event_sequence(
    text_parts: list[str],
    *,
    stop_reason: str = "end_turn",
    msg_id: str = "msg_stream_test",
):
    """Build a list of Anthropic-shaped stream events.

    Sequence: message_start → content_block_start → content_block_delta
    (one per text part) → content_block_stop → message_delta →
    message_stop. Mirrors the shape captured in K032 Probe 3.
    """
    events = [
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(id=msg_id, role="assistant"),
        ),
        SimpleNamespace(
            type="content_block_start",
            index=0,
            content_block=SimpleNamespace(type="text", text=""),
        ),
    ]
    for part in text_parts:
        events.append(
            SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=SimpleNamespace(type="text_delta", text=part),
            )
        )
    events.extend(
        [
            SimpleNamespace(type="content_block_stop", index=0),
            SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(stop_reason=stop_reason),
                usage=SimpleNamespace(input_tokens=10, output_tokens=5),
            ),
            SimpleNamespace(type="message_stop"),
        ]
    )
    return events


def _build_mock_streaming_client(events):
    """Build a MagicMock anthropic client returning a streaming mock."""
    client = MagicMock()
    client.messages.create.return_value = _MockUpstreamStream(events)
    return client


def _parse_sse_chunk(line: str) -> dict:
    """Parse a single ``data: {...}\\n\\n`` SSE line into its JSON payload."""
    assert line.startswith("data: "), f"Not an SSE line: {line!r}"
    assert line.endswith("\n\n"), f"Missing terminator: {line!r}"
    body = line[len("data: "):].rstrip()
    return json.loads(body)


# --- stream_call gating ---------------------------------------------------


def test_stream_call_rejects_non_streaming_request():
    """stream_call requires stream:true; mismatch raises on first iteration."""
    client = _build_mock_streaming_client([])
    provider = AnthropicProvider(api_key="k", client=client)
    # Generator function returns a generator without running the body;
    # the ValueError fires on first iteration.
    gen = provider.stream_call({"model": "m", "messages": []})
    with pytest.raises(ValueError, match="stream: true"):
        next(gen)


# --- Happy-path SSE shape -------------------------------------------------


def test_stream_call_emits_initial_role_chunk():
    events = _build_event_sequence(["Hi."])
    client = _build_mock_streaming_client(events)
    provider = AnthropicProvider(api_key="k", client=client)
    chunks = list(
        provider.stream_call({"model": "m", "messages": [], "stream": True})
    )
    first = _parse_sse_chunk(chunks[0])
    assert first["object"] == "chat.completion.chunk"
    assert first["choices"][0]["delta"]["role"] == "assistant"
    assert first["choices"][0]["delta"]["content"] == ""
    assert first["choices"][0]["finish_reason"] is None


def test_stream_call_emits_content_delta_per_text_part():
    events = _build_event_sequence(["Hello ", "world", "."])
    client = _build_mock_streaming_client(events)
    provider = AnthropicProvider(api_key="k", client=client)
    chunks = list(
        provider.stream_call({"model": "m", "messages": [], "stream": True})
    )
    # [initial, delta1, delta2, delta3, finish, DONE] = 6 chunks
    assert len(chunks) == 6
    delta_chunks = [_parse_sse_chunk(c) for c in chunks[1:4]]
    contents = [c["choices"][0]["delta"]["content"] for c in delta_chunks]
    assert contents == ["Hello ", "world", "."]


def test_stream_call_terminates_with_done_marker():
    events = _build_event_sequence(["x"])
    client = _build_mock_streaming_client(events)
    provider = AnthropicProvider(api_key="k", client=client)
    chunks = list(
        provider.stream_call({"model": "m", "messages": [], "stream": True})
    )
    assert chunks[-1] == "data: [DONE]\n\n"


def test_stream_call_includes_finish_reason_in_penultimate_chunk():
    events = _build_event_sequence(["x"], stop_reason="end_turn")
    client = _build_mock_streaming_client(events)
    provider = AnthropicProvider(api_key="k", client=client)
    chunks = list(
        provider.stream_call({"model": "m", "messages": [], "stream": True})
    )
    # Chunk before [DONE] carries finish_reason.
    penult = _parse_sse_chunk(chunks[-2])
    assert penult["choices"][0]["finish_reason"] == "stop"
    assert penult["choices"][0]["delta"] == {}


def test_stream_call_maps_anthropic_stop_reasons_to_openai_finish_reasons():
    """Same _FINISH_REASON_MAP as the non-streaming path."""
    mappings = {
        "end_turn": "stop",
        "max_tokens": "length",
        "stop_sequence": "stop",
        "tool_use": "stop",
    }
    for anthropic_reason, openai_reason in mappings.items():
        events = _build_event_sequence(["x"], stop_reason=anthropic_reason)
        client = _build_mock_streaming_client(events)
        provider = AnthropicProvider(api_key="k", client=client)
        chunks = list(
            provider.stream_call(
                {"model": "m", "messages": [], "stream": True}
            )
        )
        penult = _parse_sse_chunk(chunks[-2])
        assert (
            penult["choices"][0]["finish_reason"] == openai_reason
        ), f"{anthropic_reason} should map to {openai_reason}"


def test_stream_call_shares_id_and_model_across_chunks():
    """id captured from message_start; model normalised by the openai/ strip."""
    events = _build_event_sequence(["a", "b"], msg_id="msg_xyz")
    client = _build_mock_streaming_client(events)
    provider = AnthropicProvider(api_key="k", client=client)
    chunks = list(
        provider.stream_call(
            {
                "model": "openai/claude-haiku-4-5",
                "messages": [],
                "stream": True,
            }
        )
    )
    parsed = [_parse_sse_chunk(c) for c in chunks[:-1]]
    ids = {c["id"] for c in parsed}
    models = {c["model"] for c in parsed}
    assert ids == {"msg_xyz"}
    assert models == {"claude-haiku-4-5"}


def test_stream_call_chunk_object_marker_is_completion_chunk():
    events = _build_event_sequence(["x"])
    client = _build_mock_streaming_client(events)
    provider = AnthropicProvider(api_key="k", client=client)
    chunks = list(
        provider.stream_call({"model": "m", "messages": [], "stream": True})
    )
    for c in chunks[:-1]:
        parsed = _parse_sse_chunk(c)
        assert parsed["object"] == "chat.completion.chunk"


# --- Guardrails ordering (streaming variant) ------------------------------


def test_stream_call_redacts_pii_before_upstream_call():
    """PDPL guardrails fire before the streaming upstream call (Y6 parity).

    Mirrors the existing test_guardrails_invoked_before_llm_call test for
    the non-streaming path. This is what G2 will verify end-to-end at the
    proof-gate stage.
    """
    events = _build_event_sequence(["x"])
    client = _build_mock_streaming_client(events)
    provider = AnthropicProvider(api_key="k", client=client)
    list(
        provider.stream_call(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "Email a@b.com"}],
                "stream": True,
            }
        )
    )
    kwargs = client.messages.create.call_args.kwargs
    sent_content = kwargs["messages"][0]["content"]
    assert "[EMAIL-REDACTED]" in sent_content
    assert "a@b.com" not in sent_content
    assert kwargs.get("stream") is True


# --- Error paths ----------------------------------------------------------


def test_stream_call_emits_error_chunk_on_anthropic_api_error():
    """Upstream Anthropic API error → SSE error chunk + [DONE]."""
    client = MagicMock()
    client.messages.create.side_effect = _MockAPIError("upstream went boom")
    provider = AnthropicProvider(api_key="k", client=client)
    chunks = list(
        provider.stream_call({"model": "m", "messages": [], "stream": True})
    )
    assert len(chunks) == 2
    error_chunk = _parse_sse_chunk(chunks[0])
    assert error_chunk["choices"][0]["finish_reason"] == "error"
    assert "upstream_error" in error_chunk["error"]["message"]
    assert chunks[-1] == "data: [DONE]\n\n"


def test_stream_call_emits_error_chunk_on_internal_exception():
    """Non-APIError exception → SSE internal_error chunk + [DONE]."""
    client = MagicMock()
    client.messages.create.side_effect = RuntimeError("internal bug")
    provider = AnthropicProvider(api_key="k", client=client)
    chunks = list(
        provider.stream_call({"model": "m", "messages": [], "stream": True})
    )
    assert len(chunks) == 2
    error_chunk = _parse_sse_chunk(chunks[0])
    assert error_chunk["choices"][0]["finish_reason"] == "error"
    assert "internal_error" in error_chunk["error"]["message"]
    assert chunks[-1] == "data: [DONE]\n\n"


def test_stream_call_closes_upstream_on_generator_exit():
    """Client disconnect closes upstream so Anthropic billing stops."""
    events = _build_event_sequence(["a", "b", "c", "d"])
    upstream = _MockUpstreamStream(events)
    client = MagicMock()
    client.messages.create.return_value = upstream
    provider = AnthropicProvider(api_key="k", client=client)
    gen = provider.stream_call(
        {"model": "m", "messages": [], "stream": True}
    )
    # Consume two chunks then close the generator (simulates client
    # disconnect mid-stream).
    next(gen)
    next(gen)
    gen.close()
    assert upstream.closed, "upstream was not closed on client disconnect"


# --- Non-text deltas are ignored (K036 forward-compat) --------------------


def test_stream_call_ignores_non_text_deltas():
    """input_json_delta events (tool-use; K036 territory) produce no SSE."""
    events = [
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(id="msg_x", role="assistant"),
        ),
        # Hypothetical tool-use input delta — no SSE emitted in v0.2.0.
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(
                type="input_json_delta", partial_json='{"a":'
            ),
        ),
        SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason="end_turn"),
        ),
        SimpleNamespace(type="message_stop"),
    ]
    client = _build_mock_streaming_client(events)
    provider = AnthropicProvider(api_key="k", client=client)
    chunks = list(
        provider.stream_call({"model": "m", "messages": [], "stream": True})
    )
    # Expect: initial role chunk + finish chunk + [DONE]. No content delta.
    assert len(chunks) == 3
    initial = _parse_sse_chunk(chunks[0])
    assert initial["choices"][0]["delta"] == {
        "role": "assistant",
        "content": "",
    }
