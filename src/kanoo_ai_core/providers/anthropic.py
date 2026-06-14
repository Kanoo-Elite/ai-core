"""Anthropic provider — OpenAI Chat Completions ↔ Anthropic Messages adapter.

v0.2.0 scope:

- Non-streaming request flow (unchanged from v0.1.0): OpenAI Chat
  Completions request → PDPL guardrails → Anthropic Messages call →
  OpenAI Chat Completions response.
- **NEW: streaming request flow** — OpenAI Chat Completions request with
  ``stream: true`` → PDPL guardrails → Anthropic Messages streaming call
  → OpenAI Chat Completions SSE chunks (``chat.completion.chunk`` shape).
- **NEW: ``max_completion_tokens`` compatibility** (KAIROS-037) — reads
  modern OpenAI ``max_completion_tokens`` field with fallback to legacy
  ``max_tokens``. OpenClaw sends the modern field; v0.1.0 silently
  defaulted to 1024 because it only read the legacy one.
- No tool/function call support (translation does not forward
  ``tools`` or ``messages[*].tool_calls`` to Anthropic; deferred to
  KAIROS-036).
- No NER for named-individual redaction (deferred to KAIROS-034 /
  v0.3.0).
- No response-side redaction (deferred to KAIROS-035 / v0.4.0).
- No prompt-cache hints (separate capability; remains backlog).

Request translation:

- OpenAI's ``messages`` array has roles ``system | user | assistant``;
  Anthropic accepts ``user | assistant`` only with a separate top-level
  ``system`` field. System messages are concatenated and lifted to the
  Anthropic top-level system field.
- ``model`` passes through. The OpenAI request might say
  ``claude-haiku-4-5`` directly, or via OpenClaw be
  ``openai/claude-haiku-4-5`` — the ``openai/`` prefix is stripped here.
- ``max_tokens`` resolution: ``max_completion_tokens`` first (modern
  OpenAI; what OpenClaw actually sends), then legacy ``max_tokens``,
  defaulting to 1024.
- ``temperature``, ``top_p`` pass through.
- ``stop`` is converted to Anthropic's ``stop_sequences``.

Response translation (non-streaming):

- Anthropic's content blocks are concatenated into a single string for
  OpenAI's ``message.content``.
- ``stop_reason`` mapped to OpenAI ``finish_reason``: ``end_turn`` →
  ``stop``, ``max_tokens`` → ``length``, ``stop_sequence`` → ``stop``,
  ``tool_use`` → ``stop`` (v0.2.0 does not expose tool calls).
- Token counts mapped from ``input_tokens`` / ``output_tokens`` to
  ``prompt_tokens`` / ``completion_tokens`` / ``total_tokens``.

Streaming translation:

- Each Anthropic stream event maps to zero or more OpenAI
  ``chat.completion.chunk`` SSE chunks:

  * ``message_start`` → initial chunk with
    ``delta = {role: "assistant", content: ""}``; captures ``id``,
    ``created``, ``model`` for all subsequent chunks
  * ``content_block_delta`` (``text_delta`` only) → chunk with
    ``delta.content = event.delta.text``
  * ``message_delta`` → final chunk with ``finish_reason`` set via
    the existing ``_FINISH_REASON_MAP``
  * other event types (``content_block_start``, ``content_block_stop``,
    ``message_stop``) → no SSE output (they are framing only)

- Stream terminates with ``data: [DONE]\\n\\n`` per OpenAI convention.
- Mid-stream errors emit an OpenAI-shaped chunk with
  ``finish_reason: "error"`` and an additional ``error.message`` field,
  followed by ``[DONE]``.
- Client disconnect (``GeneratorExit``) closes the upstream Anthropic
  stream cleanly so Anthropic billing stops on cancellation.
"""

from __future__ import annotations

import json
import time
from collections.abc import Generator
from typing import Any

import anthropic
import structlog

from kanoo_ai_core.guardrails.pdpl import (
    redact_openai_request,
    security_log_event,
)

_log = structlog.get_logger("ai_core.provider.anthropic")


class AnthropicProvider:
    """Adapter from OpenAI Chat Completions to the Anthropic Messages API."""

    def __init__(
        self,
        api_key: str,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        """Initialise with an API key. ``client`` may be injected for tests."""
        if not api_key:
            raise ValueError("Anthropic API key is required.")
        self._client = client or anthropic.Anthropic(api_key=api_key)

    def call(self, openai_request: dict) -> dict:
        """Translate, guardrail, dispatch (non-streaming), and translate back.

        Args:
            openai_request: An OpenAI Chat Completions request payload.

        Returns:
            An OpenAI Chat Completions response payload.

        Raises:
            ValueError: If the request has ``stream: true``. Streaming
                requests must be routed to :meth:`stream_call` instead.
        """
        if openai_request.get("stream"):
            raise ValueError(
                "Streaming requests must be routed to stream_call(), "
                "not call()."
            )

        # PDPL guardrails BEFORE the LLM call.
        redaction = redact_openai_request(openai_request)
        model_id = _normalise_model(openai_request.get("model", ""))
        security_log_event(redaction, model_id=model_id)

        anthropic_request = _openai_to_anthropic(redaction.payload, model_id)
        anthropic_response = self._client.messages.create(**anthropic_request)
        return _anthropic_to_openai(anthropic_response, model_id=model_id)

    def stream_call(
        self, openai_request: dict
    ) -> Generator[str, None, None]:
        """Translate, guardrail, dispatch (streaming), and translate back as SSE.

        Yields OpenAI Chat Completions ``chat.completion.chunk`` SSE lines
        (``data: {...}\\n\\n``), terminating with ``data: [DONE]\\n\\n``.

        Args:
            openai_request: An OpenAI Chat Completions request payload with
                ``stream: true``.

        Yields:
            SSE-formatted strings ready to write to the wire.

        Raises:
            ValueError: If the request does NOT have ``stream: true``.
        """
        if not openai_request.get("stream"):
            raise ValueError(
                "stream_call() requires stream: true in the request. "
                "Non-streaming requests should use call()."
            )

        # PDPL guardrails BEFORE the LLM call (same path as call()).
        redaction = redact_openai_request(openai_request)
        model_id = _normalise_model(openai_request.get("model", ""))
        security_log_event(redaction, model_id=model_id)

        anthropic_request = _openai_to_anthropic(redaction.payload, model_id)

        state: dict[str, Any] = {
            "chunk_id": "",
            "created": int(time.time()),
            "model_id": model_id,
        }

        upstream_stream = None
        try:
            upstream_stream = self._client.messages.create(
                stream=True, **anthropic_request
            )
            for event in upstream_stream:
                yield from _anthropic_event_to_openai_sse(event, state)
            yield _sse_done()
        except anthropic.APIError as exc:
            _log.error(
                "stream_upstream_error",
                error_type=type(exc).__name__,
                model_id=model_id,
            )
            yield _sse_error_chunk(
                state, f"upstream_error: {type(exc).__name__}: {exc}"
            )
            yield _sse_done()
        except GeneratorExit:
            # Client disconnected — close upstream so Anthropic stops
            # billing on the abandoned completion. Do not yield further.
            if upstream_stream is not None:
                try:
                    upstream_stream.close()
                except Exception:
                    pass
            raise
        except Exception as exc:
            _log.exception("stream_internal_error", model_id=model_id)
            yield _sse_error_chunk(
                state, f"internal_error: {type(exc).__name__}: {exc}"
            )
            yield _sse_done()


# --- Helpers ----------------------------------------------------------------


def _normalise_model(model: str) -> str:
    """Strip the ``openai/`` prefix that OpenClaw's proxy mode injects."""
    if model.startswith("openai/"):
        return model[len("openai/"):]
    return model


def _resolve_max_tokens(payload: dict) -> int:
    """Resolve ``max_tokens`` with KAIROS-037 compatibility.

    Reads ``max_completion_tokens`` first (modern OpenAI; what OpenClaw
    actually sends), falls back to legacy ``max_tokens``, defaults to
    1024. Matches OpenAI's own deprecation behaviour.
    """
    return int(
        payload.get("max_completion_tokens")
        or payload.get("max_tokens")
        or 1024
    )


def _openai_to_anthropic(payload: dict, model_id: str) -> dict[str, Any]:
    """Convert an OpenAI Chat Completions request to Anthropic Messages args.

    Tools and ``messages[*].tool_calls`` are NOT forwarded to Anthropic
    in v0.2.0 (deferred to KAIROS-036). Other fields translate as in
    v0.1.0, with the K037 ``max_completion_tokens`` precedence applied
    via :func:`_resolve_max_tokens`.
    """
    messages_in = payload.get("messages", []) or []

    system_parts: list[str] = []
    user_messages: list[dict] = []

    for msg in messages_in:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        system_parts.append(part.get("text", ""))
        elif role in ("user", "assistant"):
            # Assistant tool_calls and tool-role messages will be
            # forwarded once KAIROS-036 lands; v0.2.0 passes content
            # through and drops the rest.
            user_messages.append({"role": role, "content": content or ""})

    args: dict[str, Any] = {
        "model": model_id,
        "max_tokens": _resolve_max_tokens(payload),
        "messages": user_messages,
    }
    if system_parts:
        args["system"] = "\n\n".join(system_parts)
    if payload.get("temperature") is not None:
        args["temperature"] = payload["temperature"]
    if payload.get("top_p") is not None:
        args["top_p"] = payload["top_p"]
    if payload.get("stop") is not None:
        stop_val = payload["stop"]
        args["stop_sequences"] = (
            [stop_val] if isinstance(stop_val, str) else list(stop_val)
        )

    return args


_FINISH_REASON_MAP: dict[str, str] = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "tool_use": "stop",
}


def _anthropic_to_openai(response: Any, *, model_id: str) -> dict:
    """Convert an Anthropic Messages response to OpenAI Chat Completions shape."""
    content_parts: list[str] = []
    for block in getattr(response, "content", []) or []:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            content_parts.append(text)
    content_str = "".join(content_parts)

    stop_reason = getattr(response, "stop_reason", None) or "end_turn"
    finish_reason = _FINISH_REASON_MAP.get(stop_reason, "stop")

    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
    output_tokens = getattr(usage, "output_tokens", 0) if usage else 0

    return {
        "id": getattr(response, "id", ""),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content_str},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


# --- Streaming helpers ------------------------------------------------------


def _anthropic_event_to_openai_sse(
    event: Any, state: dict[str, Any]
) -> list[str]:
    """Translate one Anthropic stream event to zero-or-more OpenAI SSE chunks.

    Updates ``state`` in place — ``chunk_id`` is captured on
    ``message_start`` and reused for every subsequent chunk in the same
    stream. The mapping is:

    - ``message_start`` → one chunk with ``delta = {role: "assistant",
      content: ""}``
    - ``content_block_delta`` with ``delta.type == "text_delta"`` → one
      chunk with ``delta.content = event.delta.text``
    - ``message_delta`` → one chunk with ``finish_reason`` set via the
      ``_FINISH_REASON_MAP`` lookup against ``event.delta.stop_reason``
    - all other event types → no SSE output (Anthropic framing only)
    """
    event_type = getattr(event, "type", None)

    if event_type == "message_start":
        message = getattr(event, "message", None)
        state["chunk_id"] = getattr(message, "id", "") if message else ""
        return [_sse_chunk(state, {"role": "assistant", "content": ""})]

    if event_type == "content_block_delta":
        delta = getattr(event, "delta", None)
        delta_type = getattr(delta, "type", None) if delta else None
        # v0.2.0 emits text deltas only; tool-use deltas (input_json_delta)
        # belong to KAIROS-036.
        if delta_type == "text_delta":
            text = getattr(delta, "text", "")
            return [_sse_chunk(state, {"content": text})]
        return []

    if event_type == "message_delta":
        delta = getattr(event, "delta", None)
        stop_reason = getattr(delta, "stop_reason", None) if delta else None
        stop_reason = stop_reason or "end_turn"
        finish_reason = _FINISH_REASON_MAP.get(stop_reason, "stop")
        return [_sse_chunk(state, {}, finish_reason=finish_reason)]

    # content_block_start, content_block_stop, message_stop, and any
    # unrecognised event types produce no SSE output (they are framing
    # only — the caller emits the terminal ``data: [DONE]`` itself when
    # the upstream iteration completes).
    return []


def _sse_chunk(
    state: dict[str, Any],
    delta: dict[str, Any],
    finish_reason: str | None = None,
) -> str:
    """Format a single OpenAI ``chat.completion.chunk`` SSE line."""
    payload = {
        "id": state["chunk_id"],
        "object": "chat.completion.chunk",
        "created": state["created"],
        "model": state["model_id"],
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _sse_error_chunk(state: dict[str, Any], error_message: str) -> str:
    """Format an OpenAI-shaped error chunk for mid-stream failures.

    Includes ``finish_reason: "error"`` plus an additional ``error.message``
    field. OpenClaw's openai provider plugin parses this as a terminal
    chunk with an error indication.
    """
    payload = {
        "id": state.get("chunk_id", ""),
        "object": "chat.completion.chunk",
        "created": state["created"],
        "model": state["model_id"],
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "error",
            }
        ],
        "error": {"message": error_message},
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _sse_done() -> str:
    """Final SSE line per OpenAI streaming convention."""
    return "data: [DONE]\n\n"
