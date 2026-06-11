"""Anthropic provider — OpenAI Chat Completions ↔ Anthropic Messages adapter.

v0.1.0 scope:

- Single request flow: OpenAI Chat Completions request → PDPL guardrails →
  Anthropic Messages call → OpenAI Chat Completions response.
- Non-streaming only.
- No tool/function call support.

Request translation:

- OpenAI's ``messages`` array has roles ``system | user | assistant``;
  Anthropic accepts ``user | assistant`` only with a separate top-level
  ``system`` field. System messages are concatenated and lifted to the
  Anthropic top-level system field.
- ``model`` passes through. The OpenAI request might say
  ``claude-haiku-4-5`` directly, or via OpenClaw be
  ``openai/claude-haiku-4-5`` — the ``openai/`` prefix is stripped here.
- ``max_tokens`` defaults to 1024 if absent (Anthropic requires it).
- ``temperature``, ``top_p`` pass through.
- ``stop`` is converted to Anthropic's ``stop_sequences``.
- ``stream: true`` is rejected with :class:`ValueError` in v0.1.0.

Response translation:

- Anthropic's content blocks are concatenated into a single string for
  OpenAI's ``message.content``.
- ``stop_reason`` mapped to OpenAI ``finish_reason``:
  ``end_turn`` → ``stop``, ``max_tokens`` → ``length``,
  ``stop_sequence`` → ``stop``, ``tool_use`` → ``stop`` (v0.1.0 does not
  expose tool calls).
- Token counts mapped from ``input_tokens`` / ``output_tokens`` to
  ``prompt_tokens`` / ``completion_tokens`` / ``total_tokens``.
"""

from __future__ import annotations

import time
from typing import Any

import anthropic

from kanoo_ai_core.guardrails.pdpl import (
    redact_openai_request,
    security_log_event,
)


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
        """Translate, guardrail, dispatch, and translate back.

        Args:
            openai_request: An OpenAI Chat Completions request payload.

        Returns:
            An OpenAI Chat Completions response payload.

        Raises:
            ValueError: For unsupported inputs (e.g. ``stream: true``).
        """
        if openai_request.get("stream"):
            raise ValueError(
                "Streaming responses are not supported in ai-core v0.1.0."
            )

        # PDPL guardrails BEFORE the LLM call.
        redaction = redact_openai_request(openai_request)
        model_id = _normalise_model(openai_request.get("model", ""))
        security_log_event(redaction, model_id=model_id)

        anthropic_request = _openai_to_anthropic(redaction.payload, model_id)
        anthropic_response = self._client.messages.create(**anthropic_request)
        return _anthropic_to_openai(anthropic_response, model_id=model_id)


# --- Helpers ----------------------------------------------------------------


def _normalise_model(model: str) -> str:
    """Strip the ``openai/`` prefix that OpenClaw's proxy mode injects."""
    if model.startswith("openai/"):
        return model[len("openai/"):]
    return model


def _openai_to_anthropic(payload: dict, model_id: str) -> dict[str, Any]:
    """Convert an OpenAI Chat Completions request to Anthropic Messages args."""
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
            user_messages.append({"role": role, "content": content})

    args: dict[str, Any] = {
        "model": model_id,
        "max_tokens": int(payload.get("max_tokens") or 1024),
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
