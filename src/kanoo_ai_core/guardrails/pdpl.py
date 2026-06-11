"""PDPL guardrails — request-side PII redaction for outbound LLM calls.

v0.1.0 scope: redact PII patterns in an OpenAI Chat Completions request
payload (across ``messages[].content`` for both string and list-of-parts
content, plus an optional top-level ``system`` field). Each redaction is
counted and logged to a structured ``security_log`` event — never the
redacted-or-original text.

Coverage in v0.1.0:

- Emails
- Phone numbers (E.164 + common Gulf/regional shapes)
- Saudi National ID (10 digits starting with 1 or 2)
- Bahraini National ID (9-digit pattern, year-prefix-constrained)

NER (named individuals) deferred to v0.2.0; every ``security_log`` event
carries ``ner_enabled: false`` so the limitation is grep-able. Response-side
redaction also deferred to v0.2.0 (paired with streaming).
"""

from __future__ import annotations

import copy
import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

import structlog

_log = structlog.get_logger("ai_core.guardrails.pdpl")


# Pattern order matters: NID first (most specific), phone, email.
# Anchored with \b on both sides where applicable to reduce false positives.

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")

_PHONE_RE = re.compile(
    r"(?:"
    r"\+\d{8,15}"
    r"|"
    r"\b\d{3}[-.\s]\d{3,4}[-.\s]\d{4}\b"
    r")"
)

_KSA_NID_RE = re.compile(r"\b[12]\d{9}\b")

# Bahraini NID is 9 digits, commonly starting with the last two digits of the
# birth year. Constrain to 00–25 prefix (covers 1900–2025 births) to reduce
# false positives on arbitrary 9-digit numbers.
_BH_NID_RE = re.compile(r"\b(?:0\d|1\d|2[0-5])\d{7}\b")

KNOWN_REDACTION_TYPES: tuple[str, ...] = ("email", "phone", "ksa_nid", "bh_nid")

_REPLACEMENTS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("ksa_nid", _KSA_NID_RE, "[NATIONAL-ID-REDACTED]"),
    ("bh_nid", _BH_NID_RE, "[NATIONAL-ID-REDACTED]"),
    ("phone", _PHONE_RE, "[PHONE-REDACTED]"),
    ("email", _EMAIL_RE, "[EMAIL-REDACTED]"),
)


@dataclass
class PdplRedactionResult:
    """Outcome of a single redaction pass over a request payload."""

    payload: dict
    counts: dict[str, int] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def total(self) -> int:
        return sum(self.counts.values())


# --- Internal helpers -------------------------------------------------------


def _redact_text(text: str, counts: dict[str, int]) -> str:
    redacted = text
    for kind, pattern, replacement in _REPLACEMENTS:
        new_redacted, n = pattern.subn(replacement, redacted)
        if n:
            counts[kind] = counts.get(kind, 0) + n
            redacted = new_redacted
    return redacted


def _redact_content(content: object, counts: dict[str, int]) -> object:
    """Redact ``messages[*].content`` which may be a str or a list of parts."""
    if isinstance(content, str):
        return _redact_text(content, counts)
    if isinstance(content, list):
        new_parts: list[object] = []
        for part in content:
            if (
                isinstance(part, dict)
                and part.get("type") == "text"
                and isinstance(part.get("text"), str)
            ):
                new_part = dict(part)
                new_part["text"] = _redact_text(part["text"], counts)
                new_parts.append(new_part)
            else:
                new_parts.append(part)
        return new_parts
    return content


# --- Public API -------------------------------------------------------------


def redact_openai_request(payload: dict) -> PdplRedactionResult:
    """Apply PDPL request-side redaction to an OpenAI Chat Completions payload.

    Pure function: the input ``payload`` is not mutated. The returned payload
    is a deep copy with PII replaced by sentinel tokens.

    Args:
        payload: A dict shaped like an OpenAI Chat Completions request.
            ``messages`` (a list of role/content dicts) is the primary
            target. The optional top-level ``system`` field is also redacted
            if present.

    Returns:
        :class:`PdplRedactionResult` with a fresh ``request_id``, the
        redacted payload, and per-pattern counts.
    """
    redacted_payload = copy.deepcopy(payload)
    counts: dict[str, int] = {}

    for msg in redacted_payload.get("messages", []) or []:
        if isinstance(msg, dict) and "content" in msg:
            msg["content"] = _redact_content(msg["content"], counts)

    if isinstance(redacted_payload.get("system"), str):
        redacted_payload["system"] = _redact_text(redacted_payload["system"], counts)

    return PdplRedactionResult(payload=redacted_payload, counts=counts)


def security_log_event(
    result: PdplRedactionResult,
    *,
    model_id: str,
) -> dict:
    """Construct and emit the canonical ``security_log`` event for a redaction.

    Emitted via :mod:`structlog` at INFO level on the
    ``ai_core.guardrails.pdpl`` channel. The event contains **only counts**
    — never redacted-or-original text. Returns the event dict for callers
    that wish to inspect or persist it.

    The ``redaction_count_by_type`` field always includes all known types
    (with ``0`` for types not observed in this call) so the schema is stable
    for downstream consumers.
    """
    counts = {t: result.counts.get(t, 0) for t in KNOWN_REDACTION_TYPES}
    event = {
        "event": "security_log",
        "timestamp": datetime.now(UTC).isoformat(),
        "request_id": result.request_id,
        "redaction_count_by_type": counts,
        "model_id": model_id,
        "ner_enabled": False,
        "response_redaction_enabled": False,
    }
    _log.info("security_log", **{k: v for k, v in event.items() if k != "event"})
    return event
