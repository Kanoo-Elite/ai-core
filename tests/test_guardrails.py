"""X2 proof gate — PDPL guardrails unit tests."""

from __future__ import annotations

from kanoo_ai_core.guardrails.pdpl import (
    KNOWN_REDACTION_TYPES,
    PdplRedactionResult,
    redact_openai_request,
    security_log_event,
)

# --- Positive cases per pattern -------------------------------------------


def test_redact_email_in_user_string_content():
    payload = {"messages": [{"role": "user", "content": "Email me at a@b.com"}]}
    result = redact_openai_request(payload)
    assert "[EMAIL-REDACTED]" in result.payload["messages"][0]["content"]
    assert "a@b.com" not in result.payload["messages"][0]["content"]
    assert result.counts == {"email": 1}


def test_redact_e164_phone():
    payload = {"messages": [{"role": "user", "content": "Call +966512345678"}]}
    result = redact_openai_request(payload)
    assert "[PHONE-REDACTED]" in result.payload["messages"][0]["content"]
    assert "+966512345678" not in result.payload["messages"][0]["content"]
    assert result.counts == {"phone": 1}


def test_redact_phone_with_dashes():
    payload = {"messages": [{"role": "user", "content": "Call 555-123-4567"}]}
    result = redact_openai_request(payload)
    assert "[PHONE-REDACTED]" in result.payload["messages"][0]["content"]
    assert result.counts == {"phone": 1}


def test_redact_ksa_national_id():
    payload = {"messages": [{"role": "user", "content": "My NID: 1234567890"}]}
    result = redact_openai_request(payload)
    assert "[NATIONAL-ID-REDACTED]" in result.payload["messages"][0]["content"]
    assert "1234567890" not in result.payload["messages"][0]["content"]
    assert result.counts == {"ksa_nid": 1}


def test_redact_bahraini_national_id():
    payload = {"messages": [{"role": "user", "content": "BH NID 080123456"}]}
    result = redact_openai_request(payload)
    assert "[NATIONAL-ID-REDACTED]" in result.payload["messages"][0]["content"]
    assert result.counts == {"bh_nid": 1}


# --- Negative cases -------------------------------------------------------


def test_no_redaction_when_no_pii():
    original = {"messages": [{"role": "user", "content": "Hello world"}]}
    result = redact_openai_request(original)
    assert result.payload == original
    assert result.counts == {}


def test_arbitrary_9_digit_does_not_match_bh_nid():
    """BH NID requires the year-prefix shape; 999000000 (starts with 9) excluded."""
    payload = {"messages": [{"role": "user", "content": "Ref 999000000"}]}
    result = redact_openai_request(payload)
    assert result.counts == {}


def test_9_digit_starting_with_3_does_not_match_bh_nid():
    """30-99 prefixes are out of the v0.1.0 year-prefix range."""
    payload = {"messages": [{"role": "user", "content": "Ref 350000000"}]}
    result = redact_openai_request(payload)
    assert result.counts == {}


# --- Multi-message / system / multi-part content -------------------------


def test_redact_across_system_and_user_messages():
    payload = {
        "messages": [
            {"role": "system", "content": "Contact ops@kanooelite.com"},
            {"role": "user", "content": "Email me at customer@example.com"},
        ]
    }
    result = redact_openai_request(payload)
    assert result.counts == {"email": 2}


def test_redact_multipart_content():
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Phone: +966512345678"},
                    {"type": "image_url", "image_url": "http://example.com/foo.png"},
                ],
            }
        ]
    }
    result = redact_openai_request(payload)
    parts = result.payload["messages"][0]["content"]
    assert parts[0]["text"] == "Phone: [PHONE-REDACTED]"
    assert parts[1] == {"type": "image_url", "image_url": "http://example.com/foo.png"}
    assert result.counts == {"phone": 1}


def test_redact_top_level_system_string():
    payload = {"system": "Operator: ops@kanooelite.com", "messages": []}
    result = redact_openai_request(payload)
    assert "[EMAIL-REDACTED]" in result.payload["system"]
    assert result.counts == {"email": 1}


def test_pure_function_does_not_mutate_input():
    payload = {"messages": [{"role": "user", "content": "Email a@b.com"}]}
    original = {"messages": [{"role": "user", "content": "Email a@b.com"}]}
    redact_openai_request(payload)
    assert payload == original


# --- Canonical fixture -----------------------------------------------------


def test_canonical_fixture_redaction_counts(pii_canonical_payload):
    """Counts vector matches the X6 proof gate spec."""
    result = redact_openai_request(pii_canonical_payload)
    assert result.counts == {"email": 1, "phone": 1, "ksa_nid": 1}
    assert "bh_nid" not in result.counts  # no BH NID in fixture


# --- security_log_event ----------------------------------------------------


def test_security_log_event_shape():
    result = PdplRedactionResult(payload={}, counts={"email": 2, "phone": 1})
    event = security_log_event(result, model_id="claude-haiku-4-5")
    assert event["event"] == "security_log"
    assert event["timestamp"].endswith("+00:00")
    assert event["model_id"] == "claude-haiku-4-5"
    assert event["ner_enabled"] is False
    assert event["response_redaction_enabled"] is False
    assert event["request_id"] == result.request_id


def test_security_log_event_includes_all_known_types_with_zero_defaults():
    """The schema is stable: all KNOWN_REDACTION_TYPES present in every event."""
    result = PdplRedactionResult(payload={}, counts={"email": 2})
    event = security_log_event(result, model_id="m")
    counts = event["redaction_count_by_type"]
    for kind in KNOWN_REDACTION_TYPES:
        assert kind in counts
    assert counts["email"] == 2
    assert counts["phone"] == 0
    assert counts["ksa_nid"] == 0
    assert counts["bh_nid"] == 0


def test_security_log_event_never_contains_original_text():
    """Counts only; no text payload in the event."""
    result = PdplRedactionResult(
        payload={"sensitive": "info"},
        counts={"email": 1},
    )
    event = security_log_event(result, model_id="m")
    serialised = str(event)
    assert "sensitive" not in serialised
    assert "info" not in serialised
