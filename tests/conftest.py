"""Shared pytest fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def pii_canonical_payload() -> dict:
    """Canonical PII fixture shaped as an OpenAI Chat Completions request."""
    return json.loads((FIXTURES_DIR / "pii_canonical.json").read_text("utf-8"))
