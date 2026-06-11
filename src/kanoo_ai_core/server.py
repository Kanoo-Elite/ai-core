"""FastAPI server — OpenAI-compatible ``POST /v1/chat/completions`` + ``/healthz``.

Designed to be consumed by OpenClaw's ``openai`` provider with
``OPENAI_BASE_URL`` pointing at this service. See the architecture doc §07
and KAIROS-020 sub-decision D for the rationale.

The Anthropic API key is loaded from GCP Secret Manager at process startup
via ADC. Rotation requires container restart.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from kanoo_shared_libs import (
    SecretAccessDeniedError,
    SecretNotFoundError,
    get_secret,
)

from kanoo_ai_core import __version__
from kanoo_ai_core.providers.anthropic import AnthropicProvider

_log = structlog.get_logger("ai_core.server")


def _load_anthropic_key() -> str:
    """Fetch the Anthropic API key from GCP Secret Manager via ADC."""
    project = os.environ.get("KAIROS_GCP_PROJECT", "kanoo-kairos-dev")
    secret_id = os.environ.get("KAIROS_ANTHROPIC_SECRET_ID", "anthropic-api-key")
    try:
        return get_secret(secret_id, project_id=project)
    except SecretNotFoundError:
        _log.error(
            "anthropic_key_not_found", project=project, secret_id=secret_id
        )
        raise
    except SecretAccessDeniedError:
        _log.error(
            "anthropic_key_access_denied", project=project, secret_id=secret_id
        )
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load secrets + wire provider at startup; tear down on shutdown."""
    api_key = _load_anthropic_key()
    app.state.anthropic_provider = AnthropicProvider(api_key=api_key)
    app.state.anthropic_key_loaded = True
    _log.info("startup_complete", version=__version__)
    yield
    _log.info("shutdown")


app = FastAPI(
    title="kanoo-ai-core",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/healthz")
async def healthz(request: Request) -> dict[str, Any]:
    return {
        "status": "ok",
        "version": __version__,
        "anthropic_key_loaded": getattr(
            request.app.state, "anthropic_key_loaded", False
        ),
        "ner_enabled": False,
        "response_redaction_enabled": False,
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> JSONResponse:
    payload = await request.json()
    provider: AnthropicProvider = request.app.state.anthropic_provider
    try:
        result = provider.call(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(content=result)
