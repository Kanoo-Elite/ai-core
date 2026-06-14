"""FastAPI server — OpenAI-compatible ``POST /v1/chat/completions`` + ``/healthz``.

v0.2.0: streaming endpoint added (returns SSE when ``stream: true`` is
present in the request body). Non-streaming behaviour unchanged from
v0.1.0. NER and response-side redaction remain deferred (KAIROS-034 /
KAIROS-035 respectively).

Designed to be consumed by OpenClaw's ``openai`` provider with
``OPENAI_BASE_URL`` pointing at this service. See the architecture doc §07
and KAIROS-020 sub-decision D for the rationale; KAIROS-032 for the
streaming addition.

The Anthropic API key is loaded from GCP Secret Manager at process startup
via ADC. Rotation requires container restart.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
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
    """Liveness + capability flags. v0.2.0 adds ``streaming_enabled``."""
    return {
        "status": "ok",
        "version": __version__,
        "anthropic_key_loaded": getattr(
            request.app.state, "anthropic_key_loaded", False
        ),
        "streaming_enabled": True,
        "ner_enabled": False,
        "response_redaction_enabled": False,
    }


# SSE response headers (per OpenAI streaming convention).
# - ``text/event-stream`` is the standard SSE media type.
# - ``Cache-Control: no-cache`` prevents intermediary caches.
# - ``X-Accel-Buffering: no`` disables Nginx/proxy buffering if any
#   reverse proxy ever sits in front of ai-core (none today, but
#   defensive and zero-cost).
_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    """OpenAI Chat Completions endpoint with dual non-streaming/streaming dispatch.

    - ``stream: false`` (or absent) → returns a JSONResponse with the
      translated Anthropic Messages response in OpenAI Chat Completions
      shape (unchanged from v0.1.0).
    - ``stream: true`` → returns a StreamingResponse emitting OpenAI
      ``chat.completion.chunk`` SSE lines, terminating with
      ``data: [DONE]\\n\\n``.
    """
    payload = await request.json()
    provider: AnthropicProvider = request.app.state.anthropic_provider

    if payload.get("stream"):
        # Streaming path. Note: provider.stream_call returns a generator;
        # its body (including PDPL redaction + the upstream Anthropic call)
        # only starts running when iteration begins inside StreamingResponse.
        # Any ValueError from stream_call's internal check would surface
        # mid-iteration, not here — but we already guard against the
        # mismatch case (stream:true here, stream:false in stream_call)
        # by routing only when the payload says stream:true.
        try:
            stream_iter = provider.stream_call(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return StreamingResponse(
            stream_iter,
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )

    # Non-streaming path (unchanged from v0.1.0).
    try:
        result = provider.call(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(content=result)
