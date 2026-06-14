# kanoo-ai-core

Kanoo Elite AI abstraction package — PDPL-guardrailed LLM provider
abstraction.

## v0.2.0 — what's in here

- **PDPL guardrails** (request-side, regex-based) — `kanoo_ai_core.guardrails.pdpl`
  - Redaction patterns: email, Gulf phone, KSA NID, BH NID (unchanged from v0.1.0)
  - `security_log` event emitted via `structlog` → stdout for every call
- **Anthropic provider** — `kanoo_ai_core.providers.anthropic.AnthropicProvider`
  - OpenAI Chat Completions ↔ Anthropic Messages API translation
  - **NEW in v0.2.0:** streaming support via `stream_call()` — SSE chunks
    in OpenAI `chat.completion.chunk` shape; client-disconnect cancels
    the upstream Anthropic stream cleanly (KAIROS-032)
  - **NEW in v0.2.0:** `max_completion_tokens` precedence over legacy
    `max_tokens` — OpenClaw sends the modern field; v0.1.0 silently
    defaulted to 1024 (KAIROS-037)
  - No tool/function call translation or forwarding yet (deferred to
    KAIROS-036)
- **FastAPI server** — `kanoo_ai_core.server`
  - `POST /v1/chat/completions` — OpenAI-compatible. Dispatches to
    `StreamingResponse` (SSE) when `stream: true`, `JSONResponse` otherwise.
  - `GET /healthz` — liveness + capability flags (`streaming_enabled: true`
    added in v0.2.0)

Future modules per architecture doc §14 (RAG, text-to-SQL, vector store,
tenant isolation) land when downstream consumers materialise.

## Network posture

The ai-core container needs outbound access to:

- `secretmanager.googleapis.com:443` — Anthropic API key fetch via ADC
- `metadata.google.internal` — ADC token acquisition (on GCE)
- `api.anthropic.com:443` — actual LLM call

v0.1.0 runs on Docker default bridge with VM-level NAT outbound. A dedicated
Squid egress proxy for ai-core is filed as a Year-2 backlog item (see
KAIROS-020 sub-decision G).

The ai-core container does **not** authenticate inbound requests in v0.1.0
— it trusts the internal `ai-bridge` Docker network. v0.2.0+ adds bearer
token auth if/when ai-core is exposed beyond a single VM.

## Environment

| Variable | Default | Description |
|---|---|---|
| `KAIROS_GCP_PROJECT` | `kanoo-kairos-dev` | Project hosting the Anthropic API key secret |
| `KAIROS_ANTHROPIC_SECRET_ID` | `anthropic-api-key` | Secret Manager secret ID |

The Anthropic API key is fetched at process startup and held in memory.
Rotation requires container restart in v0.1.0.

## Development

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -v
ruff check .
```

## Docker

```bash
docker build -t kanoo/ai-core:0.1.0 .
docker run --rm -p 8000:8000 \
  -e KAIROS_GCP_PROJECT=kanoo-kairos-dev \
  kanoo/ai-core:0.1.0
```

ADC: on a GCE VM with an attached service account, the container picks up
credentials via the metadata server. On a developer workstation, mount
your local ADC: `-v ~/.config/gcloud:/root/.config/gcloud:ro`.

## Limitations explicitly accepted in v0.2.0

- No NER for named-individual redaction — deferred to **KAIROS-034 /
  v0.3.0**. Every `security_log` entry carries `ner_enabled: false`
  so the limitation is grep-able.
- No response-side redaction — deferred to **KAIROS-035 / v0.4.0**.
  Decoupled from streaming per K032 sub-decision J (the original v0.1.0
  README framed it as "paired with streaming"; K032 reversed that).
- No tool/function call translation or forwarding — deferred to
  **KAIROS-036** (conditional trigger: openclaw agent-mode flows start
  sending tool defs on the wire).
- No prompt-cache hints — separate capability with its own design
  surface (cache key shape, TTL, invalidation). Remains backlog; no
  specific version target.
- Inbound auth: none (internal `ai-bridge` trust)

### Changed in v0.2.0

- **Streaming** — `stream: true` in the request body now returns SSE
  (OpenAI `chat.completion.chunk` shape) instead of HTTP 400 (KAIROS-032).
- **`max_completion_tokens`** — modern OpenAI field is now read with
  precedence over legacy `max_tokens` (KAIROS-037).

## License

MIT — see [`LICENSE`](./LICENSE).

## Decision record

Bootstrapped under **KAIROS-020 β3**. Full Active entry (sub-decisions
A–J, three-pass lens reasoning, proof gates X1–X7) in
[kairos-platform/decisions-pending.md](https://github.com/Kanoo-Elite/kairos-platform/blob/main/decisions-pending.md).
