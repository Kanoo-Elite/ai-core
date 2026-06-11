# kanoo-ai-core

Kanoo Elite AI abstraction package — PDPL-guardrailed LLM provider
abstraction.

## v0.1.0 — what's in here

- **PDPL guardrails** (request-side, regex-based) — `kanoo_ai_core.guardrails.pdpl`
  - Redaction patterns: email, Gulf phone, KSA NID, BH NID
  - `security_log` event emitted via `structlog` → stdout for every call
- **Anthropic provider** — `kanoo_ai_core.providers.anthropic.AnthropicProvider`
  - OpenAI Chat Completions ↔ Anthropic Messages API translation
  - Non-streaming only in v0.1.0; no tool calls
- **FastAPI server** — `kanoo_ai_core.server`
  - `POST /v1/chat/completions` — OpenAI-compatible (consumed by OpenClaw's
    `openai` provider with `OPENAI_BASE_URL` pointing here)
  - `GET /healthz` — liveness + flag status

Future modules per architecture doc §14 (RAG, text-to-SQL, vector store,
tenant isolation) land when downstream consumers materialise.

## Network posture (v0.1.0)

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

## Limitations explicitly accepted in v0.1.0

- No streaming (SSE deferred to v0.2.0; paired with response-side redaction)
- No NER for named-individual redaction (every `security_log` entry carries
  `ner_enabled: false` so the limitation is grep-able)
- No response-side redaction
- No tool/function call support
- No prompt-cache hints (lost via the proxy-mode pattern; restored in
  v0.2.0 via direct `anthropic-beta` header injection)
- Inbound auth: none (internal `ai-bridge` trust)

## License

MIT — see [`LICENSE`](./LICENSE).

## Decision record

Bootstrapped under **KAIROS-020 β3**. Full Active entry (sub-decisions
A–J, three-pass lens reasoning, proof gates X1–X7) in
[kairos-platform/decisions-pending.md](https://github.com/Kanoo-Elite/kairos-platform/blob/main/decisions-pending.md).
