"""kanoo-ai-core — Kanoo Elite AI abstraction package.

v0.1.0 ships:
- PDPL guardrails (request-side, regex-based)
- Anthropic provider via OpenAI-compatible adapter
- FastAPI server (POST /v1/chat/completions, GET /healthz)

Future modules per architecture doc §14 (RAG, text-to-SQL, vector store,
tenant isolation) land when downstream consumers materialise.
"""

__version__ = "0.1.0"
