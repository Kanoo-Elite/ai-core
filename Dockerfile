# syntax=docker/dockerfile:1
# Multi-stage build: install deps in builder, copy minimal runtime layer.

FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --target=/opt/ai-core/lib .


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/ai-core/lib

# Non-root runtime user (UID 1000 matches other Kanoo Elite containers).
RUN groupadd --system --gid 1000 ai \
    && useradd --system --uid 1000 --gid 1000 \
       --no-create-home --shell /usr/sbin/nologin ai

COPY --from=builder /opt/ai-core/lib /opt/ai-core/lib

USER 1000:1000
WORKDIR /opt/ai-core

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "kanoo_ai_core.server:app", "--host", "0.0.0.0", "--port", "8000"]
