# syntax=docker/dockerfile:1
# ── build stage: install deps with uv ─────────────────────────────────────────
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv pip install --system --no-cache .[all]

# ── runtime stage ─────────────────────────────────────────────────────────────────
FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY . .
# Including the trained model pickle (git-ignored but required at runtime)
COPY ML/lightgbm_best.pkl /app/ML/lightgbm_best.pkl
# Ensuring runtime directories exist
RUN mkdir -p /app/PQ /app/ML
ENV PQ_DIR=/app/PQ
ENTRYPOINT ["python", "main.py"]
