# syntax=docker/dockerfile:1

# Based on LiveKit's uv template (docs.livekit.io/deploy/agents/builds), with
# three deltas for this project: Python 3.12 (pyproject requires >=3.12), a
# two-step uv sync because this is a src-layout hatchling package, and a CMD
# that launches the `collector-voice` console script rather than agent.py.
ARG PYTHON_VERSION=3.12
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS base

ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1

# --- Build stage ---
FROM base AS build

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, for layer caching. --no-install-project is required
# here: the package itself lives under src/, which hasn't been copied yet,
# so a plain `uv sync` would fail trying to build `collector`.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project

COPY . .

# Now that src/ is present, install the project itself — this is what puts
# the `collector-voice` entry point on the venv's PATH.
RUN uv sync --locked

# --- Production stage ---
FROM base

ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/app" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser

WORKDIR /app

COPY --from=build --chown=appuser:appuser /app /app

USER appuser

# `start` is the production worker mode (dev/console are for local runs).
# AuditStore creates its own data/ directory relative to this WORKDIR.
CMD ["uv", "run", "collector-voice", "start"]
