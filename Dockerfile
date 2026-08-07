# syntax=docker/dockerfile:1

# Based on LiveKit's uv template (docs.livekit.io/deploy/agents/builds), with
# three deltas for this project: the Python version, a two-step uv sync because
# this is a src-layout hatchling package, and a CMD that launches the
# `collector-voice` console script rather than agent.py.
#
# This ARG must satisfy pyproject's `requires-python`, and the guard below is
# why. When it does not, uv ignores the image's interpreter and downloads a
# managed one *as root* into /root/.local/share/uv/python. The build succeeds,
# `.venv/bin/python` symlinks in there, and the image then dies on first start
# under the non-root user this stage creates:
#
#   failed to canonicalize path `/app/.venv/bin/python3`: Permission denied
#
# because /root is 0700. `UV_PYTHON_DOWNLOADS=never` turns that runtime
# crashloop into a build failure on the line that causes it.
ARG PYTHON_VERSION=3.14
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS base

ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_PYTHON_DOWNLOADS=never

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

# The audit log — verbatim transcripts of debt-collection calls — lives here.
# AuditStore creates the directory 0700 and every file in it 0600, and refuses
# to start if it cannot: the container runs as UID 10001, so a volume mounted
# with a different owner fails loudly rather than writing readable transcripts.
# Set COLLECTOR_DB_PATH to move the log off the container filesystem.
#
# THE MOUNTED VOLUME MUST BE ON AN ENCRYPTED FILESYSTEM (dm-crypt/LUKS, an
# encrypted EBS volume, or the equivalent). 0600 is an access control between
# accounts on one host and nothing more: it does not protect the data from
# root, from a stolen disk, from a snapshot, or from a backup. There is no
# application-level encryption in this build — that decision was deliberate,
# and this line is the other half of it.
#
# Deliberately no VOLUME instruction: an anonymous volume comes up root-owned,
# and this image runs as UID 10001, so the directory chmod would fail on every
# start. Mount the encrypted volume at /app/data with `docker run -v`, owned by
# UID 10001.

# `start` is the production worker mode (dev/console are for local runs).
CMD ["uv", "run", "collector-voice", "start"]
