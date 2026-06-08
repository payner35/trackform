# syntax=docker/dockerfile:1.7

# DJ Loop Service container.
#
# Single-stage Python 3.11 image. uv handles dependency resolution from
# pyproject.toml + uv.lock so the host's Python/architecture is irrelevant —
# the container runs the same Linux Python on any host (M-series Mac, Intel
# Mac, Linux server, Fly.io, Railway, etc.).
#
# System deps:
#   - ffmpeg     fallback audio decoder (some MP3s need it)
#   - libchromaprint-tools  provides `fpcalc` for pyacoustid fingerprinting
#   - libsndfile1           soundfile uses this to read audio headers

FROM python:3.11-slim-bookworm

# Install system audio tools. Pin if reproducibility becomes important.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libchromaprint-tools \
        libsndfile1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv (Astral's Python project manager). Pinning the version keeps
# image builds reproducible.
COPY --from=ghcr.io/astral-sh/uv:0.11.19 /uv /uvx /bin/

WORKDIR /app

# Copy dependency manifests first so `uv sync` is cached separately from
# source changes (faster rebuilds during dev).
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

# Install into a project-local .venv. UV_LINK_MODE=copy avoids hardlink
# issues across bind mounts; UV_COMPILE_BYTECODE speeds up first-run.
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

RUN uv sync --frozen --no-dev

# Put the venv on PATH so the `service` entry point is callable directly.
ENV PATH="/app/.venv/bin:${PATH}"

# Default: print help. docker-compose.yml overrides this for real runs.
ENTRYPOINT ["service"]
CMD ["--help"]
