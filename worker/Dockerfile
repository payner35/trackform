# DJ Loop Worker container.
#
# Runs the analyzer plugins (Beat This!, Essentia, Demucs, MuQ-MuLan).
# In production this image is baked into a DO snapshot and only spawned
# when the job queue has work — see DEPLOYMENT_SPEC.md §3.3 / §9.
#
# In local dev this runs alongside the control plane via docker compose.
# CPU-only on Mac (Docker Desktop has no GPU passthrough); the same image
# uses CUDA when run on a real GPU droplet.
#
# Slice A: scaffolding only. Heavy ML deps land in Slice B.

FROM python:3.11-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        ca-certificates \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# `git` is needed because madmom installs from git master (PyPI release is
# broken on Python 3.10+ — LEARNINGS.md §2). `build-essential` covers the
# C/Cython compile step madmom runs at install time.

COPY --from=ghcr.io/astral-sh/uv:0.11.19 /uv /uvx /bin/

WORKDIR /app

# Layer ordering matters: deps install (slow, ~700 MB) must happen BEFORE
# COPY main.py so that source-code edits don't invalidate the install
# layer. Without this, every line of code change triggers a full reinstall.
COPY worker/pyproject.toml worker/uv.lock ./

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

RUN uv sync --frozen --no-dev --no-install-project

ENV PATH="/app/.venv/bin:$PATH"

# Source last — only this layer rebuilds on code-only changes.
COPY worker/main.py worker/plugin.py worker/run_madmom_subprocess.py ./
COPY worker/plugins ./plugins

CMD ["python", "main.py"]
