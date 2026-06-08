# DJ Loop Service

Analysis (and eventually realtime) service for DJ Loop Player.

This is the **Service** side of the Player ↔ Service architectural split.
The Player plays audio. The Service analyzes audio and writes metadata into
the shared `library.db`. See the Player project's `docs/service/OVERVIEW.md`
for the full vision.

## Status

Phase 1 — minimum end-to-end analyzer:
- Project scaffold with `uv` and Python 3.11
- Plugin framework (`@analyzer_stage` decorator, pipeline orchestrator)
- `core.load` plugin: librosa load + mutagen tag extraction → `content` table
- STUBS: `core.beats_madmom`, `core.key`, `core.sections_msaf`,
  `core.loop_mining`, `core.loop_features_basic`
- FUTURE: watch mode, WebSocket realtime plane, CLAP/mood embeddings

## Install

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11 (uv will fetch it).

```bash
cd ~/Documents/Dev/dj-loop-service
uv sync
```

System tools the Python wrappers shell out to:

```bash
brew install ffmpeg chromaprint
```

- `ffmpeg` — fallback audio decoder (some MP3s need it).
- `chromaprint` — provides the `fpcalc` binary that `pyacoustid` calls for
  acoustic fingerprinting. Without it, `core.load` still runs but
  `content.fingerprint` is left NULL (the library still self-heals via
  `file_hash`, just less robustly across format conversion).

## Run

```bash
# Analyze a single file
uv run service analyze ~/Music/MyTrack.mp3 --db /path/to/library.db

# Analyze a folder recursively (multiple files in parallel)
uv run service analyze ~/Music --db /path/to/library.db --workers 4
```

The `library.db` is expected to exist already (created by the Player at first
launch — schema is OneLibrary-aligned, see Player's
`docs/lib/ONELIBRARY_SPEC.md`). The Service will fail loudly if expected
tables are missing. For dev, point at a fresh Player-created DB.

## Project Layout

```
src/dj_loop_service/
  cli.py                  click-based CLI entry point
  pipeline.py             per-track stage orchestrator
  plugin.py               @analyzer_stage decorator, base classes
  db.py                   SQLite read/write helpers
  config.py               config loading (TOML)
  plugins/
    core_load.py          librosa load + mutagen tags  → content row
    core_beats.py         STUB — Madmom beat tracking
    core_key.py           STUB — Madmom key detection
    core_sections.py      STUB — MSAF section boundaries
    core_loop_mining.py   STUB — bar-aligned candidate loops
    core_loop_features_basic.py  STUB — MFCC, chroma, spectral, energy
tests/
```

## Development

```bash
uv run pytest          # tests
uv run ruff check .    # lint
uv run ruff format .   # format
```

## Architecture

See the Player project's `docs/service/`:
- `OVERVIEW.md` — three planes (analysis / realtime / plugins), Player↔Service split
- `ANALYZER_SPEC.md` — this service's normative spec
- `REALTIME_SPEC.md` — WebSocket event plane (future)
- `PLUGINS_SPEC.md` — plugin contract
