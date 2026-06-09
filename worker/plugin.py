"""Worker-side plugin framework.

Mirrors `src/dj_loop_service/plugin.py` (control plane) but for the worker
process. Each registered stage is a callable that takes a `WorkerCtx`,
reads `ctx.audio_path`, populates `ctx.columns` with content-row values,
and may emit sub-stage progress via `ctx.progress(sub_stage, fraction)`.

After every stage the worker POSTs the accumulated columns to
`/v1/worker/result` and emits an `analyzer.track_stage` event. To swap
the implementation of any stage, drop in a new plugin file with the same
`name` — Python import order decides who wins (the loader controls that).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Module-level registry. Plugins register at import time; the worker reads
# this at startup. Single-process, so a module-global is fine.
_STAGES: list["StageRegistration"] = []


@dataclass
class StageRegistration:
    name: str
    order: int
    func: Callable[..., None]


def analyzer_stage(name: str, order: int) -> Callable[[Callable[..., None]], Callable[..., None]]:
    """Decorator registering a worker-side analyzer stage.

    `name`  — logical stage name (e.g. "structure", "key"). Surfaces in
              `analyzer.track_stage` events.
    `order` — sort key. Worker stages use 20–60. Lower runs first.
    """

    def decorator(func: Callable[..., None]) -> Callable[..., None]:
        _STAGES.append(StageRegistration(name=name, order=order, func=func))
        return func

    return decorator


def registered_stages() -> list[StageRegistration]:
    """Return all currently registered stages, sorted by order then name."""
    return sorted(_STAGES, key=lambda s: (s.order, s.name))


@dataclass
class WorkerCtx:
    """Per-job context shared across worker stages.

    Stages read `audio_path`, populate `columns` (becomes the body of
    `POST /v1/worker/result`), and optionally call `progress(sub_stage,
    fraction)` to emit `analyzer.track_stage` events with `stage_name`
    of the form `<stage>.<sub_stage>`. Sub-stage progress is how long-
    running stages (Beat This! checkpoint load + inference) advance the
    Player progress bar during silent periods.
    """

    audio_path: Path
    job_id: int
    host_path: str
    content_id: int | None = None
    # `columns` accumulates across stages within a job — earlier stages
    # (structure: beat_grid_csv, bars_csv, tempo) seed later ones
    # (loop_mining needs the beat grid). The runner POSTs only the delta
    # written by the current stage to /v1/worker/result.
    columns: dict[str, Any] = field(default_factory=dict)
    # Cues (loops) produced by this stage. loop_mining writes here;
    # other stages leave it empty. Each dict is a cue row matching the
    # cue table shape — start_sample / end_sample optional, the control
    # plane converts to pointNumerator/loopNumerator. source defaults
    # to 'analyzer' so user cues are never overwritten.
    cues: list[dict[str, Any]] = field(default_factory=list)
    # Callback installed by main.py before each stage. Default = no-op so
    # plugins can be unit-tested without an HTTP client.
    progress: Callable[[str, float], None] = lambda _sub, _frac: None


def load_builtin_plugins() -> None:
    """Import every built-in plugin module so it registers its stages.

    Plugins register at import time via the @analyzer_stage decorator.
    Adding a new built-in plugin is one import line here. Removing a
    plugin is removing its import line — the file stays on disk but the
    decorator never runs so the stage isn't registered.
    """
    # core.structure — beats + downbeats + tempo. Exactly one of these
    # must be active (they both register `@analyzer_stage(name="structure")`,
    # so importing both would double-register the stage).
    #
    # Active:  madmom (RNN + DBN — global HMM decode, handles EDM intros)
    # Standby: structure_beat_this — keep on disk for A/B comparison
    from plugins import structure_madmom        # noqa: F401  — active (tighter bar phase via DBN HMM decode, +30s vs Beat This!)
    # from plugins import structure_beat_this  # noqa: F401  — standby (faster but less reliable downbeats on sparse intros)

    # core.key — Essentia EDMA musical key + Camelot wheel.
    from plugins import key_essentia  # noqa: F401

    # core.loop_propose — deterministic RMS + slope picker (6 loops).
    # Active. See ANALYZER_SPEC §2.2 / STATUS.md §6 for the picker+tagger
    # split and the autopsy of the retired ML-driven loop_mining.
    #
    # Standby: loop_mining (ML-driven scoring via MuQ-MuLan bar embeddings).
    # Retired 2026-06-07 for poor musical results — 2-second bar embeddings
    # cluster near the mean and "consistency" rewards homogeneity. Kept on
    # disk for A/B if a better scoring scheme emerges.
    from plugins import loop_propose  # noqa: F401
    # from plugins import loop_mining  # noqa: F401
