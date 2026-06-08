"""Per-track pipeline orchestrator.

Loads registered stages, runs them in order against a Ctx, persists results
to library.db. Worker pool for track-level parallelism per
`docs/service/ANALYZER_SPEC.md §7`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from rich.console import Console

from .config import Config
from .db import connect, ensure_schema_present, transaction, upsert_content
from .events import event_bus
from .plugin import Ctx, registered_stages


@dataclass
class TrackResult:
    file_path: Path
    content_id: int | None
    ok: bool
    error: str | None = None


def analyze_file(
    file_path: Path,
    config: Config,
    *,
    host_path: Path | None = None,
    emit_terminal: bool = True,
) -> TrackResult:
    """Run the pipeline for a single file. One DB connection per call (thread-safe).

    `file_path` is what the Analyzer opens — could be a host path (CLI mode)
    or a temp path inside the container (HTTP upload mode).

    `host_path` is what gets written to `content.file_path_absolute` so the
    Player can find/play the file later. Defaults to `file_path` for CLI use.

    `emit_terminal`: emit `analyzer.track_done` / `track_failed` at the end.
    The HTTP path passes `False` because the heavy analysis runs in the worker
    container, which emits the real terminal event when it finishes. Without
    this, the control plane emits `track_done` after `core.load` (milliseconds
    after POST), the Player treats it as final, and the worker's later result
    arrives with no listener — the UI renders an empty grid.
    """
    host_path = host_path or file_path
    host_path_str = str(host_path)
    event_bus.publish(
        "analyzer.track_started",
        {"host_path": host_path_str, "user_id": config.user_id},
    )
    stages = [s for s in registered_stages() if not s.per_loop]
    conn = connect(config.db_path)
    ensure_schema_present(conn)
    content_id: int | None = None
    try:
        ctx = Ctx(
            file_path=str(file_path),
            host_path=host_path_str,
            db_path=str(config.db_path),
            user_id=config.user_id,
        )
        for i, stage in enumerate(stages):
            event_bus.publish(
                "analyzer.track_stage",
                {
                    "host_path": host_path_str,
                    "stage_name": stage.name,
                    "stage_index": i,
                    "stage_count": len(stages),
                    "fraction_overall": i / max(len(stages), 1),
                    # content_id is None until the first upsert (after `load`).
                    # Player treats this as "row not yet persisted, just show progress".
                    "content_id": content_id,
                },
            )
            stage.func(ctx)
            # Incremental persist: after each stage, write whatever's been
            # accumulated so far. The first call (after `load`) creates the row
            # via upsert_content's file_hash/fingerprint lookup; subsequent calls
            # just UPDATE. Lets the Player see fields trickle in (BPM after
            # `beats`, loops after `loop_mining`, etc.) instead of one big pop
            # at the end, and survives a crash mid-pipeline.
            if ctx.to_persist:
                with transaction(conn):
                    content_id = upsert_content(
                        conn,
                        file_path_absolute=host_path_str,
                        columns=ctx.to_persist,
                    )

        if emit_terminal:
            event_bus.publish(
                "analyzer.track_done",
                {"host_path": host_path_str, "content_id": content_id, "user_id": config.user_id},
            )
        return TrackResult(file_path=host_path, content_id=content_id, ok=True)
    except Exception as e:
        if emit_terminal:
            event_bus.publish(
                "analyzer.track_failed",
                {"host_path": host_path_str, "error_message": str(e)},
            )
        return TrackResult(file_path=host_path, content_id=None, ok=False, error=str(e))
    finally:
        conn.close()


def analyze_many(
    files: Iterable[Path],
    config: Config,
    *,
    on_result: Callable[[TrackResult], None] | None = None,
) -> list[TrackResult]:
    """Run the pipeline across multiple files in parallel."""
    files = list(files)
    results: list[TrackResult] = []

    with ThreadPoolExecutor(max_workers=config.workers) as pool:
        futures = {pool.submit(analyze_file, f, config): f for f in files}
        for fut in as_completed(futures):
            result = fut.result()
            results.append(result)
            if on_result:
                on_result(result)
    return results


def load_builtin_plugins() -> None:
    """Import every built-in plugin module so it registers its stages.

    Plugins register at import time via the @analyzer_stage decorator.
    Adding a new built-in plugin is just an import line here.
    """
    # Real implementations
    from .plugins import core_load  # noqa: F401

    # NOTE: the core_beats / core_key / core_sections / core_loop_mining /
    # core_loop_features_basic stub modules predate the worker rewrite and
    # are no longer loaded — those stages now live in worker/plugins/ and
    # emit their own analyzer.track_stage events (`structure`, `key`,
    # `loop_mining`). Importing the stubs here would register them on the
    # control plane and emit a second set of stage events under the old
    # names, confusing the Player's stage-progress UI. The files stay on
    # disk for git history (see STATUS.md "What's Next" item 0).
