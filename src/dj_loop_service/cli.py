"""Click-based CLI entry point.

Exposed as `service` via the [project.scripts] table in pyproject.toml.
Usage:

    uv run service analyze <path>... --db <library.db> [--workers N]
"""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .config import Config
from .pipeline import TrackResult, analyze_many, load_builtin_plugins

console = Console()

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".aiff", ".aif", ".m4a", ".aac"}


def _collect_audio_files(paths: tuple[Path, ...]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            out.append(p.resolve())
        elif p.is_dir():
            for child in p.rglob("*"):
                if child.is_file() and child.suffix.lower() in AUDIO_EXTS:
                    out.append(child.resolve())
    return sorted(set(out))


@click.group()
def cli() -> None:
    """DJ Loop Service — audio analysis for DJ Loop Player."""


@cli.command()
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option(
    "--db",
    "db_path",
    required=True,
    type=click.Path(path_type=Path),
    help="Path to library.db (must already exist — create it by launching the Player once).",
)
@click.option(
    "--workers",
    default=1,
    type=int,
    show_default=True,
    help="Parallel worker count (track-level parallelism).",
)
@click.option(
    "--force",
    is_flag=True,
    help="Re-analyze tracks even if already in the library.",
)
@click.option(
    "--user",
    "user_id",
    default="local",
    show_default=True,
    help="Owner user_id stamped onto each row (multi-tenancy, ONELIBRARY_SPEC §8.9.9).",
)
def analyze(
    paths: tuple[Path, ...],
    db_path: Path,
    workers: int,
    force: bool,
    user_id: str,
) -> None:
    """Analyze one or more audio files / folders into library.db."""
    load_builtin_plugins()

    files = _collect_audio_files(paths)
    if not files:
        console.print("[yellow]No audio files found in the given paths.[/yellow]")
        return

    console.print(
        f"[bold]Analyzing[/bold] {len(files)} file(s) into {db_path}  (user=[cyan]{user_id}[/cyan])"
    )
    config = Config(
        db_path=db_path.resolve(),
        workers=workers,
        force_reanalyze=force,
        user_id=user_id,
    )

    results: list[TrackResult] = []

    def on_result(r: TrackResult) -> None:
        results.append(r)
        marker = "[green]✓[/green]" if r.ok else "[red]✗[/red]"
        path_short = str(r.file_path).replace(str(Path.home()), "~")
        if r.ok:
            console.print(f"{marker} {path_short}  →  content_id={r.content_id}")
        else:
            console.print(f"{marker} {path_short}  →  {r.error}")

    analyze_many(files, config, on_result=on_result)

    ok = sum(1 for r in results if r.ok)
    fail = len(results) - ok
    summary = Table(show_header=False, box=None)
    summary.add_row("Total", str(len(results)))
    summary.add_row("[green]Succeeded[/green]", str(ok))
    summary.add_row("[red]Failed[/red]" if fail else "Failed", str(fail))
    console.print(summary)


@cli.command()
@click.option(
    "--db",
    "db_path",
    required=True,
    type=click.Path(path_type=Path),
    help="Path to library.db (must already exist — created by the Player on first launch).",
)
@click.option("--host", default="0.0.0.0", show_default=True, help="Bind address.")
@click.option("--port", default=7777, show_default=True, help="Bind port.")
def serve(db_path: Path, host: str, port: int) -> None:
    """Run the HTTP server. POST audio to /v1/analyze for analysis."""
    import os

    import uvicorn

    from .db import connect, init_schema

    db_path = db_path.resolve()
    # The Service owns its DB — create it (and the OneLibrary-aligned schema)
    # if missing. No dependency on the Player having ever launched.
    conn = connect(db_path)
    init_schema(conn)
    conn.close()

    # The FastAPI app reads DLP_DB_PATH at request time. Cleaner than passing
    # global state through the app factory.
    os.environ["DLP_DB_PATH"] = str(db_path)

    console.print(f"[bold]Serving[/bold] on http://{host}:{port}  (db=[cyan]{db_path}[/cyan])")

    # Filter uvicorn access logs so the worker's per-poll 204 hits don't
    # drown the actually-interesting messages.
    import logging as _logging

    class _SkipNoise(_logging.Filter):
        def filter(self, record: _logging.LogRecord) -> bool:
            msg = record.getMessage()
            # Worker poll noise.
            if "/v1/worker/next" in msg or "/v1/health" in msg:
                return False
            # The Player re-fetches GET /v1/tracks/{id} after every WS event,
            # so each stage produces multiple identical lines. The "dls →"
            # logger now reports each result/event meaningfully — these are
            # redundant. Drop them too.
            if "/v1/worker/event" in msg or "/v1/worker/result" in msg:
                return False
            if "GET /v1/tracks/" in msg:
                return False
            return True

    _logging.getLogger("uvicorn.access").addFilter(_SkipNoise())

    uvicorn.run("dj_loop_service.server:app", host=host, port=port, log_level="info")


if __name__ == "__main__":  # pragma: no cover
    cli()
