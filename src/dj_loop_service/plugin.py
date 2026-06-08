"""Plugin framework.

A plugin is a Python module that registers one or more analyzer stages via
the `@analyzer_stage` decorator. Stages are functions that take a per-track
context (`Ctx`) and produce values into it. The pipeline orchestrator
(`pipeline.py`) runs registered stages in `order` for each track.

Future expansion: realtime hooks (`@on_event`) — not implemented in Phase 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# Module-level registry. Plugins register at import time; the pipeline reads
# this at startup. Single-process service, so a module-global is fine.
_STAGES: list[StageRegistration] = []


@dataclass
class StageRegistration:
    name: str
    order: int
    func: Callable[..., None]
    per_loop: bool = False
    replaces: str | None = None


def analyzer_stage(
    name: str,
    order: int,
    *,
    per_loop: bool = False,
    replaces: str | None = None,
) -> Callable[[Callable[..., None]], Callable[..., None]]:
    """Decorator registering an analyzer pipeline stage.

    `name`     — logical stage name (e.g. "load", "beats", "key"). Used by
                 the pipeline for ordering and by realtime events for status.
    `order`    — sort key. Built-in stages use 10–60. Lower runs first.
    `per_loop` — if True, the stage runs once per discovered loop instead of
                 once per track. Per-loop stages receive a Loop object.
    `replaces` — if set, this stage takes the place of the built-in stage
                 with the matching dotted import path.
    """

    def decorator(func: Callable[..., None]) -> Callable[..., None]:
        _STAGES.append(
            StageRegistration(
                name=name, order=order, func=func, per_loop=per_loop, replaces=replaces
            )
        )
        return func

    return decorator


def registered_stages() -> list[StageRegistration]:
    """Return all currently registered stages, sorted by order then name."""
    return sorted(_STAGES, key=lambda s: (s.order, s.name))


@dataclass
class Ctx:
    """Per-track context shared across pipeline stages.

    Stages read inputs via `get()` and produce outputs via `set()`. The DB
    persistence stage at the end reads `to_persist` and writes to SQLite.
    """

    file_path: str           # path the Analyzer can actually open (host or container-temp)
    host_path: str           # path the Player will use to find/play the audio (host-local)
    db_path: str
    user_id: str = "local"
    values: dict[str, Any] = field(default_factory=dict)
    loop_values: dict[int, dict[str, Any]] = field(default_factory=dict)
    to_persist: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.values[key] = value

    def set_persistable(self, column: str, value: Any) -> None:
        """Mark a value for persistence to the `content` row."""
        self.to_persist[column] = value
