"""core.beats — STUB.

Phase 1 placeholder. Phase 2 implementation will use Madmom's CNN beat
tracker to produce per-beat onset times and downbeat indices, written into
content.beat_grid_csv and content.bars_csv.
"""

from __future__ import annotations

from ..plugin import Ctx, analyzer_stage


@analyzer_stage(name="beats", order=20)
def beats(ctx: Ctx) -> None:
    # No-op until Madmom is integrated.
    pass
