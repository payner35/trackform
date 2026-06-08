"""core.loop_mining — STUB.

Phase 1 placeholder. Phase 2 will produce bar-aligned candidate loops
(8/16/32-bar slices at section starts) and write them into the cue table
with kind=4 and source='native'.
"""

from __future__ import annotations

from ..plugin import Ctx, analyzer_stage


@analyzer_stage(name="loop_mining", order=40)
def loop_mining(ctx: Ctx) -> None:
    pass
