"""core.key — STUB.

Phase 1 placeholder. Phase 2 will use Madmom's CNN key estimator and write
content.musical_key + content.camelot_key + content.key_id (FK).
"""

from __future__ import annotations

from ..plugin import Ctx, analyzer_stage


@analyzer_stage(name="key", order=25)
def key(ctx: Ctx) -> None:
    pass
