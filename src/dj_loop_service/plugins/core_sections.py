"""core.sections — STUB.

Phase 1 placeholder. Phase 2 will use MSAF to detect section boundaries
(intro/verse/drop/breakdown/outro) keyed to beat indices, written into the
track_section table (see ANALYZER_SPEC §4.1).
"""

from __future__ import annotations

from ..plugin import Ctx, analyzer_stage


@analyzer_stage(name="sections", order=30)
def sections(ctx: Ctx) -> None:
    pass
