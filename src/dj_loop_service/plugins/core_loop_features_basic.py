"""core.loop_features_basic — STUB.

Phase 1 placeholder. Phase 2 will compute librosa-derived per-loop features
(MFCC mean/std, chroma mean, spectral flux, brightness, dynamic range,
energy_value, energy_label) and write into cue.* extension columns
(ANALYZER_SPEC §4.2).
"""

from __future__ import annotations

from ..plugin import Ctx, analyzer_stage


@analyzer_stage(name="loop_features_basic", order=50, per_loop=True)
def loop_features_basic(ctx: Ctx) -> None:
    pass
