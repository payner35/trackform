"""core.loop_propose — deterministic 6-loop picker (RMS energy + slope).

Replaces the retired ML-driven `loop_mining` (see ANALYZER_SPEC §2.2 history
and STATUS.md §6 "Loop selection inversion"). Runs in sub-second, no model
load, no embedding work. MuQ-MuLan moves to the on-demand `loop_tag` stage
which the Player triggers when the user confirms a loop set.

Algorithm:
    1. Read beats / downbeats from ctx.columns (set by core.structure).
    2. Load audio at 22.05 kHz mono.
    3. For each downbeat i where an 8-bar window fits:
         energy[i] = mean |audio[start..end]|     (~loudness)
         slope[i]  = energy[i+8] - energy[i-8]    (>0 means rising)
    4. Pick 6 loops:
         - 3 build-ups: in the first half of the track, take the 3 windows
           with the highest positive slope, enforcing a minimum gap of
           8 downbeats so we don't pick the same buildup three times.
         - 3 variety: split the remaining downbeats into 3 segments; in
           each, pick the window with the highest absolute energy
           (the loudest moment per region — usually the drops).
    5. Write 6 cue rows kind=4 via ctx.cues.

`overall_score` is set to the picker's combined score (normalised energy
0..1) so the Player's existing Quality display has something to render.
No mood_tags / genre_hints / embedding — those land later via loop_tag.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from plugin import WorkerCtx, analyzer_stage

log = logging.getLogger("worker")

LOOP_BARS = 8
NUM_LOOPS = 6
NUM_BUILDUPS = 3
MIN_GAP_DOWNBEATS = 8   # don't pick build-up loops closer than this many downbeats
AUDIO_SR = 22_050


def _load_audio_mono(audio_path: Path, sr: int) -> np.ndarray:
    import librosa
    wav, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    return wav.astype(np.float32)


def _format_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"


@analyzer_stage(name="loop_propose", order=40)
def loop_propose(ctx: WorkerCtx) -> None:
    beat_grid_csv = ctx.columns.get("beat_grid_csv") or ""
    bars_csv = ctx.columns.get("bars_csv") or ""
    if not beat_grid_csv or not bars_csv:
        log.warning("       loop_propose: missing beat_grid_csv / bars_csv (structure didn't run?) — skipping")
        return

    beats: list[float] = [float(x) for x in beat_grid_csv.split(",") if x]
    bars: list[int] = [int(x) for x in bars_csv.split(",") if x]
    bar_count = len(bars)
    if bar_count < LOOP_BARS + 1:
        log.warning("       loop_propose: only %d downbeats, need ≥ %d for an 8-bar window — skipping",
                    bar_count, LOOP_BARS + 1)
        return

    ctx.progress("loading_audio", 0.20)
    log.info("       loop_propose: loading audio at %d Hz", AUDIO_SR)
    wav = _load_audio_mono(ctx.audio_path, AUDIO_SR)
    log.info("       loop_propose: %d downbeats, %.1fs of audio", bar_count, len(wav) / AUDIO_SR)

    # ---- Step 1: per-window energy
    ctx.progress("scoring", 0.50)
    # n_windows valid start downbeats — i must satisfy i + LOOP_BARS <= bar_count - 1
    # so beats[bars[i + LOOP_BARS]] is defined.
    n_windows = bar_count - LOOP_BARS
    energy = np.zeros(n_windows, dtype=np.float32)
    for i in range(n_windows):
        s = int(beats[bars[i]] * AUDIO_SR)
        e = int(beats[bars[i + LOOP_BARS]] * AUDIO_SR)
        s = max(0, s)
        e = min(len(wav), e)
        if e > s:
            energy[i] = float(np.abs(wav[s:e]).mean())

    # ---- Step 2: slope (rising energy indicates a build-up)
    slope = np.zeros(n_windows, dtype=np.float32)
    for i in range(n_windows):
        prev = energy[max(0, i - LOOP_BARS)]
        nxt = energy[min(n_windows - 1, i + LOOP_BARS)]
        slope[i] = nxt - prev

    log.info("       loop_propose: energy range [%.4f, %.4f]  slope range [%.4f, %.4f]",
             float(energy.min()), float(energy.max()),
             float(slope.min()), float(slope.max()))

    # ---- Step 3: pick 3 build-up loops (first half, highest positive slope, with min gap)
    first_half_end = n_windows // 2
    buildup_idx: list[int] = []
    # rank windows in first half by slope descending
    ranked = sorted(range(first_half_end), key=lambda i: -slope[i])
    for cand in ranked:
        if slope[cand] <= 0:
            break   # no more positive-slope candidates
        if all(abs(cand - chosen) >= MIN_GAP_DOWNBEATS for chosen in buildup_idx):
            buildup_idx.append(cand)
            if len(buildup_idx) >= NUM_BUILDUPS:
                break
    log.info("       loop_propose: %d build-up loops picked (idx=%s)",
             len(buildup_idx), buildup_idx)

    # ---- Step 4: pick variety loops — split the REST of the track into 3 segments,
    #              top energy in each. "Rest" = after first_half_end, OR fall back
    #              to whole track if no build-ups were found.
    variety_start = first_half_end
    variety_end = n_windows
    seg_count = NUM_LOOPS - len(buildup_idx)   # fill out to 6 total
    variety_idx: list[int] = []
    if seg_count > 0 and variety_end - variety_start >= seg_count:
        seg_size = (variety_end - variety_start) / seg_count
        for k in range(seg_count):
            seg_lo = variety_start + int(round(k * seg_size))
            seg_hi = variety_start + int(round((k + 1) * seg_size))
            seg_hi = min(seg_hi, variety_end)
            if seg_hi <= seg_lo:
                continue
            # argmax in this segment, but skip anything already picked as buildup
            candidates = [i for i in range(seg_lo, seg_hi) if i not in buildup_idx and i not in variety_idx]
            if not candidates:
                continue
            best = max(candidates, key=lambda i: energy[i])
            variety_idx.append(best)
    log.info("       loop_propose: %d variety loops picked (idx=%s)",
             len(variety_idx), variety_idx)

    # ---- Step 5: emit cues (ordered by track time)
    ctx.progress("writing_cues", 0.90)
    all_picks = sorted(set(buildup_idx + variety_idx))
    if not all_picks:
        log.warning("       loop_propose: no candidates picked — empty cue set")
        return

    # Normalise energy to 0..1 for overall_score so the Player's existing
    # Quality display has something interpretable.
    e_min, e_max = float(energy.min()), float(energy.max())
    e_range = max(e_max - e_min, 1e-9)

    for idx in all_picks:
        is_buildup = idx in buildup_idx
        start_s = beats[bars[idx]]
        end_s = beats[bars[idx + LOOP_BARS]]
        start_ms = int(round(start_s * 1000))
        length_ms = int(round((end_s - start_s) * 1000))
        score = float((energy[idx] - e_min) / e_range)

        loop_kind = "buildup" if is_buildup else "variety"
        label = f"{'Build-up' if is_buildup else 'Loop'} 8-bar @ {_format_time(start_s)}"

        ctx.cues.append({
            "kind": 4,
            "label": label,
            "loop_type": loop_kind,
            "pointNumerator": start_ms,
            "pointDenominator": 1000,
            "loopNumerator": length_ms,
            "loopDenominator": 1000,
            "start_bar": idx + 1,  # 1-indexed for display
            "bars": LOOP_BARS,
            "overall_score": round(score, 4),
        })

    log.info("       loop_propose: emitted %d cues (%d build-ups, %d variety)",
             len(all_picks), len(buildup_idx), len(variety_idx))
