"""core.structure plugin — Beat This! (CPJKU/beat_this, ISMIR 2024).

Beats + downbeats + tempo from a single forward pass. Uses the framewise
`Audio2Frames` + `Postprocessor("minimal")` API. Tempo is the median
inter-beat interval. Bar phase (which beats are downbeats) is corrected
post-hoc by a "dominant gap → phase projection" heuristic because the
minimal postprocessor over-fires downbeats on tracks with sparse intros.

The heuristic is the band-aid we're keeping until we swap to madmom's
DBN postprocessor (separate plugin, separate file). When that lands,
this plugin gets disabled by removing its import from
`worker/plugin.py:load_builtin_plugins`.
"""

from __future__ import annotations

import logging
import statistics
from pathlib import Path

from plugin import WorkerCtx, analyzer_stage

log = logging.getLogger("worker")

# Lazy-loaded model + postprocessor. Held module-global so subsequent
# tracks reuse the loaded checkpoint (~10 s first-use cost otherwise).
_a2f = None
_postp = None


@analyzer_stage(name="structure", order=20)
def structure(ctx: WorkerCtx) -> None:
    """Populate ctx.columns with tempo + beat_grid_csv + bars_csv + anchors."""
    global _a2f, _postp

    if _a2f is None:
        ctx.progress("loading_model", 0.05)
        from beat_this.inference import Audio2Frames, Postprocessor
        log.info("loading beat_this model (first use; downloads checkpoint if absent)")
        _a2f = Audio2Frames(checkpoint_path="final0", device="cpu")
        _postp = Postprocessor(type="minimal")

    ctx.progress("running_inference", 0.20)

    from beat_this.inference import load_audio
    signal, sr = load_audio(str(ctx.audio_path))
    beat_logits, downbeat_logits = _a2f(signal, sr)

    ctx.progress("computing_tempo", 0.85)
    beats, downbeats = _postp(beat_logits, downbeat_logits)

    beats_list = [round(float(b), 4) for b in beats]
    downbeats_list = [float(b) for b in downbeats]

    # Diagnostic — raw model output, useful when anchor placement looks
    # wrong on a specific track. Remove once anchor logic is settled.
    log.info("       BEAT_THIS RAW: first 5 beats     = %s",
             [round(b, 3) for b in beats_list[:5]])
    log.info("       BEAT_THIS RAW: first 5 downbeats = %s",
             [round(b, 3) for b in downbeats_list[:5]])
    if len(beats_list) >= 6:
        intervals_head = [round(beats_list[i + 1] - beats_list[i], 3)
                          for i in range(5)]
        log.info("       BEAT_THIS RAW: first 5 inter-beat intervals = %s s",
                 intervals_head)

    # Tempo from median inter-beat interval, stored as BPM × 100 integer
    # (ONELIBRARY_SPEC §8.2 / ANALYZER_SPEC §4.5).
    if len(beats_list) >= 2:
        intervals = [beats_list[i + 1] - beats_list[i] for i in range(len(beats_list) - 1)]
        median_interval = statistics.median(intervals)
        tempo_x100 = round(60.0 / median_interval * 100) if median_interval > 0 else 0
    else:
        tempo_x100 = 0

    # bars_csv = beat indices of downbeats (ANALYZER_SPEC §4.5). For each
    # downbeat time, find the index of the closest beat in beats_list.
    bar_indices: list[int] = []
    if beats_list and downbeats_list:
        i = 0
        for db in downbeats_list:
            while i + 1 < len(beats_list) and abs(beats_list[i + 1] - db) < abs(beats_list[i] - db):
                i += 1
            bar_indices.append(i)

    # Bar-phase anchor selection. Find the dominant inter-downbeat gap
    # (4 for 4/4 tracks). Find the longest contiguous run of that gap.
    # Compute the phase (beat-index mod dominant_gap) inside the run.
    # Project the phase to the whole audio: bar 1 = first beat matching
    # the phase. See doc comment in `structure_beat_this.py` for why.
    start_anchor_beat_index: int | None = None
    end_anchor_beat_index: int | None = None
    if len(bar_indices) >= 3:
        gaps = [bar_indices[i + 1] - bar_indices[i] for i in range(len(bar_indices) - 1)]
        gap_counts: dict[int, int] = {}
        for g in gaps:
            gap_counts[g] = gap_counts.get(g, 0) + 1
        dominant_gap = max(gap_counts.items(), key=lambda kv: kv[1])[0]

        best_start_i = 0
        best_end_i = 0
        best_len = 0
        cur_start_i = 0
        cur_len = 0
        for i, g in enumerate(gaps):
            if g == dominant_gap:
                if cur_len == 0:
                    cur_start_i = i
                cur_len += 1
                if cur_len > best_len:
                    best_len = cur_len
                    best_start_i = cur_start_i
                    best_end_i = i + 1
            else:
                cur_len = 0

        if best_len >= 2:
            run_first_db_beat = bar_indices[best_start_i]
            phase = run_first_db_beat % dominant_gap
            start_anchor_beat_index = phase
            n_beats = len(beats_list)
            end_anchor_beat_index = ((n_beats - 1 - phase) // dominant_gap) * dominant_gap + phase
            log.info("       BAR-PHASE: dominant gap=%d beats  longest run=%d downbeats  "
                     "phase=%d  (run was beat%d..beat%d)",
                     dominant_gap, best_len + 1, phase,
                     run_first_db_beat, bar_indices[best_end_i])
            log.info("       BAR-PHASE: start anchor=beat%d (%.2fs)  "
                     "end anchor=beat%d (%.2fs)  → projected across full track",
                     start_anchor_beat_index, beats_list[start_anchor_beat_index],
                     end_anchor_beat_index, beats_list[end_anchor_beat_index])
        else:
            log.warning("       BAR-PHASE: no run of consecutive %d-beat gaps — "
                        "falling back to bars edges", dominant_gap)

    if start_anchor_beat_index is None and bar_indices:
        start_anchor_beat_index = bar_indices[0]
    if end_anchor_beat_index is None and bar_indices:
        end_anchor_beat_index = bar_indices[-1]

    log.info("       start_anchor_beat_index=%s  end_anchor_beat_index=%s  (of %d downbeats)",
             start_anchor_beat_index, end_anchor_beat_index, len(downbeats_list))

    ctx.columns["tempo"] = tempo_x100
    ctx.columns["beat_grid_csv"] = ",".join(str(b) for b in beats_list)
    ctx.columns["bars_csv"] = ",".join(str(b) for b in bar_indices)
    if start_anchor_beat_index is not None:
        ctx.columns["start_anchor_beat_index"] = int(start_anchor_beat_index)
    if end_anchor_beat_index is not None:
        ctx.columns["end_anchor_beat_index"] = int(end_anchor_beat_index)
