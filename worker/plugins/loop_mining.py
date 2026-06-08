"""core.loop_mining plugin (v0) — MuQ-MuLan embedding consistency.

Enumerates downbeat-aligned 8/16/32-bar candidates across the whole track
(sections fallback per ANALYZER_SPEC §2.2 step 3), scores each by mean
bar-to-mean cosine consistency on MuQ-MuLan embeddings, keeps the top N
per length, and writes them to the cue table via the worker→control
contract (`ctx.cues`).

Reads from ctx.columns:
    beat_grid_csv  — comma-separated beat times (seconds), set by structure
    bars_csv       — comma-separated beat-indices of downbeats

Writes per cue:
    kind = 4                      (loop)
    label                         "Auto 16-bar @ 1:24"
    loop_type = "auto"
    pointNumerator / loopNumerator  start / length in MILLISECONDS
                                    (matches OneLibrary cue convention —
                                     pointDenominator defaults to 1000)
    start_bar                     downbeat index this loop starts on
    bars                          length in bars
    embedding                     MuQ-MuLan 512-d float32 mean vector
                                  (raw bytes, np.float32.tobytes())
    embedding_consistency         mean cosine(bar_i, mean), 0..1
    overall_score                 v0 = embedding_consistency
"""

from __future__ import annotations

import logging
import math
from typing import Iterable

import base64
import json
import numpy as np
import torch

from plugin import WorkerCtx, analyzer_stage
from plugins.embed_muq_mulan import SAMPLE_RATE, embed_audio, score_mood, score_genre

log = logging.getLogger("worker")

# v0 parameters. Conservative defaults; tunable per-track later.
_LOOP_LENGTHS_BARS = (8, 16, 32)
_TOP_N_PER_LENGTH = 4
# Loops below this consistency threshold aren't worth surfacing — windows
# crossing a section boundary typically land here.
_MIN_CONSISTENCY = 0.70
# Hop between candidate start downbeats. 1 = every downbeat (densest, slow);
# 4 = every 4 bars (sparse, ~4x faster). v0 ships dense — quality > speed.
_DOWNBEAT_HOP = 1


def _load_audio_mono_24k(audio_path) -> np.ndarray:
    """Read the file at any sample rate, downmix, resample to 24 kHz mono."""
    import librosa
    wav, _sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    return wav.astype(np.float32)


def _slice_bar(wav: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    s = max(0, int(round(start_s * SAMPLE_RATE)))
    e = min(len(wav), int(round(end_s * SAMPLE_RATE)))
    if e <= s:
        return np.zeros(SAMPLE_RATE, dtype=np.float32)
    return wav[s:e]


def _format_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"


@analyzer_stage(name="loop_mining", order=40)
def loop_mining(ctx: WorkerCtx) -> None:
    beat_grid_csv = ctx.columns.get("beat_grid_csv") or ""
    bars_csv = ctx.columns.get("bars_csv") or ""
    if not beat_grid_csv or not bars_csv:
        log.warning("       loop_mining: missing beat_grid_csv / bars_csv "
                    "(structure stage didn't run?) — skipping")
        return

    beats: list[float] = [float(x) for x in beat_grid_csv.split(",") if x]
    bars: list[int] = [int(x) for x in bars_csv.split(",") if x]
    if len(bars) < max(_LOOP_LENGTHS_BARS) + 1:
        log.warning("       loop_mining: only %d downbeats, need ≥ %d for "
                    "the smallest candidate — skipping",
                    len(bars), max(_LOOP_LENGTHS_BARS) + 1)
        return

    ctx.progress("loading_audio", 0.05)
    log.info("       loop_mining: loading audio at %d Hz", SAMPLE_RATE)
    wav = _load_audio_mono_24k(ctx.audio_path)
    duration_s = len(wav) / SAMPLE_RATE
    log.info("       loop_mining: %d downbeats, %.1fs of audio", len(bars), duration_s)

    # Embed every individual bar ONCE — sliding windows reuse them.
    # Each bar = downbeat[i] → downbeat[i+1].
    ctx.progress("embedding_bars", 0.15)
    bar_count = len(bars) - 1
    log.info("       loop_mining: embedding %d bars via MuQ-MuLan", bar_count)
    bar_wavs: list[np.ndarray] = []
    for i in range(bar_count):
        start_s = beats[bars[i]]
        end_s = beats[bars[i + 1]]
        bar_wavs.append(_slice_bar(wav, start_s, end_s))

    # Batch in chunks to keep memory bounded on long tracks (a 6-min track
    # at 4 beats/bar is ~90 bars; batch of 32 keeps peak around 1.5 GB).
    bar_embeds_list = []
    BATCH = 16
    for i in range(0, len(bar_wavs), BATCH):
        chunk = bar_wavs[i:i + BATCH]
        # Pad each window to the longest in the chunk so they stack.
        max_len = max(len(w) for w in chunk)
        padded = np.stack([
            np.pad(w, (0, max_len - len(w))) for w in chunk
        ]).astype(np.float32)
        embeds = embed_audio(padded).numpy()
        bar_embeds_list.append(embeds)
        ctx.progress("embedding_bars",
                     0.15 + 0.55 * min(1.0, (i + BATCH) / max(1, len(bar_wavs))))
    bar_embeds = np.concatenate(bar_embeds_list, axis=0)  # (bar_count, D)
    log.info("       loop_mining: bar embeddings shape=%s", bar_embeds.shape)

    # Enumerate candidates: every downbeat × every length.
    ctx.progress("scoring", 0.75)
    candidates: list[dict] = []
    for length in _LOOP_LENGTHS_BARS:
        for start_bar_idx in range(0, bar_count - length, _DOWNBEAT_HOP):
            window = bar_embeds[start_bar_idx:start_bar_idx + length]
            mu = window.mean(axis=0)
            mu_norm = mu / max(np.linalg.norm(mu), 1e-9)
            # cosine(bar_i, mu_norm); bar embeddings are already L2-normalised.
            sims = window @ mu_norm
            consistency = float(sims.mean())
            if consistency < _MIN_CONSISTENCY:
                continue
            start_s = beats[bars[start_bar_idx]]
            end_s = beats[bars[start_bar_idx + length]]
            candidates.append({
                "start_bar_idx": start_bar_idx,
                "length_bars": length,
                "start_s": start_s,
                "end_s": end_s,
                "consistency": consistency,
                "embedding_mu": mu_norm.astype(np.float32),
            })

    log.info("       loop_mining: %d candidates above threshold %.2f",
             len(candidates), _MIN_CONSISTENCY)

    # Top-N per length, non-overlapping. Greedy: take highest score, drop
    # any overlapping window, repeat.
    selected: list[dict] = []
    for length in _LOOP_LENGTHS_BARS:
        by_score = sorted(
            (c for c in candidates if c["length_bars"] == length),
            key=lambda c: -c["consistency"],
        )
        taken: list[tuple[int, int]] = []  # (start_bar_idx, end_bar_idx)
        for cand in by_score:
            s = cand["start_bar_idx"]
            e = s + cand["length_bars"]
            if any(not (e <= ts or s >= te) for ts, te in taken):
                continue
            taken.append((s, e))
            selected.append(cand)
            if len(taken) >= _TOP_N_PER_LENGTH:
                break

    log.info("       loop_mining: selected %d non-overlapping loops",
             len(selected))

    # Emit cue rows.
    ctx.progress("writing_cues", 0.95)
    for cand in sorted(selected, key=lambda c: c["start_s"]):
        start_ms = int(round(cand["start_s"] * 1000))
        length_ms = int(round((cand["end_s"] - cand["start_s"]) * 1000))
        label = (f"Auto {cand['length_bars']}-bar @ "
                 f"{_format_time(cand['start_s'])}")

        # Zero-shot mood + genre from the mean embedding. Store top-5 of each
        # as JSON; the Player and AI selection layer pick what to surface.
        mu_t = torch.from_numpy(cand["embedding_mu"])
        mood_scores = score_mood(mu_t)
        genre_scores = score_genre(mu_t)
        top_mood = dict(list(mood_scores.items())[:5])
        top_genre = dict(list(genre_scores.items())[:5])

        ctx.cues.append({
            "kind": 4,
            "label": label,
            "loop_type": "auto",
            "pointNumerator": start_ms,
            "pointDenominator": 1000,
            "loopNumerator": length_ms,
            "loopDenominator": 1000,
            "start_bar": cand["start_bar_idx"] + 1,  # 1-indexed for display
            "bars": cand["length_bars"],
            # Base64-encoded float32 bytes so the cue dict JSON-serialises
            # cleanly over httpx. db.replace_analyzer_cues decodes back to
            # raw bytes before INSERT into the BLOB column.
            "embedding": base64.b64encode(cand["embedding_mu"].tobytes()).decode("ascii"),
            "embedding_consistency": round(cand["consistency"], 4),
            "overall_score": round(cand["consistency"], 4),
            "mood_tags": json.dumps({k: round(v, 4) for k, v in top_mood.items()}),
            "genre_hints": json.dumps({k: round(v, 4) for k, v in top_genre.items()}),
        })
