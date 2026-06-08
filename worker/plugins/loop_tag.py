"""core.loop_tag — on-demand MuQ-MuLan tagging for a specific cue list.

NOT registered as a pipeline stage. Called by the worker's tag-job handler
(see main.py) when a `POST /v1/tracks/{id}/loops/tag` request arrives.
That's the architectural inversion described in ANALYZER_SPEC §2.2:
loop selection is now deterministic (`loop_propose`); MuQ runs only on
loops the user has confirmed.

Each cue describes an 8-bar (or any-length) window of the audio. We
embed the whole window — ~16 seconds at 8 bars/120 BPM — which is at
MuQ-MuLan's native scale (vs the retired bar-by-bar embedding that
collapsed to noise).

Dirty-check: if a cue's current `point_ms + '_' + length_ms` signature
matches its stored `embedded_at_position`, the cue is skipped (already
tagged at this boundary). A typical re-open of a previously-tagged
track produces zero MuQ calls.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from plugins.embed_muq_mulan import (
    SAMPLE_RATE,
    embed_audio,
    score_bass_presence,
    score_energy,
    score_genre,
    score_melodic_presence,
    score_mood,
    score_percussion_density,
    score_vocal_density,
)

log = logging.getLogger("worker")


def _load_audio_mono_24k(audio_path: Path) -> np.ndarray:
    import librosa
    wav, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    return wav.astype(np.float32)


def _signature(point_ms: int, length_ms: int) -> str:
    return f"{int(point_ms)}_{int(length_ms)}"


def tag_cues(audio_path: Path, cues: list[dict[str, Any]], *, force: bool = False) -> list[dict[str, Any]]:
    """Run MuQ-MuLan over the supplied cues, return per-cue update dicts.

    Each input cue must have:
        cue_id, point_ms, length_ms, embedded_at_position (str | None)

    Returns a list of dicts:
        {cue_id, columns: {embedding, embedding_consistency, mood_tags,
                           genre_hints, embedded_at_position}}

    Cues whose signature matches their stored embedded_at_position are
    returned with an empty `columns` dict so the caller can log/skip
    cleanly. The caller (main.py) decides whether to forward those as
    no-op writes or filter them out.
    """
    if not cues:
        return []

    # Audio + model — both lazy; first cue pays the load cost.
    wav: np.ndarray | None = None

    updates: list[dict[str, Any]] = []

    for c in cues:
        cue_id = int(c["cue_id"])
        point_ms = int(c.get("point_ms") or 0)
        length_ms = int(c.get("length_ms") or 0)
        sig = _signature(point_ms, length_ms)
        stored_sig = (c.get("embedded_at_position") or "").strip()

        if stored_sig == sig and stored_sig and not force:
            log.info("       loop_tag: cue %d unchanged (sig=%s) — skipping", cue_id, sig)
            updates.append({"cue_id": cue_id, "columns": {}, "skipped": True})
            continue

        if length_ms <= 0:
            log.warning("       loop_tag: cue %d has length_ms=%d — skipping",
                        cue_id, length_ms)
            continue

        if wav is None:
            log.info("       loop_tag: loading audio at %d Hz", SAMPLE_RATE)
            wav = _load_audio_mono_24k(audio_path)
            log.info("       loop_tag: %.1fs of audio", len(wav) / SAMPLE_RATE)

        start = max(0, int(round((point_ms / 1000.0) * SAMPLE_RATE)))
        end = min(len(wav), int(round(((point_ms + length_ms) / 1000.0) * SAMPLE_RATE)))
        if end - start < SAMPLE_RATE:  # < 1 second of audio
            log.warning("       loop_tag: cue %d window too short (%d samples) — skipping",
                        cue_id, end - start)
            continue

        window = wav[start:end].astype(np.float32)

        # Whole-window embedding — single forward pass through MuQ-MuLan.
        # Shape (1, D) on CPU, L2-normalised by embed_audio.
        emb = embed_audio(window).numpy()[0]
        mu_t = torch.from_numpy(emb)

        # Zero-shot mood + genre, top-5 of each.
        mood_scores = score_mood(mu_t)
        genre_scores = score_genre(mu_t)
        top_mood = dict(list(mood_scores.items())[:5])
        top_genre = dict(list(genre_scores.items())[:5])

        # Continuous axes — single REAL each (ANALYZER_SPEC §2.2).
        energy_value, energy_label, energy_confidence = score_energy(mu_t)
        vocal_density = score_vocal_density(mu_t)
        percussion_density = score_percussion_density(mu_t)
        bass_presence = score_bass_presence(mu_t)
        melodic_presence = score_melodic_presence(mu_t)

        # Consistency for whole-window embedding is conceptually different
        # (no bar-level breakdown), but we still want a number for the UI
        # consistency display. Use the max mood score as a proxy: a strong
        # single mood = a recognisable musical character.
        consistency = max(top_mood.values()) if top_mood else 0.0

        log.info("       loop_tag: cue %d  emb_dim=%d  mood=%s  genre=%s  "
                 "energy=%.2f(%s)  vox=%.2f  perc=%.2f  bass=%.2f  mel=%.2f",
                 cue_id, emb.shape[0],
                 next(iter(top_mood.keys()), "?"),
                 next(iter(top_genre.keys()), "?"),
                 energy_value, energy_label,
                 vocal_density, percussion_density,
                 bass_presence, melodic_presence)

        updates.append({
            "cue_id": cue_id,
            "columns": {
                "embedding": base64.b64encode(emb.tobytes()).decode("ascii"),
                "embedding_consistency": round(float(consistency), 4),
                "mood_tags": json.dumps({k: round(v, 4) for k, v in top_mood.items()}),
                "genre_hints": json.dumps({k: round(v, 4) for k, v in top_genre.items()}),
                "energy_value": round(energy_value, 4),
                "energy_label": energy_label,
                "energy_confidence": round(energy_confidence, 4),
                "vocal_density": round(vocal_density, 4),
                "percussion_density": round(percussion_density, 4),
                "bass_presence": round(bass_presence, 4),
                "melodic_presence": round(melodic_presence, 4),
                "embedded_at_position": sig,
            },
        })

    return updates
