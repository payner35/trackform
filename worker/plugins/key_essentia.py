"""core.key plugin — Essentia KeyExtractor with the EDMA profile.

EDMA (Electronic Dance Music Annotations) is tuned for EDM and outperforms
Krumhansl / Shaath / Madmom CNN on dance material. AGPL licence is fine
because the Service is not distributed as a binary (ANALYZER_SPEC §2).

Output: `musical_key` ("A minor"), `camelot_key` ("8A"), `key_confidence`
(0..1). The Camelot wheel mapping is hard-coded — there's only one of it.
"""

from __future__ import annotations

import logging
from pathlib import Path

from plugin import WorkerCtx, analyzer_stage

log = logging.getLogger("worker")

# Standard Camelot wheel mapping. Essentia returns key as 'C'..'B' (with
# sharps, no flats) and scale as 'major'|'minor', so the flat-key entries
# below are defensive in case the model ever changes its convention.
_CAMELOT: dict[tuple[str, str], str] = {
    ("C",  "major"):  "8B", ("C",  "minor"):  "5A",
    ("C#", "major"):  "3B", ("C#", "minor"): "12A",
    ("Db", "major"):  "3B", ("Db", "minor"): "12A",
    ("D",  "major"): "10B", ("D",  "minor"):  "7A",
    ("D#", "major"):  "5B", ("D#", "minor"):  "2A",
    ("Eb", "major"):  "5B", ("Eb", "minor"):  "2A",
    ("E",  "major"): "12B", ("E",  "minor"):  "9A",
    ("F",  "major"):  "7B", ("F",  "minor"):  "4A",
    ("F#", "major"):  "2B", ("F#", "minor"): "11A",
    ("Gb", "major"):  "2B", ("Gb", "minor"): "11A",
    ("G",  "major"):  "9B", ("G",  "minor"):  "6A",
    ("G#", "major"):  "4B", ("G#", "minor"):  "1A",
    ("Ab", "major"):  "4B", ("Ab", "minor"):  "1A",
    ("A",  "major"): "11B", ("A",  "minor"):  "8A",
    ("A#", "major"):  "6B", ("A#", "minor"):  "3A",
    ("Bb", "major"):  "6B", ("Bb", "minor"):  "3A",
    ("B",  "major"):  "1B", ("B",  "minor"): "10A",
}


@analyzer_stage(name="key", order=25)
def key(ctx: WorkerCtx) -> None:
    """Populate ctx.columns with musical_key / camelot_key / key_confidence."""
    import essentia.standard as es

    log.info("       running essentia KeyExtractor(profileType='edma')")
    audio = es.MonoLoader(filename=str(ctx.audio_path), sampleRate=44100)()
    detected_key, scale, strength = es.KeyExtractor(profileType="edma")(audio)

    musical_key = f"{detected_key} {scale}"
    camelot_key = _CAMELOT.get((detected_key, scale))
    if camelot_key is None:
        log.warning("       no Camelot mapping for (%s, %s)", detected_key, scale)

    log.info("       → musical_key='%s'  camelot_key='%s'  strength=%.3f",
             musical_key, camelot_key, strength)

    ctx.columns["musical_key"] = musical_key
    ctx.columns["key_confidence"] = round(float(strength), 4)
    if camelot_key is not None:
        ctx.columns["camelot_key"] = camelot_key
