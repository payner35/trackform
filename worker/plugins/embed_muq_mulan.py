"""MuQ-MuLan loader and embedding helpers — shared library for embed + loop_mining.

This module is **not** a stage plugin on its own. It exposes:

- `get_mulan()`        — process-wide singleton, downloads weights from HF on
                         first call (~3 GB), caches in the HF cache volume.
- `embed_audio(wavs)`  — embed one or more 24 kHz mono audio windows
                         (np.ndarray or torch.Tensor) → L2-normalised vectors.
- `embed_texts(texts)` — embed a list of strings → L2-normalised vectors.
- `cosine(a, b)`       — convenience similarity helper.

Model: `OpenMuQ/MuQ-MuLan-large` (Tencent AI Lab, ~700M params, 24 kHz).
Weights are CC-BY-NC 4.0 — fine for self-hosted, blocker for paid SaaS.
LAION-CLAP `larger_clap_music` is the named commercial fallback
(ANALYZER_SPEC §2 / §2.2).

The model itself is loaded lazily because (a) the import is slow and
(b) we want the worker to start up even if no job needs embeddings yet.
"""

from __future__ import annotations

import logging
import threading
from typing import Iterable

import numpy as np
import torch

log = logging.getLogger("worker")

# 24 kHz is hard-coded by MuQ-MuLan; passing audio at any other rate
# produces silently wrong embeddings.
SAMPLE_RATE = 24_000

_mulan = None
_device: str | None = None
_load_lock = threading.Lock()


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    # MPS works but MuQ-MuLan's conformer uses ops that fall back to CPU on
    # MPS today — net slower than CPU. Stick to CPU on Mac.
    return "cpu"


def get_mulan():
    """Return the process-wide MuQ-MuLan model, loading on first call."""
    global _mulan, _device
    if _mulan is not None:
        return _mulan
    with _load_lock:
        if _mulan is not None:
            return _mulan
        from muq import MuQMuLan  # imported lazily — pulls in torch + librosa
        _device = _pick_device()
        log.info("       loading MuQ-MuLan-large on %s (first call, ~3 GB)", _device)
        m = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
        m = m.to(_device).eval()
        _mulan = m
        log.info("       MuQ-MuLan ready")
        return _mulan


def unload_mulan() -> bool:
    """Drop the resident MuQ-MuLan model and return RAM to the OS.

    Called by `structure` before madmom runs — the parent worker can hold
    ~3 GB resident from previous tag rounds, and madmom subprocess peaks
    at ~3.5 GB, so coexistence pushes past the cgroup cap on the 8 GB
    control droplet (Phase 4a). Next `get_mulan()` call reloads (~50 s
    on this hardware).

    PyTorch's CPU caching allocator does not return freed memory to the
    OS via `gc.collect()` alone — the bytes go back to the pool, not to
    the kernel. We `malloc_trim(0)` to force glibc to release fully-free
    pages. Without this the parent worker RSS stays elevated after the
    "unload" and madmom can OOM regardless.

    @returns True if a model was actually freed, False if nothing was loaded.
    """
    global _mulan
    import ctypes
    import gc
    with _load_lock:
        if _mulan is None:
            return False
        log.info("       unloading MuQ-MuLan to free memory before structure stage")
        _mulan = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
            log.info("       malloc_trim'd glibc heap back to OS")
        except OSError as e:
            log.warning("       malloc_trim unavailable on this system: %s", e)
        return True


def embed_audio(wavs) -> torch.Tensor:
    """Embed a batch of mono 24 kHz audio windows.

    `wavs` is either a single 1-D np.ndarray / torch.Tensor of samples,
    or a 2-D batch (B, T). Returns an L2-normalised tensor of shape
    (B, D) on CPU. Caller decides whether to keep them on GPU.
    """
    m = get_mulan()
    if isinstance(wavs, np.ndarray):
        t = torch.from_numpy(wavs.astype(np.float32))
    else:
        t = wavs.float()
    if t.ndim == 1:
        t = t.unsqueeze(0)
    t = t.to(_device)
    with torch.no_grad():
        embeds = m(wavs=t)
    embeds = torch.nn.functional.normalize(embeds, dim=-1)
    return embeds.detach().cpu()


def embed_texts(texts: Iterable[str]) -> torch.Tensor:
    """Embed a list of text prompts → L2-normalised tensor (N, D) on CPU."""
    m = get_mulan()
    with torch.no_grad():
        embeds = m(texts=list(texts))
    embeds = torch.nn.functional.normalize(embeds, dim=-1)
    return embeds.detach().cpu()


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between L2-normalised vectors. Pure dot product."""
    return (a * b).sum(dim=-1)


# ---------------------------------------------------------------------------
# Zero-shot mood / genre tagging
# ---------------------------------------------------------------------------
#
# MuQ-MuLan is a joint music+text embedding, so any text prompt projects
# into the same space as audio. For each candidate tag we author a short
# natural-language description; the cosine between a loop's audio
# embedding and each tag embedding is that tag's similarity score.
#
# The vocab is fixed at plugin load — the text embeddings get computed
# once and cached. Adding/removing a tag is one line below.

_MOOD_PROMPTS: dict[str, str] = {
    "driving":      "a driving, propulsive track with relentless forward momentum",
    "dark":         "a dark, brooding track with ominous atmosphere",
    "uplifting":    "an uplifting, euphoric track that feels triumphant",
    "dreamy":       "a dreamy, ethereal track with soft ambient textures",
    "aggressive":   "an aggressive, hard-hitting track with intense energy",
    "melancholic":  "a melancholic, wistful track with emotional weight",
    "playful":      "a playful, bouncy track with cheerful energy",
    "hypnotic":     "a hypnotic, repetitive track that pulls the listener in",
    "tense":        "a tense, suspenseful track with rising pressure",
    "groovy":       "a groovy, funky track with a tight rhythmic pocket",
}

_GENRE_PROMPTS: dict[str, str] = {
    "techno":         "a techno track with a four-on-the-floor kick and industrial textures",
    "house":          "a house music track with a soulful groove",
    "deep house":     "a deep house track with warm bass and lush chords",
    "tech house":     "a tech house track with a minimal punchy groove",
    "disco":          "a disco track with live drums, strings, and a funky bassline",
    "ambient":        "an ambient track with no beat, just textures and pads",
    "drum and bass":  "a drum and bass track with fast breakbeats and a heavy sub bass",
    "hip hop":        "a hip hop track with sampled drums and a rap vocal",
    "jazz":           "a jazz track with live instrumentation and improvisation",
    "rock":           "a rock track with electric guitars and live drums",
}

# Ordered low → high so we can read out a continuous "energy_value" by
# bucket index (0..N-1) → 0.0..1.0 instead of just a winning label.
_ENERGY_PROMPTS: dict[str, str] = {
    "ambient":      "a low-energy ambient track with no drums and a calm, still atmosphere",
    "chill":        "a chilled, downtempo track with a relaxed groove",
    "groovy":       "a medium-energy groove with a steady danceable beat",
    "driving":      "a high-energy driving club track with relentless propulsion",
    "peak":         "a peak-time, high-intensity banger with maximum drive and pressure",
}
_ENERGY_LABEL_ORDER = list(_ENERGY_PROMPTS.keys())

_VOCAL_PROMPTS: dict[str, str] = {
    "vocal":        "a track dominated by lead vocals, with a singer carrying the melody",
    "instrumental": "an instrumental track with no vocals at all",
}

_PERCUSSION_PROMPTS: dict[str, str] = {
    "percussive":   "a percussion-heavy track with prominent kicks, hats, and rhythmic drums",
    "sparse":       "a track with minimal or no percussion, mostly tones and pads",
}

_BASS_PROMPTS: dict[str, str] = {
    "bass":         "a track with a prominent bassline and strong low-end weight",
    "bassless":     "a track with no audible bass, only mid- and high-frequency content",
}

_MELODIC_PROMPTS: dict[str, str] = {
    "melodic":      "a melodic track with strong melody lines and clear harmonic movement",
    "rhythmic":     "a rhythmic, non-melodic track focused on drums and texture",
}

_tag_cache: dict[str, torch.Tensor] = {}


def _get_tag_embeddings(prompts: dict[str, str], cache_key: str) -> tuple[list[str], torch.Tensor]:
    """Return (tag_names, embeddings tensor) — embeddings computed once."""
    if cache_key in _tag_cache:
        emb = _tag_cache[cache_key]
        # Tag names are stable per cache_key; recover order by lookup.
        return list(prompts.keys()), emb
    names = list(prompts.keys())
    texts = [prompts[n] for n in names]
    emb = embed_texts(texts)
    _tag_cache[cache_key] = emb
    return names, emb


def score_tags(audio_emb: torch.Tensor, prompts: dict[str, str], cache_key: str) -> dict[str, float]:
    """Score one audio embedding against a prompt vocabulary.

    Returns {tag_name: cosine_similarity} sorted desc by score. Caller
    decides whether to top-K, threshold, or store the full distribution.
    """
    names, tag_emb = _get_tag_embeddings(prompts, cache_key)
    if audio_emb.ndim == 1:
        audio_emb = audio_emb.unsqueeze(0)
    # audio_emb: (1, D), tag_emb: (N, D) — both L2-normalised already.
    sims = (audio_emb @ tag_emb.T).squeeze(0)  # (N,)
    pairs = sorted(
        ((names[i], float(sims[i])) for i in range(len(names))),
        key=lambda kv: -kv[1],
    )
    return dict(pairs)


def score_mood(audio_emb: torch.Tensor) -> dict[str, float]:
    return score_tags(audio_emb, _MOOD_PROMPTS, "mood")


def score_genre(audio_emb: torch.Tensor) -> dict[str, float]:
    return score_tags(audio_emb, _GENRE_PROMPTS, "genre")


def score_energy(audio_emb: torch.Tensor) -> tuple[float, str, float]:
    """Map audio → (energy_value 0..1, energy_label, energy_confidence).

    `energy_value` is the bucket-index weighted mean of the softmaxed
    cosine scores — gives a continuous 0..1 number rather than just a
    winning label, so the UI can show "Energy 0.62" not just "groovy".
    `energy_label` is the top bucket name; `energy_confidence` is the
    top-vs-second-bucket margin (0..1).
    """
    scores = score_tags(audio_emb, _ENERGY_PROMPTS, "energy")
    # Softmax-ish normalisation: clip to [0, 1] then divide. Cosine sims
    # against MuQ-MuLan are usually in [-0.1, 0.4]; we want a proper
    # distribution to weight bucket indices.
    vals = [max(0.0, scores[name]) for name in _ENERGY_LABEL_ORDER]
    total = sum(vals) or 1.0
    probs = [v / total for v in vals]
    n = len(_ENERGY_LABEL_ORDER)
    energy_value = sum(probs[i] * (i / max(1, n - 1)) for i in range(n))
    energy_label = max(scores, key=scores.get)
    sorted_vals = sorted(scores.values(), reverse=True)
    energy_confidence = float(max(0.0, sorted_vals[0] - sorted_vals[1]))
    return float(energy_value), energy_label, energy_confidence


def _score_binary_axis(
    audio_emb: torch.Tensor, prompts: dict[str, str], cache_key: str, positive_key: str
) -> float:
    """Two-prompt axis → single 0..1 value for the positive side.

    Softmax over the (positive, negative) cosine sims. Used for
    vocal_density, percussion_density, bass_presence, melodic_presence —
    all of which the UI stores as a single REAL.
    """
    scores = score_tags(audio_emb, prompts, cache_key)
    pos = scores.get(positive_key, 0.0)
    neg = next((v for k, v in scores.items() if k != positive_key), 0.0)
    # Shift so both sides are ≥0, then normalise. Cosine sims can be
    # negative; just clipping to 0 throws away signal when both are
    # negative. Centre on the min instead.
    base = min(pos, neg, 0.0)
    pos_p = pos - base
    neg_p = neg - base
    total = pos_p + neg_p or 1.0
    return float(pos_p / total)


def score_vocal_density(audio_emb: torch.Tensor) -> float:
    return _score_binary_axis(audio_emb, _VOCAL_PROMPTS, "vocal", "vocal")


def score_percussion_density(audio_emb: torch.Tensor) -> float:
    return _score_binary_axis(audio_emb, _PERCUSSION_PROMPTS, "percussion", "percussive")


def score_bass_presence(audio_emb: torch.Tensor) -> float:
    return _score_binary_axis(audio_emb, _BASS_PROMPTS, "bass", "bass")


def score_melodic_presence(audio_emb: torch.Tensor) -> float:
    return _score_binary_axis(audio_emb, _MELODIC_PROMPTS, "melodic", "melodic")
