"""core.structure plugin — madmom RNN + DBN.

**Subprocess model (DEPLOYMENT_SPEC §7 Memory model, 2026-06-08).**

This plugin invokes madmom in a short-lived child process rather than
loading it into the main worker. Reason: the worker also holds MuQ-MuLan
(~3 GB) for the per-loop tagging path, and madmom's DBN HMM decode adds
another ~1.5–2 GB transient on long tracks. Combined footprint blew past
Docker Desktop's macOS VM ceiling (~7.6 GiB) and OOM-killed the worker
mid-decode — silent failure, no terminal event, Player hung. The
subprocess boundary makes memory release deterministic: kernel reclaims
the address space on process exit, no glibc-arena holdback, no PyTorch
allocator pool retention.

Cost: ~1-2 s subprocess startup per track (Python + madmom import).
Dwarfed by the 60-90 s of actual decode on Mac dev.

In production with adequate RAM (GPU droplet, 32+ GB system) this
subprocess boundary is still correct but unnecessary; it costs nothing
to keep, and the subprocess interface naturally becomes the
inter-container interface when we ship the per-stage container split
(DEPLOYMENT_SPEC §7 future direction).

Output (same contract as structure_beat_this.py):
    tempo                    — BPM × 100 integer (ANALYZER_SPEC §4.5)
    beat_grid_csv            — comma-separated beat times (seconds)
    bars_csv                 — comma-separated beat indices of downbeats
    start_anchor_beat_index  — first downbeat (no projection — the DBN's
                                phase is already globally consistent)
    end_anchor_beat_index    — last downbeat
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

from plugin import WorkerCtx, analyzer_stage

log = logging.getLogger("worker")


@analyzer_stage(name="structure", order=20)
def structure(ctx: WorkerCtx) -> None:
    ctx.progress("running_inference", 0.20)
    log.info("       spawning madmom subprocess on %s", ctx.audio_path.name)

    # Per-stage subprocess. Memory returns to the OS on exit — see module
    # docstring. Timeout is generous to cover long extended mixes on Mac
    # dev where qemu emulation is ~3× slower than native.
    # In the worker container `run_madmom_subprocess.py` sits flat under
    # /app/ alongside main.py (not as a package). Invoke by path.
    try:
        proc = subprocess.run(
            [sys.executable, "/app/run_madmom_subprocess.py",
             str(ctx.audio_path)],
            capture_output=True,
            timeout=15 * 60,
            check=False,
        )
    except subprocess.TimeoutExpired:
        log.error("madmom subprocess timed out after 15 min on %s",
                  ctx.audio_path.name)
        raise RuntimeError("madmom subprocess timed out") from None

    # Forward subprocess stderr (its own log lines) into our log so the
    # full picture stays in one place.
    if proc.stderr:
        for line in proc.stderr.decode("utf-8", errors="replace").splitlines():
            if line.strip():
                log.info("       %s", line)

    if proc.returncode != 0:
        log.error("madmom subprocess exited with code %d", proc.returncode)
        raise RuntimeError(f"madmom subprocess failed (exit {proc.returncode})")

    ctx.progress("decoding_grid", 0.70)

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        log.error("madmom subprocess returned unparseable stdout: %s", e)
        raise

    beats_list: list[float] = list(payload.get("beats") or [])
    bar_indices: list[int] = list(payload.get("downbeat_indices") or [])

    log.info("       MADMOM RAW: first 5 beats     = %s",
             [round(t, 3) for t in beats_list[:5]])
    log.info("       MADMOM RAW: first 5 downbeat indices = %s", bar_indices[:5])
    if len(beats_list) >= 6:
        intervals_head = [round(beats_list[i + 1] - beats_list[i], 3)
                          for i in range(5)]
        log.info("       MADMOM RAW: first 5 inter-beat intervals = %s s",
                 intervals_head)

    # Tempo from CUMULATIVE average (not median IBI). The two-anchor warp
    # in the Player stretches the grid linearly between the first and last
    # beat positions, so the displayed BPM has to match that stretch or
    # the grid drifts. Median IBI can be ~0.3% off from cumulative when
    # individual beats are slightly microtimed — over a 6-minute track
    # that's 2 beats of accumulated drift. Cumulative average is the only
    # tempo consistent with the grid the user actually sees.
    if len(beats_list) >= 2:
        span = beats_list[-1] - beats_list[0]
        n_intervals = len(beats_list) - 1
        seconds_per_beat = span / n_intervals if span > 0 else 0.0
        tempo_x100 = round(60.0 / seconds_per_beat * 100) if seconds_per_beat > 0 else 0
    else:
        tempo_x100 = 0

    # The DBN's first downbeat IS bar 1. No phase projection needed —
    # globally optimised. Last downbeat = end anchor for symmetry.
    start_anchor_beat_index = bar_indices[0] if bar_indices else None
    end_anchor_beat_index = bar_indices[-1] if bar_indices else None

    if bar_indices and len(bar_indices) >= 2:
        log.info("       MADMOM: %d downbeats, %d intra-bar beats. "
                 "start anchor=beat%d (%.2fs)  end anchor=beat%d (%.2fs)",
                 len(bar_indices), len(beats_list) - len(bar_indices),
                 start_anchor_beat_index, beats_list[start_anchor_beat_index],
                 end_anchor_beat_index, beats_list[end_anchor_beat_index])

    ctx.columns["tempo"] = tempo_x100
    ctx.columns["beat_grid_csv"] = ",".join(str(b) for b in beats_list)
    ctx.columns["bars_csv"] = ",".join(str(b) for b in bar_indices)
    if start_anchor_beat_index is not None:
        ctx.columns["start_anchor_beat_index"] = int(start_anchor_beat_index)
    if end_anchor_beat_index is not None:
        ctx.columns["end_anchor_beat_index"] = int(end_anchor_beat_index)
