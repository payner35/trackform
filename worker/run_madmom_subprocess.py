"""Standalone madmom DBN downbeat tracker — designed to be invoked as a
short-lived subprocess from the main worker.

**Why a subprocess?** The main worker holds MuQ-MuLan resident (~3 GB) for
the per-loop tagging path. madmom's DBN HMM decode allocates another
~1.5–2 GB transient on long tracks. On Mac dev (Docker Desktop VM
ceiling ~7.6 GiB), the combined footprint blows past the limit and the
worker process gets OOM-killed mid-decode, taking the whole stage with
it. By running madmom in a subprocess that exits when done, the kernel
reclaims its memory deterministically — no allocator-pool retention, no
glibc-arena holdback. See DEPLOYMENT_SPEC §7 "Memory model" for the
broader rationale and how this evolves into per-stage containers.

Usage (from `structure_madmom.py`):

    subprocess.run(
        [sys.executable, "/app/run_madmom_subprocess.py", str(audio_path)],
        capture_output=True, timeout=15*60,
    )

The script sits flat under /app/ in the worker container (alongside
main.py / plugin.py) — not as a package — so invoke by path.

The script writes a JSON document to stdout:

    {
      "beats": [0.020, 0.510, 1.000, ...],     // every beat, in seconds
      "downbeat_indices": [0, 4, 8, ...]       // indices into beats[]
    }

Stderr carries log lines for the parent to forward into its own log.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Send our log lines to stderr so stdout stays pure JSON.
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[madmom-subproc] %(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("madmom_subproc")


def run(audio_path: Path) -> dict:
    """Run madmom RNN + DBN on the file. Returns JSON-able dict."""
    log.info("loading madmom RNN + DBN (subprocess-local)")
    from madmom.features.downbeats import (
        DBNDownBeatTrackingProcessor,
        RNNDownBeatProcessor,
    )

    rnn = RNNDownBeatProcessor()
    dbn = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)

    log.info("RNN on %s", audio_path.name)
    activations = rnn(str(audio_path))   # (T, 2) — beats, downbeats

    log.info("DBN HMM decode")
    grid = dbn(activations)              # [(time_s, beat_in_bar), ...]

    beats = [round(float(t), 4) for t, _b in grid]
    downbeat_indices = [i for i, (_t, b) in enumerate(grid) if int(b) == 1]

    log.info("done: %d beats, %d downbeats", len(beats), len(downbeat_indices))
    return {"beats": beats, "downbeat_indices": downbeat_indices}


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m worker.run_madmom_subprocess <audio_path>",
              file=sys.stderr)
        return 2

    audio_path = Path(sys.argv[1])
    if not audio_path.exists():
        log.error("audio not found: %s", audio_path)
        return 1

    try:
        result = run(audio_path)
    except Exception as e:
        log.exception("madmom failed: %s", e)
        return 1

    json.dump(result, sys.stdout)
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
