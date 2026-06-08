"""DJ Loop Worker — generic plugin runner.

Polls the control plane for queued analysis jobs. For each job, runs every
registered analyzer stage in order, POSTs partial results after each stage,
emits `analyzer.track_stage` events for the Player to react to, and emits
`analyzer.track_done` when the last stage finishes.

This file is intentionally generic: it knows about the worker↔control
contract (`DEPLOYMENT_SPEC §4`) but not about any specific analyser.
Adding a new analyser is a new file under `worker/plugins/`; removing one
is removing its import in `worker/plugin.py:load_builtin_plugins`.

Contract: DEPLOYMENT_SPEC.md §4.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import httpx

from plugin import WorkerCtx, load_builtin_plugins, registered_stages

log = logging.getLogger("worker")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
# Silence the per-poll INFO chatter from httpx — we only want our own logs.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def process_job(client: httpx.Client, job: dict) -> None:
    job_id = job["job_id"]
    host_path = job["host_path"]
    audio_path = Path(job["audio_path"])
    content_id: int | None = job.get("content_id")
    kind = job.get("kind") or "analyze"

    log.info("================================================================")
    log.info("JOB %s [%s]: %s", job_id, kind.upper(), host_path)
    log.info("       audio file: %s (%s bytes)", audio_path,
             audio_path.stat().st_size if audio_path.exists() else "missing")
    log.info("================================================================")

    if not audio_path.exists():
        msg = f"audio not found at {audio_path}"
        log.error("job %s: %s", job_id, msg)
        client.post(
            "/v1/worker/event",
            json={"type": "analyzer.track_failed", "host_path": host_path, "error_message": msg},
        )
        return

    # Tag jobs run a single off-pipeline handler, not the registered stages.
    if kind == "tag":
        _process_tag_job(client, job, audio_path, host_path, content_id)
        return

    # Per-loop tagging (ANALYZER_SPEC §2.2, added 2026-06-08). The audio
    # at audio_path IS the loop's audio (already sliced by the Player),
    # so we run MuQ directly on the whole file — no track-relative
    # decoding, no cue table interaction.
    if kind == "loop_tag_v2":
        _process_loop_tag_v2(client, job, audio_path)
        return

    stages = registered_stages()
    if not stages:
        log.error("job %s: no stages registered — load_builtin_plugins() did nothing", job_id)
        return

    # ctx persists across stages within a job so later stages can read
    # what earlier ones produced (e.g. loop_mining needs structure's
    # beat_grid_csv + bars_csv). The runner snapshots columns/cues
    # before each stage and POSTs only the per-stage delta.
    ctx = WorkerCtx(
        audio_path=audio_path,
        job_id=int(job_id),
        host_path=host_path,
        content_id=content_id,
    )

    for stage_index, stage in enumerate(stages):
        # Emit the entry event so the Player's progress bar advances even
        # before any work in the stage produces sub-stage progress.
        client.post(
            "/v1/worker/event",
            json={
                "type": "analyzer.track_stage",
                "host_path": host_path,
                "stage_name": stage.name,
                "stage_index": stage_index + 1,
                "stage_count": len(stages) + 1,  # +1 to reserve headroom for "done"
                "fraction_overall": stage_index / max(len(stages), 1),
                "content_id": content_id,
            },
        )

        # Sub-stage progress flows back to the Player via the same channel
        # so long stages can advance the progress bar during silent periods
        # (Beat This! checkpoint load, etc.).
        def progress(sub_stage: str, fraction: float, _name=stage.name, _idx=stage_index) -> None:
            client.post(
                "/v1/worker/event",
                json={
                    "type": "analyzer.track_stage",
                    "host_path": host_path,
                    "stage_name": f"{_name}.{sub_stage}",
                    "stage_index": _idx + 1,
                    "stage_count": len(stages) + 1,
                    "fraction_overall": fraction,
                    "content_id": content_id,
                },
            )

        ctx.progress = progress
        ctx.content_id = content_id

        # Snapshot so we can POST only what this stage produced.
        columns_before = dict(ctx.columns)
        cues_before_len = len(ctx.cues)

        log.info("JOB %s: running core.%s", job_id, stage.name)
        try:
            stage.func(ctx)
        except Exception as e:
            log.exception("JOB %s: stage %s failed: %s", job_id, stage.name, e)
            client.post(
                "/v1/worker/event",
                json={
                    "type": "analyzer.track_failed",
                    "host_path": host_path,
                    "stage_name": stage.name,
                    "error_message": str(e),
                },
            )
            return

        delta_columns = {k: v for k, v in ctx.columns.items()
                         if k not in columns_before or columns_before[k] != v}
        delta_cues = ctx.cues[cues_before_len:]

        is_last = stage_index == len(stages) - 1
        r = client.post(
            "/v1/worker/result",
            json={
                "job_id": job_id,
                "stage": stage.name,
                "columns": delta_columns,
                "cues": delta_cues,
                "final": is_last,
            },
        )
        r.raise_for_status()
        content_id = r.json().get("content_id") or content_id

    # Terminal event — Player flips into "done" state, persists canonical row.
    client.post(
        "/v1/worker/event",
        json={
            "type": "analyzer.track_done",
            "host_path": host_path,
            "content_id": content_id,
        },
    )

    # Storage cleanup (DEPLOYMENT_SPEC §2 Storage model, 2026-06-08).
    # /srv/audio is a SCRATCH volume — once the analyze pipeline is done,
    # the file has no further role. core.loop_tag (per-loop) carries its
    # own audio in the request body; the worker never reads /srv/audio for
    # tag jobs anymore. Delete here so a 1000-track library doesn't grow
    # into 20 GB of permanent MP3s.
    try:
        if audio_path.exists() and audio_path.is_relative_to(Path("/srv/audio")):
            audio_path.unlink()
            log.info("       cleanup: removed %s", audio_path.name)
    except Exception as e:
        log.warning("       cleanup: failed to remove %s: %s", audio_path, e)

    log.info("JOB %s: DONE (content_id=%s)", job_id, content_id)
    log.info("================================================================")


def _process_tag_job(
    client: httpx.Client,
    job: dict,
    audio_path: Path,
    host_path: str,
    content_id: int | None,
) -> None:
    """On-demand MuQ-MuLan tagging for a specific cue list.

    Triggered by POST /v1/tracks/{content_id}/loops/tag. The job payload
    carries {cue_ids: [...]}. We fetch the cue rows (positions +
    embedded_at_position signature) via GET /v1/tracks/{content_id},
    filter to the requested cue_ids, hand the list to loop_tag.tag_cues,
    and POST the resulting per-cue updates via /v1/worker/result with
    cue_updates set.
    """
    import json as _json

    from plugins.loop_tag import tag_cues

    job_id = job["job_id"]
    payload_raw = job.get("payload") or "{}"
    try:
        payload = _json.loads(payload_raw)
    except _json.JSONDecodeError as e:
        log.exception("tag job %s: bad payload: %s", job_id, e)
        client.post(
            "/v1/worker/event",
            json={"type": "analyzer.track_failed", "host_path": host_path,
                  "stage_name": "loop_tag", "error_message": f"bad payload: {e}"},
        )
        return

    requested_ids = set(payload.get("cue_ids") or [])
    force = bool(payload.get("force") or False)
    if not requested_ids or content_id is None:
        log.warning("tag job %s: empty cue_ids or missing content_id — nothing to do", job_id)
        client.post("/v1/worker/result",
                    json={"job_id": job_id, "stage": "loop_tag",
                          "columns": {}, "cues": [], "cue_updates": [], "final": True})
        return

    # Entry event for the Player's progress UI.
    client.post(
        "/v1/worker/event",
        json={"type": "analyzer.track_stage", "host_path": host_path,
              "stage_name": "loop_tag", "stage_index": 1, "stage_count": 1,
              "fraction_overall": 0.1, "content_id": content_id},
    )

    # Fetch current cue state from the control plane.
    try:
        r = client.get(f"/v1/tracks/{content_id}")
        r.raise_for_status()
        all_cues = r.json().get("cues") or []
    except httpx.HTTPError as e:
        log.exception("tag job %s: failed to fetch track: %s", job_id, e)
        client.post(
            "/v1/worker/event",
            json={"type": "analyzer.track_failed", "host_path": host_path,
                  "stage_name": "loop_tag", "error_message": f"fetch failed: {e}"},
        )
        return

    # Filter to requested cues and shape into loop_tag's input.
    tag_input: list[dict] = []
    for c in all_cues:
        if int(c.get("cue_id") or 0) not in requested_ids:
            continue
        point_num = int(c.get("pointNumerator") or 0)
        point_den = int(c.get("pointDenominator") or 1000)
        loop_num = int(c.get("loopNumerator") or 0)
        loop_den = int(c.get("loopDenominator") or 1000)
        point_ms = int(round(point_num / point_den * 1000))
        length_ms = int(round(loop_num / loop_den * 1000))
        tag_input.append({
            "cue_id": int(c["cue_id"]),
            "point_ms": point_ms,
            "length_ms": length_ms,
            "embedded_at_position": c.get("embedded_at_position") or "",
        })

    log.info("tag job %s: %d cues requested, %d resolved%s", job_id,
             len(requested_ids), len(tag_input), "  [FORCE]" if force else "")

    client.post(
        "/v1/worker/event",
        json={"type": "analyzer.track_stage", "host_path": host_path,
              "stage_name": "loop_tag.embedding", "stage_index": 1, "stage_count": 1,
              "fraction_overall": 0.3, "content_id": content_id},
    )

    try:
        updates = tag_cues(audio_path, tag_input, force=force)
    except Exception as e:
        log.exception("tag job %s: tag_cues failed: %s", job_id, e)
        client.post(
            "/v1/worker/event",
            json={"type": "analyzer.track_failed", "host_path": host_path,
                  "stage_name": "loop_tag", "error_message": str(e)},
        )
        return

    # Drop the "skipped" sentinel from the updates we POST — those are
    # dirty-check no-ops, no DB write needed.
    effective_updates = [u for u in updates if not u.get("skipped")]

    client.post(
        "/v1/worker/result",
        json={"job_id": job_id, "stage": "loop_tag",
              "columns": {}, "cues": [], "cue_updates": effective_updates,
              "final": True},
    )

    client.post(
        "/v1/worker/event",
        json={"type": "analyzer.track_done", "host_path": host_path,
              "content_id": content_id},
    )

    log.info("tag job %s: DONE  %d updates applied, %d skipped",
             job_id, len(effective_updates),
             len(updates) - len(effective_updates))


def _process_loop_tag_v2(client: httpx.Client, job: dict, clip_path: Path) -> None:
    """Per-loop MuQ tagger (ANALYZER_SPEC §2.2).

    The audio at clip_path IS the loop's audio — already sliced by the
    Player. We load it, run MuQ-MuLan once, score every prompt vocab,
    and POST the result dict back to the Service which has a blocking
    HTTP handler waiting on it.

    No cue rows are read or written here. Content-addressable, stateless.
    """
    import base64
    import json as _json
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

    job_id = int(job["job_id"])

    def post_result(result: dict) -> None:
        try:
            client.post("/v1/worker/loop_tag_result",
                        json={"job_id": job_id, "result": result})
        except Exception as e:
            log.exception("loop_tag_v2 %s: failed to post result: %s", job_id, e)

    try:
        import librosa
        wav, _ = librosa.load(str(clip_path), sr=SAMPLE_RATE, mono=True)
        wav = wav.astype(np.float32)
        if wav.size < SAMPLE_RATE:
            log.warning("loop_tag_v2 %s: clip < 1s (%d samples) — running anyway", job_id, wav.size)

        emb = embed_audio(wav).numpy()[0]
        mu_t = torch.from_numpy(emb)

        mood_scores  = score_mood(mu_t)
        genre_scores = score_genre(mu_t)
        top_mood     = dict(list(mood_scores.items())[:5])
        top_genre    = dict(list(genre_scores.items())[:5])

        energy_value, energy_label, energy_confidence = score_energy(mu_t)
        vocal_density      = score_vocal_density(mu_t)
        percussion_density = score_percussion_density(mu_t)
        bass_presence      = score_bass_presence(mu_t)
        melodic_presence   = score_melodic_presence(mu_t)

        consistency = max(top_mood.values()) if top_mood else 0.0

        log.info("       loop_tag_v2 %s: mood=%s genre=%s energy=%.2f(%s) "
                 "vox=%.2f perc=%.2f bass=%.2f mel=%.2f",
                 job_id,
                 next(iter(top_mood.keys()), "?"),
                 next(iter(top_genre.keys()), "?"),
                 energy_value, energy_label,
                 vocal_density, percussion_density,
                 bass_presence, melodic_presence)

        post_result({
            "embedding": base64.b64encode(emb.tobytes()).decode("ascii"),
            "embedding_consistency": round(float(consistency), 4),
            "mood_tags":  _json.dumps({k: round(v, 4) for k, v in top_mood.items()}),
            "genre_hints": _json.dumps({k: round(v, 4) for k, v in top_genre.items()}),
            "energy_value":       round(energy_value, 4),
            "energy_label":       energy_label,
            "energy_confidence":  round(energy_confidence, 4),
            "vocal_density":      round(vocal_density, 4),
            "percussion_density": round(percussion_density, 4),
            "bass_presence":      round(bass_presence, 4),
            "melodic_presence":   round(melodic_presence, 4),
        })
    except Exception as e:
        log.exception("loop_tag_v2 %s: failed: %s", job_id, e)
        post_result({"error": str(e)})


def _sweep_orphan_jobs(client: httpx.Client) -> None:
    """Boot-time crash recovery.

    A worker that died mid-job leaves `analyze_job` rows in `status='in_progress'`
    AND a Player blocked on a `track_done`/`track_failed` event that will
    never come. On startup, ask the control plane to fail every such job,
    fire synthetic terminal events so Players unblock, and delete the
    orphaned scratch audio files. See DEPLOYMENT_SPEC §7 Failure Modes
    + §12 Memory Model for why workers die in the first place.

    Retries because docker-compose may start the worker before the service
    is accepting connections. Backs off up to ~30 s; gives up after that
    (the regular poll loop will eventually still work, just won't sweep).
    """
    delays = [0.5, 1.0, 2.0, 5.0, 10.0]
    for i, delay in enumerate([0.0] + delays):
        if delay > 0:
            time.sleep(delay)
        try:
            r = client.post("/v1/worker/sweep_orphans")
            if r.status_code == 200:
                data = r.json()
                n = data.get("swept", 0)
                if n > 0:
                    log.warning("boot-sweep: failed %d orphaned in-progress job(s) "
                                "from a previous worker that crashed", n)
                else:
                    log.info("boot-sweep: no orphans")
                return
            else:
                log.info("boot-sweep: control plane returned %d (endpoint may be missing)",
                         r.status_code)
                return
        except httpx.HTTPError as e:
            if i < len(delays):
                log.info("boot-sweep: control plane not ready (%s), retrying in %.1fs", e, delays[i])
            else:
                log.warning("boot-sweep: control plane never came up: %s (continuing without sweep)", e)
            continue


def _log_rss_periodically() -> None:
    """Light-weight RSS heartbeat. Helps post-mortem when the worker dies
    near a memory ceiling — last log line shows what was loaded at the time.
    Runs in a daemon thread; one /proc read every 30s is negligible CPU."""
    import threading
    import time as _t

    def _read_rss_mb() -> float:
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        # "VmRSS:    123456 kB"
                        return int(line.split()[1]) / 1024.0
        except Exception:
            pass
        return 0.0

    def _loop():
        while True:
            rss = _read_rss_mb()
            log.info("       RSS: %.0f MB", rss)
            _t.sleep(30)

    t = threading.Thread(target=_loop, name="rss-heartbeat", daemon=True)
    t.start()


def main() -> None:
    control_url = os.environ.get("CONTROL_URL", "http://service:7777")
    poll_interval = float(os.environ.get("POLL_INTERVAL_SECONDS", "5"))

    log.info("worker booting; control_url=%s poll_interval=%ss", control_url, poll_interval)
    load_builtin_plugins()
    stage_names = [s.name for s in registered_stages()]
    log.info("loaded plugins → stages: %s", stage_names)

    _log_rss_periodically()

    with httpx.Client(base_url=control_url, timeout=30.0) as client:
        _sweep_orphan_jobs(client)

        while True:
            try:
                r = client.get("/v1/worker/next")
                if r.status_code == 204:
                    pass
                elif r.status_code == 200:
                    process_job(client, r.json())
                    continue  # check again immediately in case more queued
                else:
                    log.warning("unexpected status %s: %s", r.status_code, r.text[:200])
            except httpx.HTTPError as e:
                log.warning("poll failed: %s", e)
            time.sleep(poll_interval)


if __name__ == "__main__":
    main()
