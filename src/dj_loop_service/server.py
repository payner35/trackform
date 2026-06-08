"""FastAPI server — the HTTP face of the Analyzer.

Player uploads audio files here; Service runs the pipeline and writes to
library.db. The Service never reaches into the Player's filesystem.

One endpoint for now: POST /v1/analyze (multipart upload).
"""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import tempfile
import threading
import uuid

# App-level logger separate from uvicorn.access (which spams every HTTP hit).
# Set DLS_LOG_LEVEL=DEBUG to also see HTTP detail; default keeps it concise.
log = logging.getLogger("dls")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s dls: %(message)s",
        datefmt="%H:%M:%S",
    ))
    log.addHandler(h)
    log.setLevel(os.environ.get("DLS_LOG_LEVEL", "INFO"))
    log.propagate = False
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response

from . import jobs
from .auth import bearer_token_middleware, check_ws_token
from .config import Config
from .db import (
    connect,
    get_or_create_key,
    replace_analyzer_cues,
    transaction,
    update_analyzer_cues,
    upsert_content,
)
from .events import event_bus
from .pipeline import analyze_file, load_builtin_plugins

app = FastAPI(
    title="DJ Loop Service",
    version="0.1.0",
    description="Audio analysis service for DJ Loop Player.",
)
app.middleware("http")(bearer_token_middleware)


@app.on_event("startup")
def _startup() -> None:
    load_builtin_plugins()


@app.get("/v1/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Tracks (read API)
# ---------------------------------------------------------------------------

_CONTENT_LIST_COLUMNS = """
    c.content_id, c.title, c.duration, c.tempo, c.samplingRate,
    c.file_path_absolute, c.file_hash, c.fingerprint,
    c.user_id, c.analysis_source, c.analysis_version,
    c.beat_grid_csv, c.bars_csv,
    c.start_anchor_beat_index, c.end_anchor_beat_index,
    c.camelot_key, c.key_confidence,
    c.created_at, c.updated_at,
    a.name AS artist,
    al.name AS album,
    g.name AS genre,
    k.name AS musical_key
"""

_CONTENT_FROM_JOINS = """
    FROM content c
    LEFT JOIN artist a ON c.artist_id = a.artist_id
    LEFT JOIN album al ON c.album_id  = al.album_id
    LEFT JOIN genre g  ON c.genre_id  = g.genre_id
    LEFT JOIN key k    ON c.key_id    = k.key_id
"""


def _row_to_track(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    # Truncate the fingerprint for list responses — it's ~3 KB per track,
    # too heavy to send wholesale. Detail endpoint returns full fingerprint.
    if "fingerprint" in d and d["fingerprint"]:
        d["fingerprint_len"] = len(d["fingerprint"])
    return d


@app.get("/v1/tracks")
def list_tracks(
    user_id: str | None = Query(None, description="Filter by owner. Omit for all users."),
    limit: int = Query(100, ge=1, le=1000, description="Max rows to return."),
    offset: int = Query(0, ge=0, description="Pagination offset."),
    q: str | None = Query(None, description="Free-text match against title/artist/album."),
) -> JSONResponse:
    """List tracks the Service has analyzed."""
    db_path = _required_db_path()
    conn = connect(db_path)
    try:
        wheres: list[str] = []
        params: list[Any] = []
        if user_id is not None:
            wheres.append("c.user_id = ?")
            params.append(user_id)
        if q:
            wheres.append("(c.title LIKE ? OR a.name LIKE ? OR al.name LIKE ?)")
            qx = f"%{q}%"
            params.extend([qx, qx, qx])

        where = ("WHERE " + " AND ".join(wheres)) if wheres else ""
        total = conn.execute(
            f"SELECT COUNT(*) {_CONTENT_FROM_JOINS} {where}", params
        ).fetchone()[0]

        rows = conn.execute(
            f"SELECT {_CONTENT_LIST_COLUMNS} {_CONTENT_FROM_JOINS} {where} "
            f"ORDER BY c.created_at DESC LIMIT ? OFFSET ?",
            (*params, limit, offset),
        ).fetchall()

        return JSONResponse(
            {
                "total": int(total),
                "limit": limit,
                "offset": offset,
                "tracks": [
                    {**_row_to_track(r), "fingerprint": None}  # strip heavy field
                    for r in rows
                ],
            }
        )
    finally:
        conn.close()


@app.get("/v1/tracks/by-hash/{file_hash}")
def get_track_by_hash(file_hash: str, user_id: str | None = Query(None)) -> JSONResponse:
    """Hash probe — the universal content-addressable lookup.

    The Player calls this before deciding to upload audio. If the Service
    already has analysis for this file_hash, return the full track detail
    so the Player can populate its local DB without re-uploading 20 MB and
    without burning a worker job. Reuses the same payload shape as
    GET /v1/tracks/{content_id} so the Player has a single parser.
    See ANALYZER_SPEC §4.4 — content-addressable client probe.
    """
    db_path = _required_db_path()
    conn = connect(db_path)
    try:
        match = conn.execute(
            "SELECT content_id FROM content WHERE file_hash=? LIMIT 1",
            (file_hash,),
        ).fetchone()
        if not match:
            raise HTTPException(status_code=404, detail=f"file_hash {file_hash[:16]!r} not found")
        # Must also have analyzer cues — a row exists from core.load alone
        # isn't "analysis is done." Mirrors the POST /v1/analyze short-circuit.
        has_analysis = bool(
            conn.execute(
                "SELECT 1 FROM cue WHERE content_id=? AND source='analyzer' LIMIT 1",
                (int(match[0]),),
            ).fetchone()
        )
        if not has_analysis:
            raise HTTPException(status_code=404, detail=f"file_hash {file_hash[:16]!r} known but not analyzed")
    finally:
        conn.close()

    # Delegate to the existing single-track handler so the response shape
    # stays in one place. user_id filtering, joins, cue projection — all reused.
    return get_track(int(match[0]), user_id=user_id)


@app.get("/v1/tracks/{content_id}")
def get_track(content_id: int, user_id: str | None = Query(None)) -> JSONResponse:
    """Full detail for one track, including cues and full fingerprint."""
    db_path = _required_db_path()
    conn = connect(db_path)
    try:
        row = conn.execute(
            f"SELECT {_CONTENT_LIST_COLUMNS} {_CONTENT_FROM_JOINS} WHERE c.content_id = ?",
            (content_id,),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"content_id {content_id} not found")
        if user_id is not None and row["user_id"] != user_id:
            # Don't leak existence to wrong-user requests.
            raise HTTPException(status_code=404, detail=f"content_id {content_id} not found")

        cues = [
            dict(c)
            for c in conn.execute(
                "SELECT cue_id, kind, label, loop_type, "
                "pointNumerator, pointDenominator, loopNumerator, loopDenominator, "
                "start_bar, bars, bpm, musical_key, camelot_key, "
                "energy_value, energy_label, energy_confidence, overall_score, "
                "vocal_density, percussion_density, bass_presence, melodic_presence, "
                "embedding_consistency, mood_tags, genre_hints, "
                "embedded_at_position "
                "FROM cue WHERE content_id = ? ORDER BY pointNumerator",
                (content_id,),
            )
        ]

        return JSONResponse({**_row_to_track(row), "cues": cues})
    finally:
        conn.close()


@app.post("/v1/tracks/{content_id}/loops/tag")
def loops_tag(content_id: int, body: dict[str, Any]) -> JSONResponse:
    """Enqueue an on-demand MuQ-MuLan tag job for the supplied cue list.

    Body: {"cue_ids": [<int>, ...]}.

    The worker (see worker/main.py tag-job handler) embeds each cue's
    current 8-bar window, scores moods + genres, and writes back via
    POST /v1/worker/result with cue_updates. Cues whose stored
    `embedded_at_position` already matches their current
    `pointNumerator+'_'+loopNumerator` are skipped (ANALYZER_SPEC §2.2).
    """
    import json as _json

    cue_ids = body.get("cue_ids") or []
    if not isinstance(cue_ids, list) or not all(isinstance(x, int) for x in cue_ids):
        raise HTTPException(status_code=400, detail="cue_ids must be a list of int")

    # Player-supplied current cue positions (Player is source of truth for
    # positions; service applies these before the tag job runs so MuQ
    # embeds the user-edited windows, not stale ones).
    cues_in = body.get("cues") or []

    # Cross-system identity resolution. Player and Service maintain
    # independent content_id sequences, so the URL content_id alone is
    # unreliable. ANALYZER_SPEC §4.4 dedup chain: file_hash (universal
    # SHA-256) → host_path (per-machine) → URL content_id (legacy).
    file_hash_hint = body.get("file_hash") or ""
    host_path_hint = body.get("host_path") or ""
    force = bool(body.get("force") or False)

    db_path = _required_db_path()
    conn = connect(db_path)
    try:
        with transaction(conn):
            row = None
            if file_hash_hint:
                row = conn.execute(
                    "SELECT content_id, file_path_absolute, user_id "
                    "FROM content WHERE file_hash=?",
                    (file_hash_hint,),
                ).fetchone()
                if row:
                    content_id = int(row[0])
            if row is None and host_path_hint:
                row = conn.execute(
                    "SELECT content_id, file_path_absolute, user_id "
                    "FROM content WHERE file_path_absolute=?",
                    (host_path_hint,),
                ).fetchone()
                if row:
                    content_id = int(row[0])
            if row is None:
                row = conn.execute(
                    "SELECT content_id, file_path_absolute, user_id "
                    "FROM content WHERE content_id=?",
                    (content_id,),
                ).fetchone()
            if not row:
                raise HTTPException(
                    status_code=404,
                    detail=(f"content not found "
                            f"(id={content_id}, "
                            f"file_hash={file_hash_hint[:16]!r}, "
                            f"host_path={host_path_hint!r})")
                )
            host_path, user_id = row[1], row[2]

            # DEPRECATED — see ANALYZER_SPEC §2.2 (retired 2026-06-08).
            # Resolver kept temporarily so the Player keeps working during
            # the migration to per-loop tagging. Uses GREEDY position
            # matching: each Player cue claims the closest Service cue
            # not already claimed in this request. Prevents the duplicate-id
            # collision the previous "smallest |Δ|" approach produced when
            # multiple edited cues clustered near the same time.
            position_to_service_cid: dict[tuple[int, int], int] = {}
            available_service: list[tuple[int, int, int]] = []  # (cid, point_ms, length_ms)
            for srow in conn.execute(
                "SELECT cue_id, pointNumerator, loopNumerator "
                "FROM cue WHERE content_id=? AND source='analyzer'",
                (content_id,),
            ):
                cid, sp, sl = int(srow[0]), int(srow[1]), int(srow[2])
                position_to_service_cid[(sp, sl)] = cid
                available_service.append((cid, sp, sl))

            resolved_cue_ids: list[int] = []
            claimed: set[int] = set()
            for c in cues_in:
                point_ms = int(c.get("point_ms") or 0)
                length_ms = int(c.get("length_ms") or 0)
                if length_ms <= 0:
                    continue
                service_cid = position_to_service_cid.get((point_ms, length_ms))
                if service_cid is None or service_cid in claimed:
                    # Pick the closest UNCLAIMED Service cue.
                    best = None
                    best_delta = None
                    for cid, sp, _sl in available_service:
                        if cid in claimed:
                            continue
                        d = abs(sp - point_ms)
                        if best_delta is None or d < best_delta:
                            best_delta = d
                            best = cid
                    if best is None:
                        continue   # ran out of Service cues; skip extras
                    service_cid = best
                claimed.add(service_cid)
                conn.execute(
                    "UPDATE cue SET pointNumerator=?, pointDenominator=1000, "
                    "loopNumerator=?, loopDenominator=1000 "
                    "WHERE cue_id=? AND content_id=?",
                    (point_ms, length_ms, service_cid, content_id),
                )
                resolved_cue_ids.append(service_cid)

            # Fallback: if the Player didn't send a `cues` body (legacy
            # callers), use whatever cue_ids it provided. Worker will
            # log "0 resolved" if they don't exist.
            if not resolved_cue_ids and cue_ids:
                resolved_cue_ids = cue_ids

            # Find the latest analyze_job for this content to recover the
            # audio_path on /srv/audio/. If the file has been TTL-swept,
            # the worker will fail the job and the Player can re-upload.
            jrow = conn.execute(
                "SELECT audio_path FROM analyze_job "
                "WHERE content_id=? AND kind='analyze' "
                "ORDER BY created_at DESC LIMIT 1",
                (content_id,),
            ).fetchone()
            audio_path_str = jrow[0] if jrow else ""

            payload = _json.dumps({"cue_ids": resolved_cue_ids, "force": force})
            job_id = jobs.enqueue(
                conn,
                host_path=host_path,
                user_id=user_id or "local",
                audio_path=audio_path_str,
                content_id=content_id,
                kind="tag",
                payload=payload,
            )

        log.info("→ tag enqueued  content_id=%d  player_cue_ids=%s  service_cue_ids=%s  job_id=%d",
                 content_id, cue_ids, resolved_cue_ids, job_id)
        return JSONResponse({
            "ok": True,
            "job_id": job_id,
            "content_id": content_id,
            "cue_ids": resolved_cue_ids,
        })
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Per-loop tagging (ANALYZER_SPEC §2.2, added 2026-06-08)
#
# One HTTP request per loop, carrying the loop's audio bytes. No cue_id
# crosses the wire — purely content-addressable by (file_hash, point_ms,
# length_ms). The Service writes the audio to a scratch file, enqueues a
# worker job, and blocks until the worker POSTs the result back. The
# scratch file is deleted on the way out, success or fail.
#
# Result delivery: keyed by job_id in an in-memory dict, signalled by a
# threading.Event. Worker POSTs to /v1/worker/loop_tag_result.
# ---------------------------------------------------------------------------

_loop_tag_results: dict[int, dict[str, Any]] = {}
_loop_tag_events: dict[int, threading.Event] = {}
_loop_tag_lock = threading.Lock()


@app.post("/v1/loops/tag")
def loop_tag(
    audio: UploadFile = File(..., description="The loop's audio bytes (WAV, any sr/channels)."),
    file_hash: str = Form(..., description="Universal SHA-256 of the parent track's audio sample stream."),
    point_ms: int = Form(..., description="Loop start in ms within the parent track (natural key)."),
    length_ms: int = Form(..., description="Loop length in ms (natural key)."),
    force: bool = Form(False, description="Re-tag even if (file_hash, point_ms, length_ms) was tagged before."),
    timeout_s: float = Form(240.0, description="How long to wait for the worker before 504-ing. Default 240s covers the worst case where 6 parallel Player requests queue at a single serial worker that's also paying the ~20s MuQ-MuLan first-call load."),
) -> JSONResponse:
    """Per-loop MuQ-MuLan tagging. Content-addressable, no cue_id."""
    if length_ms <= 0:
        raise HTTPException(status_code=400, detail="length_ms must be > 0")

    # Resolve content_id by file_hash so we can persist the result on the
    # Service side too (optional cache; Player is the source of truth).
    db_path = _required_db_path()
    conn = connect(db_path)
    content_id: int | None = None
    try:
        row = conn.execute(
            "SELECT content_id FROM content WHERE file_hash=?",
            (file_hash,),
        ).fetchone()
        if row:
            content_id = int(row[0])
    finally:
        conn.close()

    # Persist the loop audio to the scratch volume. The worker reads from
    # the same volume — no signed URLs in local dev. Unique name so
    # concurrent requests don't collide.
    audio_dir = Path("/srv/audio")
    audio_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(audio.filename or "loop.wav").suffix or ".wav"
    clip_path = audio_dir / f"loop_{uuid.uuid4().hex}{suffix}"
    with open(clip_path, "wb") as out:
        shutil.copyfileobj(audio.file, out)

    # Enqueue the worker job. Payload carries everything the worker needs;
    # the worker writes its result back into _loop_tag_results via the
    # /v1/worker/loop_tag_result endpoint, signalling the threading.Event
    # this handler is about to wait on.
    import json as _json
    payload = _json.dumps({
        "kind": "loop_tag_v2",
        "clip_path": str(clip_path),
        "point_ms": int(point_ms),
        "length_ms": int(length_ms),
        "force": bool(force),
        "file_hash": file_hash,
    })

    conn = connect(db_path)
    try:
        with transaction(conn):
            job_id = jobs.enqueue(
                conn,
                host_path=file_hash,         # not a real host path; just satisfies NOT NULL
                user_id="local",
                audio_path=str(clip_path),
                content_id=content_id,
                kind="loop_tag_v2",
                payload=payload,
            )
    finally:
        conn.close()

    # Register the event BEFORE waiting so the worker can't post-and-set
    # before we're listening.
    event = threading.Event()
    with _loop_tag_lock:
        _loop_tag_events[job_id] = event

    log.info("→ loop_tag enqueued  job_id=%d  point_ms=%d  length_ms=%d  force=%s",
             job_id, point_ms, length_ms, force)

    got_result = event.wait(timeout=timeout_s)

    with _loop_tag_lock:
        result = _loop_tag_results.pop(job_id, None)
        _loop_tag_events.pop(job_id, None)

    # Scratch file is consumed; delete regardless of outcome.
    try:
        clip_path.unlink(missing_ok=True)
    except Exception:
        pass

    if not got_result or result is None:
        raise HTTPException(
            status_code=504,
            detail=f"loop_tag job_id={job_id} timed out after {timeout_s}s",
        )
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return JSONResponse({
        "ok": True,
        "job_id": job_id,
        "content_id": content_id,
        "point_ms": point_ms,
        "length_ms": length_ms,
        **result,
    })


@app.post("/v1/worker/loop_tag_result")
def loop_tag_result(body: dict[str, Any]) -> JSONResponse:
    """Worker calls this with the MuQ-MuLan result for a loop_tag_v2 job."""
    job_id = int(body.get("job_id") or 0)
    result = body.get("result") or {}
    if job_id <= 0 or not isinstance(result, dict):
        raise HTTPException(status_code=400, detail="job_id (int) and result (dict) required")
    with _loop_tag_lock:
        _loop_tag_results[job_id] = result
        event = _loop_tag_events.get(job_id)
    if event is not None:
        event.set()
    log.info("← loop_tag result   job_id=%d  keys=%s", job_id, list(result.keys())[:6])
    return JSONResponse({"ok": True})


@app.post("/v1/analyze")
async def analyze(
    file: UploadFile = File(..., description="The audio file (multipart upload)."),
    host_path: str = Form(..., description="Player's host-local path for content.file_path_absolute."),
    user_id: str = Form("local", description="Owner user_id (multi-tenancy, ONELIBRARY_SPEC §8.9.9)."),
    force: bool = Form(False, description="Bypass cache short-circuit; always re-run the heavy pipeline. Used by 'Regenerate Loops'."),
) -> JSONResponse:
    """Run the light identity stages here and enqueue the heavy stages for the worker.

    Phase 1 of the GPU-on-demand pattern (DEPLOYMENT_SPEC §2.1): the control
    plane runs `core.load` (hash + fingerprint + ID3) synchronously so a row
    is created immediately and the file is content-addressable. The audio is
    then persisted on the shared volume and a job is enqueued for the GPU
    worker, which will run structure / key / loop_mining / embed and POST
    results back via /v1/worker/result.
    """
    db_path = _required_db_path()

    # Persist the audio to the shared volume under /srv/audio/. The worker
    # container reads from the same volume — no signed URLs needed in local
    # dev. Production uses /v1/worker/audio/{job_id} with a signed link.
    audio_dir = Path("/srv/audio")
    audio_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "audio").suffix or ".bin"
    fd, audio_path_str = tempfile.mkstemp(prefix="job_", suffix=suffix, dir=audio_dir)
    audio_path = Path(audio_path_str)
    with os.fdopen(fd, "wb") as out:
        shutil.copyfileobj(file.file, out)

    # Run the light control-plane stages (just core.load today) so the row
    # exists before the worker picks anything up. Failures here are fatal —
    # if we can't even hash the file, the worker won't do better.
    # emit_terminal=False: the worker container emits analyzer.track_done
    # when the heavy stages finish. If we emit it here too (right after
    # core.load, milliseconds after POST), the Player treats that as final,
    # bails out, and the worker's later beats arrive with no listener.
    config = Config(db_path=db_path, workers=1, user_id=user_id)
    result = analyze_file(audio_path, config, host_path=Path(host_path), emit_terminal=False)
    if not result.ok:
        # Clean up the orphaned audio file on hard failure.
        try:
            audio_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=result.error or "load failed")

    # Short-circuit: if this content already has analyzer-produced cues
    # (loop_propose writes 6), the heavy pipeline already ran on this
    # exact audio (the universal file_hash dedup in upsert_content guarantees
    # same bytes → same content_id). Skip enqueuing — the Player can fetch
    # the existing analysis via GET /v1/tracks/{id}. See ANALYZER_SPEC §4.4.
    #
    # `force=true` (Regenerate Loops) bypasses this — the user is explicitly
    # asking for a fresh analysis pass even though we have one cached.
    conn = connect(db_path)
    try:
        already_analyzed = (not force) and bool(
            conn.execute(
                "SELECT 1 FROM cue WHERE content_id=? AND source='analyzer' LIMIT 1",
                (result.content_id,),
            ).fetchone()
        )
        if already_analyzed:
            log.info("→ analyze short-circuit  content_id=%d  already_analyzed=True  (no job enqueued)",
                     result.content_id)
            # Delete the scratch upload — no worker will read it.
            try:
                audio_path.unlink(missing_ok=True)
            except Exception:
                pass
            # CRITICAL: emit the terminal event the Player's ImportQueue is
            # waiting on. Without this the Player sits forever on its
            # WebSocket subscription. See feedback-import-queue-terminal-wait.
            event_bus.publish("analyzer.track_done", {
                "host_path": host_path,
                "content_id": result.content_id,
            })
            return JSONResponse(
                {
                    "ok": True,
                    "content_id": result.content_id,
                    "job_id": None,
                    "host_path": host_path,
                    "user_id": user_id,
                    "cached": True,
                }
            )

        with transaction(conn):
            job_id = jobs.enqueue(
                conn,
                host_path=host_path,
                user_id=user_id,
                audio_path=str(audio_path),
                content_id=result.content_id,
            )
    finally:
        conn.close()

    return JSONResponse(
        {
            "ok": True,
            "content_id": result.content_id,
            "job_id": job_id,
            "host_path": host_path,
            "user_id": user_id,
            "cached": False,
        }
    )


# ---------------------------------------------------------------------------
# Worker plane (internal) — see DEPLOYMENT_SPEC.md §4
#
# These endpoints are consumed only by GPU worker droplets (or, in local
# dev, the sibling worker container). In production they will be firewalled
# to the worker's IP and require HMAC auth. Slice A: stubs that prove the
# polling loop. Slice B: real queue + signed audio URLs.
# ---------------------------------------------------------------------------

@app.get("/v1/worker/next")
def worker_next():
    """Atomically claim the oldest pending job, or return 204 if queue is empty."""
    db_path = _required_db_path()
    conn = connect(db_path)
    try:
        with transaction(conn):
            job = jobs.claim_next(conn)
        if job is None:
            return Response(status_code=204)
        return JSONResponse(job)
    finally:
        conn.close()


@app.post("/v1/worker/sweep_orphans")
def worker_sweep_orphans() -> JSONResponse:
    """Boot-time crash recovery — called by a freshly-restarted worker.

    Any `analyze_job` rows still in `status='in_progress'` belong to the
    previous worker instance that died (OOM, segfault, whatever). The
    Player is blocked on their `track_done`/`track_failed` event. We:
      1. Find each orphaned job.
      2. Fire a synthetic `analyzer.track_failed` over the event bus so
         the Player's ImportQueue unblocks.
      3. Mark the job `status='failed'` with a reason string.
      4. Delete its scratch audio file from /srv/audio/.

    Returns: {swept: <int>}. See DEPLOYMENT_SPEC §7 Failure Modes + §12
    Memory Model for the broader failure story.
    """
    db_path = _required_db_path()
    conn = connect(db_path)
    swept = 0
    try:
        with transaction(conn):
            rows = list(conn.execute(
                "SELECT job_id, host_path, content_id, audio_path, kind "
                "FROM analyze_job WHERE status='in_progress'"
            ))
            for job_id, host_path, content_id, audio_path, kind in rows:
                # Emit terminal so the Player unblocks.
                event_bus.publish("analyzer.track_failed", {
                    "host_path": host_path,
                    "content_id": content_id,
                    "stage_name": kind or "unknown",
                    "error_message": "worker crashed mid-job; recovered on restart",
                })
                # Mark failed.
                conn.execute(
                    "UPDATE analyze_job SET status='failed' WHERE job_id=?",
                    (job_id,),
                )
                # Delete scratch audio file if it's in /srv/audio.
                try:
                    p = Path(audio_path or "")
                    if p.exists() and p.is_relative_to(Path("/srv/audio")):
                        p.unlink()
                except Exception:
                    pass
                swept += 1
    finally:
        conn.close()

    if swept > 0:
        log.warning("sweep_orphans: failed %d in-progress job(s) from a crashed worker", swept)
    return JSONResponse({"ok": True, "swept": swept})


@app.post("/v1/worker/result")
def worker_result(payload: dict[str, Any]) -> JSONResponse:
    """Worker submits result columns for a job.

    Body shape (DEPLOYMENT_SPEC §4.3):
      {
        "job_id":   <int>,
        "stage":    "structure" | "key" | ...,
        "columns":  { <content column> : <value>, ... },
        "final":    <bool>   # true on the last stage of the job
      }
    """
    job_id = int(payload.get("job_id") or 0)
    if not job_id:
        raise HTTPException(status_code=400, detail="missing job_id")

    columns = payload.get("columns") or {}
    cues = payload.get("cues") or []
    cue_updates = payload.get("cue_updates") or []
    final = bool(payload.get("final"))

    db_path = _required_db_path()
    conn = connect(db_path)
    try:
        with transaction(conn):
            row = conn.execute(
                "SELECT content_id, host_path, audio_path FROM analyze_job WHERE job_id=?",
                (job_id,),
            ).fetchone()
            if not row:
                raise HTTPException(status_code=404, detail=f"job {job_id} not found")
            content_id = int(row[0]) if row[0] is not None else None
            host_path = row[1]
            audio_path_str = row[2]

            if columns and host_path:
                # Resolve the `key` reference-table FK when the worker
                # posts musical_key as text. Keeps the FK-joined SELECT in
                # /v1/tracks/{id} working (k.name AS musical_key) alongside
                # the direct content.musical_key column.
                resolved = dict(columns)
                if "musical_key" in resolved and resolved["musical_key"]:
                    resolved["key_id"] = get_or_create_key(conn, resolved["musical_key"])

                # Reuse the same upsert path the pipeline uses so identity
                # logic (file_hash / fingerprint match) keeps applying.
                content_id = upsert_content(
                    conn,
                    file_path_absolute=host_path,
                    columns=resolved,
                )

            if cues and content_id is not None:
                replace_analyzer_cues(conn, content_id, cues)

            if cue_updates and content_id is not None:
                update_analyzer_cues(conn, content_id, cue_updates)

            if final:
                jobs.mark_done(conn, job_id)

        # Concise stage summary so a single `docker logs dj-loop-service`
        # tells you exactly what the worker just produced.
        stage = payload.get("stage") or "?"
        track_name = Path(host_path).name if host_path else "?"
        cols_summary = ",".join(sorted(columns.keys())) if columns else "-"
        log.info("→ result      %s  stage=%s  cols=[%s]  cues=%d  cue_updates=%d%s",
                 track_name, stage, cols_summary, len(cues), len(cue_updates),
                 "  FINAL" if final else "")

        # NOTE: audio file is intentionally NOT deleted here. The on-demand
        # loop_tag stage (ANALYZER_SPEC §2.2) needs the same audio to embed
        # confirmed loops via MuQ-MuLan, and we don't want the Player to
        # re-upload on every tag request. A proper TTL-based cleanup will
        # be needed before this scales — tracked separately.

        return JSONResponse({"ok": True, "content_id": content_id, "job_id": job_id})
    finally:
        conn.close()


@app.post("/v1/worker/event")
def worker_event(payload: dict[str, Any]) -> JSONResponse:
    """Forward an analyzer.* event from the worker onto the WS event bus
    so connected Players see it as if the control plane had emitted it."""
    event_type = str(payload.get("type") or "")
    if not event_type.startswith("analyzer."):
        raise HTTPException(status_code=400, detail="only analyzer.* events accepted")
    body = {k: v for k, v in payload.items() if k not in ("type", "v", "ts_ms")}

    host_path = body.get("host_path") or ""
    track_name = Path(host_path).name if host_path else "?"
    short_type = event_type.removeprefix("analyzer.")
    stage = body.get("stage_name") or ""
    frac = body.get("fraction_overall")
    err = body.get("error_message") or ""

    if short_type == "track_stage":
        log.info("← %-12s %s  stage=%s  frac=%.0f%%",
                 short_type, track_name, stage, (frac or 0) * 100)
    elif short_type == "track_done":
        log.info("← %-12s %s  cid=%s",
                 short_type, track_name, body.get("content_id"))
    elif short_type == "track_failed":
        log.warning("← %-12s %s  stage=%s  err=%s",
                    short_type, track_name, stage, err)
    else:
        log.info("← %-12s %s", short_type, track_name)

    event_bus.publish(event_type, body)
    return JSONResponse({"ok": True})


# ---------------------------------------------------------------------------
# WebSocket — realtime events
# ---------------------------------------------------------------------------

@app.websocket("/v1/events")
async def events(ws: WebSocket, token: str | None = Query(None)) -> None:
    """Stream analyzer.* (and future playback.*) events to the client.

    Schema: REALTIME_SPEC §4. Each frame is one JSON object with at minimum
    `type`, `v`, `ts_ms`. Clients filter by `type` themselves; the server
    has no subscribe-by-type yet.
    """
    if not check_ws_token(token):
        await ws.close(code=4401)
        return
    await ws.accept()
    q = event_bus.subscribe()
    try:
        # Send a hello so the client can confirm the stream is live.
        await ws.send_json({"type": "events.hello", "v": 1, "ts_ms": int(__import__("time").time() * 1000)})
        async for msg in event_bus.stream(q):
            await ws.send_text(msg)
    except WebSocketDisconnect:
        pass
    finally:
        event_bus.unsubscribe(q)


def _required_db_path() -> Path:
    """library.db path is configured via env var when the server is launched.
    The CLI's `serve` command sets DLP_DB_PATH before invoking uvicorn."""
    raw = os.environ.get("DLP_DB_PATH")
    if not raw:
        raise HTTPException(
            status_code=500,
            detail="DLP_DB_PATH env var not set — Service was not launched correctly.",
        )
    return Path(raw)
