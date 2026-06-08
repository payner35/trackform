"""SQLite-backed job queue for the GPU-on-demand worker pattern.

POST /v1/analyze enqueues; /v1/worker/next claims atomically;
/v1/worker/result marks done. See DEPLOYMENT_SPEC §4.

This is intentionally tiny — single producer (FastAPI), single consumer
(the worker). Real production may need a Redis backend if we ever scale
to multiple workers, but SQLite is plenty for one GPU droplet at a time.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from .db import now_iso


def enqueue(
    conn: sqlite3.Connection,
    *,
    host_path: str,
    user_id: str,
    audio_path: str,
    content_id: int | None,
    kind: str = "analyze",
    payload: str | None = None,
) -> int:
    cur = conn.execute(
        "INSERT INTO analyze_job (content_id, host_path, user_id, audio_path, "
        "status, created_at, kind, payload) "
        "VALUES (?, ?, ?, ?, 'pending', ?, ?, ?)",
        (content_id, host_path, user_id, audio_path, now_iso(), kind, payload),
    )
    return int(cur.lastrowid)


def claim_next(conn: sqlite3.Connection) -> dict[str, Any] | None:
    """Atomically claim the oldest pending job. Returns None if queue empty."""
    row = conn.execute(
        "SELECT job_id, content_id, host_path, user_id, audio_path, kind, payload "
        "FROM analyze_job WHERE status = 'pending' "
        "ORDER BY created_at ASC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    job_id = int(row[0])
    # UPDATE … WHERE status='pending' guards against a second worker beating us.
    cur = conn.execute(
        "UPDATE analyze_job SET status='in_progress', claimed_at=? "
        "WHERE job_id=? AND status='pending'",
        (now_iso(), job_id),
    )
    if cur.rowcount == 0:
        # Lost the race — nothing to do; caller can retry.
        return None
    return {
        "job_id": job_id,
        "content_id": row[1],
        "host_path": row[2],
        "user_id": row[3],
        "audio_path": row[4],
        "kind": row[5] or "analyze",
        "payload": row[6],
    }


def mark_done(conn: sqlite3.Connection, job_id: int) -> None:
    conn.execute(
        "UPDATE analyze_job SET status='done', finished_at=? WHERE job_id=?",
        (now_iso(), job_id),
    )


def mark_failed(conn: sqlite3.Connection, job_id: int, error: str) -> None:
    conn.execute(
        "UPDATE analyze_job SET status='failed', finished_at=?, error_message=? WHERE job_id=?",
        (now_iso(), error, job_id),
    )
