"""SQLite read/write helpers.

The Service owns its own library.db. It creates the schema on first run
(see `init_schema`). The Player has its own library.db on the host — the
two are eventually synced via HTTP, never shared as a file.

Schema is OneLibrary-aligned per the Player's `docs/lib/ONELIBRARY_SPEC.md`
§8 (a verbatim copy lives in `schema.py`).
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .schema import SCHEMA_SQL


def connect(db_path: Path) -> sqlite3.Connection:
    """Open (or create) the Service's library.db."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Create OneLibrary-aligned tables if they don't exist. Idempotent."""
    conn.executescript(SCHEMA_SQL)
    _apply_migrations(conn)
    conn.commit()


def _apply_migrations(conn: sqlite3.Connection) -> None:
    """Idempotently add columns that were introduced after the initial schema.

    SQLite has no `ALTER TABLE ADD COLUMN IF NOT EXISTS`, so we check
    `PRAGMA table_info` before each ALTER. Order matters only if a new
    column depends on another new column (none do today).
    """
    existing = {row[1] for row in conn.execute("PRAGMA table_info(content)")}
    if "start_anchor_beat_index" not in existing:
        conn.execute("ALTER TABLE content ADD COLUMN start_anchor_beat_index INTEGER")
    if "end_anchor_beat_index" not in existing:
        conn.execute("ALTER TABLE content ADD COLUMN end_anchor_beat_index INTEGER")
    if "key_confidence" not in existing:
        conn.execute("ALTER TABLE content ADD COLUMN key_confidence REAL")

    cue_existing = {row[1] for row in conn.execute("PRAGMA table_info(cue)")}
    if "embedding" not in cue_existing:
        conn.execute("ALTER TABLE cue ADD COLUMN embedding BLOB")
    if "embedding_consistency" not in cue_existing:
        conn.execute("ALTER TABLE cue ADD COLUMN embedding_consistency REAL")
    if "mood_tags" not in cue_existing:
        conn.execute("ALTER TABLE cue ADD COLUMN mood_tags TEXT")
    if "genre_hints" not in cue_existing:
        conn.execute("ALTER TABLE cue ADD COLUMN genre_hints TEXT")
    # Dirty-check signature for the on-demand loop_tag stage. Stores
    # `pointNumerator + '_' + loopNumerator` of the cue at the moment
    # MuQ-MuLan last embedded it. Compared against the current cue's
    # signature on every tag request; mismatch → re-embed.
    if "embedded_at_position" not in cue_existing:
        conn.execute("ALTER TABLE cue ADD COLUMN embedded_at_position TEXT")

    # analyze_job extensions for the picker+tagger split (ANALYZER_SPEC §2.2):
    # `kind` distinguishes 'analyze' jobs (default pipeline) from 'tag' jobs
    # (loop_tag on a specific cue list). `payload` carries job-specific data
    # like the cue_ids list for tag jobs.
    job_existing = {row[1] for row in conn.execute("PRAGMA table_info(analyze_job)")}
    if "kind" not in job_existing:
        conn.execute("ALTER TABLE analyze_job ADD COLUMN kind TEXT DEFAULT 'analyze'")
    if "payload" not in job_existing:
        conn.execute("ALTER TABLE analyze_job ADD COLUMN payload TEXT")


# Back-compat shim so existing code that called ensure_schema_present()
# still works. The Service now creates the schema itself rather than
# failing if it's missing.
def ensure_schema_present(conn: sqlite3.Connection) -> None:
    init_schema(conn)


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_or_create_artist(conn: sqlite3.Connection, name: str | None) -> int | None:
    if not name:
        return None
    # Concurrent workers may race here; INSERT OR IGNORE keeps the unique
    # constraint without raising, then SELECT gives us the canonical id
    # regardless of which worker won the race.
    conn.execute(
        "INSERT OR IGNORE INTO artist (name, nameForSearch) VALUES (?, ?)",
        (name, name.lower()),
    )
    row = conn.execute("SELECT artist_id FROM artist WHERE name = ?", (name,)).fetchone()
    return int(row[0]) if row else None


def get_or_create_album(
    conn: sqlite3.Connection, name: str | None, artist_id: int | None
) -> int | None:
    if not name:
        return None
    conn.execute(
        "INSERT OR IGNORE INTO album (name, nameForSearch, artist_id) VALUES (?, ?, ?)",
        (name, name.lower(), artist_id),
    )
    row = conn.execute("SELECT album_id FROM album WHERE name = ?", (name,)).fetchone()
    return int(row[0]) if row else None


def get_or_create_key(conn: sqlite3.Connection, name: str | None) -> int | None:
    """Resolve a musical key string (e.g. 'A minor') to a row in the `key`
    reference table, creating it if necessary. The Service stores key as
    both a direct content.musical_key TEXT column AND a content.key_id FK
    (for OneLibrary compatibility) — this helper handles the FK side."""
    if not name:
        return None
    conn.execute("INSERT OR IGNORE INTO key (name) VALUES (?)", (name,))
    row = conn.execute("SELECT key_id FROM key WHERE name = ?", (name,)).fetchone()
    return int(row[0]) if row else None


def get_or_create_genre(conn: sqlite3.Connection, name: str | None) -> int | None:
    if not name:
        return None
    conn.execute(
        "INSERT OR IGNORE INTO genre (name, nameForSearch) VALUES (?, ?)",
        (name, name.lower()),
    )
    row = conn.execute("SELECT genre_id FROM genre WHERE name = ?", (name,)).fetchone()
    return int(row[0]) if row else None


def upsert_content(
    conn: sqlite3.Connection,
    file_path_absolute: str,
    columns: dict[str, Any],
) -> int:
    """Insert or update a content row.

    Lookup precedence (ANALYZER_SPEC §4.4, ONELIBRARY_SPEC §8.9.7):
      1. file_hash   — strict-equality identity
      2. fingerprint — format-tolerant identity
      3. file_path_absolute — last-resort, host-local path match

    The first two are the durable identity. file_path_absolute is a cache
    that the Player owns — when matching by hash or fingerprint, the
    Service does NOT overwrite file_path_absolute. Reason: the Service may
    be running inside a container where the path differs from the Player's
    view of the host filesystem.

    Returns the content_id (existing or newly assigned).
    """
    columns = dict(columns)
    columns["updated_at"] = now_iso()

    # 1. file_hash match — preserve existing file_path_absolute.
    file_hash = columns.get("file_hash")
    if file_hash:
        row = conn.execute(
            "SELECT content_id FROM content WHERE file_hash = ?", (file_hash,)
        ).fetchone()
        if row:
            return _update_preserving_path(conn, int(row[0]), columns)

    # 2. fingerprint match — preserve existing file_path_absolute.
    fingerprint = columns.get("fingerprint")
    if fingerprint:
        row = conn.execute(
            "SELECT content_id FROM content WHERE fingerprint = ?", (fingerprint,)
        ).fetchone()
        if row:
            return _update_preserving_path(conn, int(row[0]), columns)

    # 3. file_path_absolute match — exact-path fallback for content the
    #    Analyzer has not previously fingerprinted. Safe to set the path
    #    here since we're matching on it.
    row = conn.execute(
        "SELECT content_id FROM content WHERE file_path_absolute = ?",
        (file_path_absolute,),
    ).fetchone()
    columns["file_path_absolute"] = file_path_absolute
    if row:
        return _update(conn, int(row[0]), columns)

    # 4. Brand-new track — INSERT with the path the Analyzer saw. The Player
    #    will rewrite this if needed via the self-heal scan
    #    (LIBRARY_SPEC §4.6) once it sees the row.
    columns["created_at"] = columns["updated_at"]
    cols = ", ".join(columns.keys())
    placeholders = ", ".join("?" * len(columns))
    cur = conn.execute(
        f"INSERT INTO content ({cols}) VALUES ({placeholders})",
        tuple(columns.values()),
    )
    return int(cur.lastrowid)


def _update_preserving_path(
    conn: sqlite3.Connection, content_id: int, columns: dict[str, Any]
) -> int:
    """Update an existing row but never overwrite file_path_absolute."""
    cols = {k: v for k, v in columns.items() if k != "file_path_absolute"}
    return _update(conn, content_id, cols)


def _update(conn: sqlite3.Connection, content_id: int, columns: dict[str, Any]) -> int:
    if not columns:
        return content_id
    sets = ", ".join(f"{k} = ?" for k in columns)
    conn.execute(
        f"UPDATE content SET {sets} WHERE content_id = ?",
        (*columns.values(), content_id),
    )
    return content_id


# Columns the analyzer is allowed to write into cue rows. Anything outside
# this set is silently dropped — prevents a misbehaving plugin from
# writing junk fields, and gives us one place to update when the cue
# schema grows.
_ANALYZER_CUE_COLUMNS = {
    "kind", "label", "loop_type",
    "pointNumerator", "pointDenominator",
    "loopNumerator", "loopDenominator",
    "start_bar", "bars", "bpm",
    "musical_key", "camelot_key", "key_confidence",
    "energy_value", "energy_label", "energy_movement", "energy_confidence",
    "vocal_density", "percussion_density", "bass_presence", "melodic_presence",
    "overall_score", "beat_alignment_score", "phrase_alignment_score",
    "stability_score", "clean_start_score", "clean_end_score", "transition_score",
    "embedding", "embedding_consistency", "mood_tags", "genre_hints",
    "embedded_at_position",
}


def update_analyzer_cues(
    conn: sqlite3.Connection, content_id: int, updates: list[dict[str, Any]]
) -> int:
    """Apply per-cue column updates produced by the on-demand loop_tag stage.

    Each update is {cue_id, columns: {<col>: <value>, ...}}. Columns are
    filtered against _ANALYZER_CUE_COLUMNS. The `embedding` column is
    base64-decoded back to raw bytes (same convention as
    replace_analyzer_cues). Updates with empty `columns` are skipped
    (those are the dirty-check no-ops from loop_tag).
    """
    import base64

    if not updates:
        return 0

    ts = now_iso()
    applied = 0
    for u in updates:
        cue_id = int(u.get("cue_id") or 0)
        cols = u.get("columns") or {}
        if not cue_id or not cols:
            continue
        row = {k: v for k, v in cols.items() if k in _ANALYZER_CUE_COLUMNS}
        if not row:
            continue
        emb = row.get("embedding")
        if isinstance(emb, str) and emb:
            try:
                row["embedding"] = base64.b64decode(emb)
            except (ValueError, TypeError):
                row["embedding"] = None
        row["updated_at"] = ts
        sets = ", ".join(f"{k} = ?" for k in row)
        conn.execute(
            f"UPDATE cue SET {sets} WHERE cue_id = ? AND content_id = ?",
            (*row.values(), cue_id, content_id),
        )
        applied += 1
    return applied


def replace_analyzer_cues(
    conn: sqlite3.Connection, content_id: int, cues: list[dict[str, Any]]
) -> int:
    """Replace all analyzer-produced cues for `content_id` with `cues`.

    Atomic on the cue set: deletes existing rows with `source='analyzer'`,
    then inserts the new batch. User-created cues (`source='native'` or
    anything else) are untouched. Returns the number of cues inserted.
    """
    if not cues:
        # No-op — but still wipe any stale analyzer cues so a re-run can
        # shrink the set. Callers pass an empty list explicitly to mean
        # "clear them"; loop_mining always passes a non-empty list.
        conn.execute(
            "DELETE FROM cue WHERE content_id = ? AND source = 'analyzer'",
            (content_id,),
        )
        return 0

    conn.execute(
        "DELETE FROM cue WHERE content_id = ? AND source = 'analyzer'",
        (content_id,),
    )

    import base64

    ts = now_iso()
    inserted = 0
    for cue in cues:
        row = {k: v for k, v in cue.items() if k in _ANALYZER_CUE_COLUMNS}
        if not row:
            continue
        # Embeddings travel as base64 strings over JSON; decode to raw bytes
        # for the BLOB column. Skip on malformed payloads rather than
        # poisoning the row — the rest of the cue (mood/genre/score) is
        # still useful even without the similarity vector.
        emb = row.get("embedding")
        if isinstance(emb, str) and emb:
            try:
                row["embedding"] = base64.b64decode(emb)
            except (ValueError, TypeError):
                row["embedding"] = None
        row.setdefault("kind", 4)  # loop
        row["content_id"] = content_id
        row["source"] = "analyzer"
        row["created_at"] = ts
        row["updated_at"] = ts
        cols = ", ".join(row.keys())
        placeholders = ", ".join("?" for _ in row)
        conn.execute(
            f"INSERT INTO cue ({cols}) VALUES ({placeholders})",
            tuple(row.values()),
        )
        inserted += 1
    return inserted
