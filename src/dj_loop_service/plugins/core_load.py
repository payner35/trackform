"""core.load — Stage 1 of the analysis pipeline.

Reads audio file shape with soundfile (sample rate + duration), extracts
ID3/tag metadata with mutagen, computes content identity (`file_hash` +
`fingerprint` per ANALYZER_SPEC §4.4), and populates the `content` row plus
reference-table FKs.

This is the only "real" plugin in Phase 1. Other Phase 1 plugins are stubs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import mutagen
import numpy as np
import soundfile

from ..db import (
    connect,
    ensure_schema_present,
    get_or_create_album,
    get_or_create_artist,
    get_or_create_genre,
)
from ..plugin import Ctx, analyzer_stage

_HASH_BLOCK_FRAMES = 65536  # ~1.4 s at 44.1 kHz, per decode read

# fileType integers per ONELIBRARY_SPEC §5.2a
FILE_TYPE_BY_EXT = {
    ".mp3": 1,
    ".m4a": 2,
    ".aac": 2,
    ".wav": 5,
    ".aiff": 6,
    ".aif": 6,
    ".flac": 11,
}


def _read_tags(path: Path) -> dict[str, str | None]:
    """Best-effort tag extraction across formats."""
    try:
        mf = mutagen.File(path, easy=True)
    except Exception:
        return {}
    if mf is None:
        return {}

    def first(key: str) -> str | None:
        val = mf.get(key)
        if not val:
            return None
        return str(val[0]) if isinstance(val, list) else str(val)

    return {
        "title": first("title"),
        "artist": first("artist"),
        "album": first("album"),
        "genre": first("genre"),
        "date": first("date") or first("year"),
    }


@analyzer_stage(name="load", order=10)
def load(ctx: Ctx) -> None:
    path = Path(ctx.file_path)
    if not path.exists():
        raise FileNotFoundError(f"audio file missing: {path}")

    # 1. Tags
    tags = _read_tags(path)
    title = tags.get("title") or path.stem
    artist = tags.get("artist") or "Unknown Artist"
    album = tags.get("album")
    genre = tags.get("genre")

    # 2. Audio file shape — sample rate + duration. soundfile reads the
    #    header without decoding the full waveform.
    duration_seconds: float = 0.0
    sampling_rate: int = 0
    try:
        info = soundfile.info(str(path))
        duration_seconds = float(info.duration)
        sampling_rate = int(info.samplerate)
    except Exception:
        # soundfile (libsndfile) doesn't decode MP3 on all platforms. Fall
        # back to mutagen for duration; sample rate may stay 0.
        try:
            mf = mutagen.File(path)
            if mf is not None and mf.info is not None:
                duration_seconds = float(getattr(mf.info, "length", 0.0))
                sampling_rate = int(getattr(mf.info, "sample_rate", 0))
        except Exception:
            pass

    # 2a. Content identity — SHA-256 of bytes (cheap, strict) and Chromaprint
    #     fingerprint (slower, format-agnostic). See ONELIBRARY_SPEC §8.9.7.
    file_hash = _sha256_file(path)
    fingerprint = _chromaprint_fingerprint(path)

    # 3. Resolve FK ids — needs a short DB hop. Single connection per plugin
    #    call keeps the lifecycle simple. The pipeline's persist stage will
    #    open its own connection for the upsert (separate transaction).
    conn = connect(Path(ctx.db_path))
    try:
        ensure_schema_present(conn)
        artist_id = get_or_create_artist(conn, artist)
        album_id = get_or_create_album(conn, album, artist_id)
        genre_id = get_or_create_genre(conn, genre)
        conn.commit()
    finally:
        conn.close()

    # 4. Persistable columns on content
    ext = path.suffix.lower()
    ctx.set_persistable("title", title)
    ctx.set_persistable("fileName", path.name)
    ctx.set_persistable("filePath", path.name)  # USB-relative-ish; absolute lives in file_path_absolute
    ctx.set_persistable("fileSize", path.stat().st_size)
    ctx.set_persistable("fileType", FILE_TYPE_BY_EXT.get(ext, 0))
    ctx.set_persistable("samplingRate", sampling_rate)
    ctx.set_persistable("duration", int(duration_seconds * 1000))
    ctx.set_persistable("artist_id", artist_id)
    ctx.set_persistable("album_id", album_id)
    ctx.set_persistable("genre_id", genre_id)
    ctx.set_persistable("analysis_source", "native")
    ctx.set_persistable("file_hash", file_hash)
    ctx.set_persistable("user_id", ctx.user_id)
    if fingerprint:
        ctx.set_persistable("fingerprint", fingerprint)

    # In-memory values for downstream stages
    ctx.set("duration_seconds", duration_seconds)
    ctx.set("sampling_rate", sampling_rate)
    ctx.set("file_hash", file_hash)
    ctx.set("fingerprint", fingerprint)


def _sha256_file(path: Path) -> str:
    """SHA-256 of the audio sample stream — tag- and container-agnostic.

    Hashing raw file bytes is fragile: any ID3 / Vorbis-comment / atom edit
    (rating writes, AI tagging, even mutagen re-saves) mutates the file and
    invalidates the hash even though the audio is identical. We decode to
    PCM via libsndfile, mix to mono, quantise to int16, and hash the
    resulting samples. Same audio → same hash, regardless of tag churn.
    """
    h = hashlib.sha256()
    with soundfile.SoundFile(str(path)) as f:
        h.update(f"samplerate={f.samplerate};channels={f.channels};".encode())
        while True:
            block = f.read(_HASH_BLOCK_FRAMES, dtype="float32", always_2d=True)
            if block.size == 0:
                break
            mono = block.mean(axis=1) if block.shape[1] > 1 else block[:, 0]
            pcm16 = np.clip(mono * 32767.0, -32768, 32767).astype(np.int16)
            h.update(pcm16.tobytes())
    return h.hexdigest()


def _chromaprint_fingerprint(path: Path) -> str | None:
    """Compute a Chromaprint acoustic fingerprint. Returns None on failure
    (missing chromaprint binary, unsupported codec, etc.) — the file_hash
    still provides strict-equality identity, so a missing fingerprint
    degrades gracefully."""
    try:
        import acoustid  # provided by pyacoustid

        _duration, fp_bytes = acoustid.fingerprint_file(str(path))
        # fp_bytes is bytes; decode to str for SQLite TEXT column.
        return fp_bytes.decode("ascii") if isinstance(fp_bytes, bytes) else str(fp_bytes)
    except Exception:
        return None
