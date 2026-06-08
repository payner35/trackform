"""Create a fresh library.db with the OneLibrary-aligned schema for dev testing.

This is a DEV CONVENIENCE ONLY. In normal operation the Player creates and
migrates library.db on first launch (see Player's src/storage/Database.cpp).
The schema here is a verbatim copy of Database::initSchema() at the time of
the service's last sync. If the Player's schema changes, update this file
or — better — just launch the Player against the same DB path.

Usage:
    uv run python scripts/init_test_db.py /tmp/test_library.db
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS artist (
  artist_id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE,
  nameForSearch TEXT
);

CREATE TABLE IF NOT EXISTS album (
  album_id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE,
  nameForSearch TEXT,
  artist_id INTEGER,
  FOREIGN KEY (artist_id) REFERENCES artist(artist_id)
);

CREATE TABLE IF NOT EXISTS genre (
  genre_id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE,
  nameForSearch TEXT
);

CREATE TABLE IF NOT EXISTS key (
  key_id INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS color (
  color_id INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS content (
  content_id INTEGER PRIMARY KEY AUTOINCREMENT,
  title TEXT,
  fileName TEXT,
  filePath TEXT,
  fileSize INTEGER,
  fileType INTEGER,
  bitrate INTEGER,
  bitDepth INTEGER,
  samplingRate INTEGER,
  duration INTEGER,
  tempo INTEGER,
  rating INTEGER DEFAULT 0,
  artist_id INTEGER,
  album_id INTEGER,
  genre_id INTEGER,
  key_id INTEGER,
  color_id INTEGER,
  file_path_absolute TEXT,
  musical_key TEXT,
  camelot_key TEXT,
  key_confidence REAL,
  analysis_source TEXT DEFAULT 'native',
  analysis_version TEXT,
  beat_grid_csv TEXT,
  bars_csv TEXT,
  bpm_override INTEGER DEFAULT 0,
  star_rating INTEGER DEFAULT 0,
  play_count INTEGER DEFAULT 0,
  -- Extensions added by the Service per ANALYZER_SPEC §4.3 / ONELIBRARY_SPEC §8.9.7, §8.9.9
  file_hash TEXT,
  fingerprint TEXT,
  user_id TEXT DEFAULT 'local',
  created_at TEXT,
  updated_at TEXT,
  FOREIGN KEY (artist_id) REFERENCES artist(artist_id),
  FOREIGN KEY (album_id) REFERENCES album(album_id),
  FOREIGN KEY (genre_id) REFERENCES genre(genre_id),
  FOREIGN KEY (key_id) REFERENCES key(key_id),
  FOREIGN KEY (color_id) REFERENCES color(color_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_content_file_path ON content(file_path_absolute);
CREATE INDEX IF NOT EXISTS idx_content_file_hash ON content(file_hash);
CREATE INDEX IF NOT EXISTS idx_content_fingerprint ON content(fingerprint);
CREATE INDEX IF NOT EXISTS idx_content_user ON content(user_id);

CREATE TABLE IF NOT EXISTS cue (
  cue_id INTEGER PRIMARY KEY AUTOINCREMENT,
  content_id INTEGER NOT NULL,
  kind INTEGER NOT NULL DEFAULT 4,
  colorTableIndex INTEGER,
  activeLoop INTEGER DEFAULT 1,
  attribute INTEGER DEFAULT 0,
  pointNumerator INTEGER,
  pointDenominator INTEGER DEFAULT 1000,
  loopNumerator INTEGER,
  loopDenominator INTEGER DEFAULT 1000,
  label TEXT,
  loop_type TEXT,
  start_bar INTEGER,
  bars INTEGER,
  bpm REAL,
  musical_key TEXT,
  camelot_key TEXT,
  key_confidence REAL,
  energy_value REAL,
  energy_label TEXT,
  energy_movement TEXT,
  energy_confidence REAL,
  vocal_density REAL,
  percussion_density REAL,
  bass_presence REAL,
  melodic_presence REAL,
  overall_score REAL,
  beat_alignment_score REAL,
  phrase_alignment_score REAL,
  stability_score REAL,
  clean_start_score REAL,
  clean_end_score REAL,
  transition_score REAL,
  source TEXT DEFAULT 'native',
  created_at TEXT,
  updated_at TEXT,
  FOREIGN KEY (content_id) REFERENCES content(content_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_cue_content ON cue(content_id);
"""


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    db_path = Path(argv[1])
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        print(f"refusing to overwrite existing file: {db_path}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()
    print(f"created {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
