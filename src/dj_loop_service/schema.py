"""SQLite schema for the Service's library.db.

This is a self-contained copy of the OneLibrary-aligned schema. The Player's
C++ Database::initSchema() in the player repo is the cross-checked equivalent;
this Python copy is what the Service creates inside its own container.

Player and Service each own their own library.db — they are not the same
file. The Player's DB is for what the user sees on their machine; the
Service's DB is the analysis source of truth. They eventually sync via the
HTTP API (POST /v1/analyze writes Service DB; future GET endpoints feed
the Player). They never share a file.
"""

from __future__ import annotations

SCHEMA_SQL = """
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
  start_anchor_beat_index INTEGER,
  end_anchor_beat_index INTEGER,
  bpm_override INTEGER DEFAULT 0,
  star_rating INTEGER DEFAULT 0,
  play_count INTEGER DEFAULT 0,
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

CREATE INDEX IF NOT EXISTS idx_content_file_path    ON content(file_path_absolute);
CREATE INDEX IF NOT EXISTS idx_content_file_hash    ON content(file_hash);
CREATE INDEX IF NOT EXISTS idx_content_fingerprint  ON content(fingerprint);
CREATE INDEX IF NOT EXISTS idx_content_user         ON content(user_id);

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
  embedding BLOB,
  embedding_consistency REAL,
  mood_tags TEXT,
  genre_hints TEXT,
  embedded_at_position TEXT,
  source TEXT DEFAULT 'native',
  created_at TEXT,
  updated_at TEXT,
  FOREIGN KEY (content_id) REFERENCES content(content_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_cue_content ON cue(content_id);

CREATE TABLE IF NOT EXISTS playlist (
  playlist_id INTEGER PRIMARY KEY AUTOINCREMENT,
  is_folder INTEGER DEFAULT 0,
  name TEXT NOT NULL,
  parent_id INTEGER,
  colour TEXT,
  source TEXT DEFAULT 'user',
  user_id TEXT DEFAULT 'local',
  created_at TEXT,
  FOREIGN KEY (parent_id) REFERENCES playlist(playlist_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_playlist_user ON playlist(user_id);

CREATE TABLE IF NOT EXISTS playlistEntry (
  entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
  sort_index INTEGER,
  playlist_id INTEGER NOT NULL,
  track_id INTEGER NOT NULL,
  added_at TEXT,
  FOREIGN KEY (playlist_id) REFERENCES playlist(playlist_id) ON DELETE CASCADE,
  FOREIGN KEY (track_id) REFERENCES content(content_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS play_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  content_id INTEGER NOT NULL,
  cue_id INTEGER,
  played_at TEXT NOT NULL,
  duration_seconds REAL,
  user_id TEXT DEFAULT 'local',
  FOREIGN KEY (content_id) REFERENCES content(content_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_play_history_content ON play_history(content_id);
CREATE INDEX IF NOT EXISTS idx_play_history_user    ON play_history(user_id);

CREATE TABLE IF NOT EXISTS settings (
  key TEXT PRIMARY KEY,
  value TEXT,
  updated_at TEXT
);

-- Job queue for the GPU-on-demand worker pattern (DEPLOYMENT_SPEC §4).
-- POST /v1/analyze enqueues; /v1/worker/next atomically claims; /v1/worker/result
-- marks done. Server-local only — not part of the OneLibrary schema, not synced.
CREATE TABLE IF NOT EXISTS analyze_job (
  job_id        INTEGER PRIMARY KEY AUTOINCREMENT,
  content_id    INTEGER,                       -- set after core.load runs
  host_path     TEXT NOT NULL,                 -- Player's host-local path
  user_id       TEXT NOT NULL DEFAULT 'local',
  audio_path    TEXT NOT NULL,                 -- path on shared volume the worker reads from
  status        TEXT NOT NULL DEFAULT 'pending', -- pending | in_progress | done | failed
  error_message TEXT,
  created_at    TEXT NOT NULL,
  claimed_at    TEXT,
  finished_at   TEXT,
  kind          TEXT NOT NULL DEFAULT 'analyze', -- 'analyze' | 'tag'
  payload       TEXT                            -- job-specific JSON (cue_ids for tag jobs)
);
CREATE INDEX IF NOT EXISTS idx_analyze_job_status ON analyze_job(status, created_at);
"""
