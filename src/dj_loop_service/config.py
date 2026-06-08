"""Config loader (minimal).

Phase 1 has no on-disk config file — values come from CLI flags. This module
exists so plugins can `from .config import Config` and the surface won't
change when we add TOML config loading later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    db_path: Path
    workers: int = 1
    force_reanalyze: bool = False
    # Owner of rows written this run. Multi-tenancy at the data layer
    # (ONELIBRARY_SPEC §8.9.9). Default "local" for single-user / dev.
    user_id: str = "local"
