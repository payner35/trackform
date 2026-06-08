"""Smoke test: built-in plugins import and register stages in the expected order."""

from __future__ import annotations

from dj_loop_service.pipeline import load_builtin_plugins
from dj_loop_service.plugin import registered_stages


def test_builtin_plugins_register():
    load_builtin_plugins()
    stages = registered_stages()
    names = [s.name for s in stages]

    assert "load" in names
    assert "beats" in names
    assert "key" in names
    assert "sections" in names
    assert "loop_mining" in names
    assert "loop_features_basic" in names

    # Order matters — load must run before beats.
    assert names.index("load") < names.index("beats") < names.index("key")
