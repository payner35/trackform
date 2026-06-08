"""Worker-side analyzer plugins.

Each module registers one stage via `@analyzer_stage`. The loader in
`worker/plugin.py` decides which plugins are active by importing them.
"""
