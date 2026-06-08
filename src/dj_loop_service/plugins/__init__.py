"""Built-in analyzer plugins.

Each module here registers stages via `@analyzer_stage`. Plugins are imported
explicitly by `pipeline.load_builtin_plugins()` so registration order is
deterministic and so that an unused plugin doesn't get loaded.
"""
