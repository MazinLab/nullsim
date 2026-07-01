---
date: 2026-06-09
---
### Stage cache: source digest excludes output and CLI code

The stage cache source digest (`_compute_package_source_digest`, folded into
every stage's cache key) now excludes all of `nullsim/outputs/**` and the CLI
entry points (`cli.py`, `__main__.py`) — not just `outputs/plots/**` and
`outputs/tables/**` as before. These modules run after the pipeline has
produced `SimulationState` and never feed back into a stage's result, so
editing a plot, table, the paper bundle writer, or the CLI no longer
invalidates the cached (often multi-hour) det/char unitary optimizations.

The exclusion is centralized in a single `_affects_stage_state` predicate
shared by both the digest and the mtime-memoization scan, so the two cannot
drift (a stricter mtime filter would miss edits and serve a stale digest).
Everything that can run during stage execution — pipeline, config, physics,
stages, catalogs — is still hashed conservatively, so any edit that could
change a stage result still busts the cache.
