---
date: 2026-06-09
---
### Stage cache: per-family source digest (Tier 2 granularity)

The stage cache source digest is now computed per stage family instead of over
the whole package. A stage's cache key folds in the shared core (pipeline,
config, physics, catalogs, the shared `stages/_*.py` helpers) plus the
transitive import closure of the stage's own family over other stage families.
Output and CLI code remain excluded (Tier 1).

Effect: editing one stage family's code no longer invalidates the cached
(often multi-hour) optimizations of unrelated families. The det path
(`chip_optimization`, `sensitivity`) imports no part of the char path, so
editing `characterization` reuses the det cache and editing `sensitivity`
reuses the char cache. Editing genuinely shared code (e.g. `chip_optimization`
— which `characterization` and `chip` import — or anything under `physics/`)
still correctly busts every dependent family.

The family-import closure is derived from the AST of each family's modules
(real import statements only, never docstring mentions), so it stays correct as
dependencies change with no hand-maintained dependency map — and it already
caught a real `chip_optimization` → `delay_lines.geometric` edge that manual
greps missed. Any ambiguity (unparseable file, unresolved relative import,
unknown stage family) falls back to including every family, so the digest can
only over-invalidate, never serve a stale result. The digest and the
mtime-memoization scan share the `_affects_stage_state` filter so they cannot
drift.
