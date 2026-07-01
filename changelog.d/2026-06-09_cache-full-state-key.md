---
date: 2026-06-09
---
### Stage cache: key digests the full input state (sweep-cell state resurrection fix)

The Tier 3 config_hash-independent cache key digested only the stage's
consumed budgets while a cache hit restores the entire pickled post-stage
state. In a parallel sweep, a stage that does not consume `scene` (e.g.
`atmosphere.kolmogorov`) had identical keys across cells, so one cell's full
state — `scene.star_jmag` included — was resurrected into every other cell,
poisoning all downstream photon budgets and silently skipping the
characterization stages (every cell believed it was J=6). `key_for` now
digests the full input state; the redundant `cache_key_extras`/grid folds
were removed; regression tests reproduce the sweep poisoning end-to-end.
