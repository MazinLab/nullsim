---
date: 2026-05-17
---
### Phase 5 — cache RNG identity: raise on None components

Per the KILO parity audit (2026-05-17), `_rng_identity_bytes` in
`nullsim/pipeline/cache.py` f-stringed `None` directly into the cache
key when `root_seed`, `config_hash`, or `sweep_coord_tuple` was missing.
Two runs whose identity components both happened to be `None` would
collide despite carrying different actual seeds — a direct violation of
the SPEC §6.5 cache invariant ("a stochastic stage's second run with a
different seed must NOT cache-hit the first run's entry").

- `_rng_identity_bytes` now raises `ValueError` when any of
  `root_seed`, `config_hash`, or `sweep_coord_tuple` is `None`. The
  single legitimate "no context" path (low-level unit tests that
  instantiate a stage outside the runner) still returns the
  `<no-context>` sentinel.
- Empty `sweep_coord_tuple = ()` continues to mean "single-cell run"
  and is accepted; only `None` is rejected.
- Two regression tests pin the new behaviour: missing `root_seed` and
  missing `config_hash` both raise `ValueError` with a clear,
  instance-tagged message.

The pre-Phase-5 defensive `or ()` default that masked the `None` case
for `sweep_coord_tuple` is removed.
