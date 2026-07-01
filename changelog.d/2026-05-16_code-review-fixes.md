---
date: 2026-05-16
---
### Top-7 code-review fixes applied (commit b6f36c4)

Addressed the highest-priority findings from the first code-review pass:

- `nullsim/cli.py`: improved error reporting and CLI robustness.
- `nullsim/config/loader.py`, `sweeps.py`: edge-case handling in config loading and sweep expansion.
- `nullsim/pipeline/_canonical.py`: added (new module — type-tagged canonical-bytes hashing used by state, rng, and cache).
- `nullsim/pipeline/cache.py`, `rng.py`, `runner.py`, `stage.py`, `state.py`: correctness fixes for cache key derivation, RNG stream isolation, runner validation, and state mutation guards.
- `nullsim/stages/` (all five trivial stages): fixed contract conformance issues.
- Tests updated to cover the fixed paths.
