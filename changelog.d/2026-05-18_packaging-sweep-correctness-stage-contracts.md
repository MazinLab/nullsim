---
date: 2026-05-18
---
### Packaging, sweep correctness, stage contracts, and numerical fixes (phases 1–9)

**Packaging (phase 1):** wheel now bundles `data/**/*.csv` and `config/presets/*.toml`
so pip-installed packages find catalog snapshots and preset configs at runtime.
`[mlx]` extra replaced by `[dev]` (`pytest>=8`).

**Chip-param wrap (phase 2):** `chip.kernel_mzi_mesh._quantize_phases` wraps to
`[0, 2π)` even when `dac_bits is None`, matching the scipy optimizer's convention
and preventing deployment of a different chromatic chip than the optimizer evaluated.

**Sweep correctness (phase 3):** `materialize_pipeline_stages` gains `skip_stage_params`
so sweep-written params are not clobbered by re-materialization. `_run_hash_with_externals`
folds all per-cell externals into run identity (not just cell-0).

**FT servo read-through (phase 5):** `performance.standard._resolve_servo_bw` reads
live `servo_bandwidth_hz` from fringe-tracking results before falling back to the
default. Double-`eta` bias in `fringe_tracking.closed_loop._resolve_servo_bandwidth_hz`
removed.

**Stage contracts (phase 6):** `performance.standard.consumes` removes `photon_rates`
(rates arrive via `state.results`). `sensitivity.detection_curve.consumes` adds
`throughput`. `injection.ideal` raises on ambiguous Strehl fallback.

**Numerical fixes (phase 7):** Pooled median/std in `realized_null`; monotonic
separation validation in `stellar_disk`; deterministic-zero phase and seed-raise in
`transport.smf28`; mismatch guards in `sensitivity.detection_curve`.

**Catalog provenance + CLI robustness (phase 8):** `detection_space` loads the
user-configured snapshot path for the planet overlay. Empty-sweep guard raises
`NullsimConfigError`. `RunCollection` uses materialized cell-0 config.

**Lazy outputs + cache digest (phase 9):** `nullsim validate` no longer imports
matplotlib. Cache digest invalidates on source file mtime changes.

**detection_space legend and x-bound:** legend lower-left; x-axis scaled to curve data.
