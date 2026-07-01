---
date: 2026-06-09
---
### Stage cache: config_hash-independent keys (Tier 3)

A stage's cache key no longer folds in the global `config_hash` unless the
stage actually draws from the `config_hash`-seeded `context.rng`
(`uses_rng=True` — only `transport.smf28`/`pm980`/`pm780` and
`background.space_zodi`). Every other stage's result is `config_hash`-
independent, so its key now depends only on its own params, the content digest
of its consumed + `cache_key_extras` budgets, the run-global wavelength grid,
the per-family source digest, and instance identity. Editing one part of a
config (e.g. a `sensitivity` piston param) now **reuses** the cached results of
the expensive `chip_optimization`/`characterization` stages instead of busting
the whole pipeline — turning a ~34 min config re-run into minutes.

`config_hash` was a global catch-all, so removing it required closing the
dependencies it was masking (audited across every stage):

- **The wavelength grid** (`[grid]`) is a run-global input in no stage's
  Params; it is now folded into every stage's cache key directly, so a grid
  edit still invalidates all wavelength-indexed results.
- **`cache_key_extras`** added for stages that read undeclared upstream state:
  `injection.ideal` (`throughput`), `wavefront.space_residual` (`wavefront`,
  `opd_budget`), `telescope.fixed_aperture` (`throughput`),
  `fringe_tracking.closed_loop` (`scene`, `field`).

The per-stage RNG derivation (`derive_rng`) is **unchanged**, so no stochastic
result is re-seeded — the contrast and characterization curves are unaffected.
Verified end-to-end that a cached run after an upstream config edit matches a
fresh no-cache run (no stale results).
