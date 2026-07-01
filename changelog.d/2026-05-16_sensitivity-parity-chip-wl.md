---
date: 2026-05-16
---

### Sensitivity parity fixes (commits 118c5dc, 1fda201)

- `sensitivity.detection_curve` now reads the post-fringe-tracker OPD residual
  (`state.opd_budget.components['fringe_tracking.closed_loop_residual']`) instead
  of the raw AO-piston residual; falls back to the AO residual when no FT stage
  has run. For kilo_keck this corrects σ_OPD from 10.7 nm to ~1.35 nm, dropping
  the realized null from 0.20 to ~4e-4. `DetectionCurve.consumes` now includes
  `opd_budget`.
- `_split_apertures_to_modes` uses the pupil-packed d_sub (via
  `optimize_subaperture_positions`) when a `pupil_template` is available from the
  upstream pyramid_wfs result, rather than the naive area-equivalent diameter.
  For Keck N=12 this corrects d_sub from 2.89 m to 1.93 m, removing a ~1000×
  over-attenuation of the planet field at 300 mas. After both fixes the J=4
  contrast at 300 mas drops ~10000× and the detection-space plot tracks KILO's
  shape across the full separation range.

### Chip-optimizer wavelength subsampling (commit 910f2de)

- `ChipOptimizationParams` gains `n_wl_optimize` (int, default 0 = full grid).
  When set, the torch backend subsamples the wavelength axis to that many
  uniformly-spaced indices for the optimization only; the downstream
  `chip.kernel_mzi_mesh` applies the optimized unitary at every bin of the
  original grid. Matches KILO's `n_wl_broadband=5` strategy and reduces
  per-iteration optimizer cost ~4× on a 20-bin grid.
- `examples/kilo_keck.toml` sets `n_wl_optimize=5`, `max_iter=15000`,
  `n_restarts=10`, `restart_patience=4`.
