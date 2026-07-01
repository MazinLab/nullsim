---
date: 2026-05-17
---

### ChipOptimizationParams throughput-penalty knobs (commits dd16ee05, 6f485003)

- `ChipOptimizationParams` gains `throughput_penalty_alpha` (float, default 0.0),
  `throughput_penalty_seps_mas` (tuple, default `(3.0,)`), and
  `throughput_penalty_n_pa` (int, default 8). Port of KILO_BB_THROUGHPUT_PENALTY_ALPHA
  convention (`kilo/studies/natural_design.py` L590-625). When alpha > 0 the bootstrap
  objective adds a soft penalty `−alpha × <mean planet throughput at rep seps>` to
  favor basins that preserve the kernel-null's throughput rolloff. Without the penalty
  the optimizer finds ultra-deep-null basins (chip_ideal ~4e-6) that collapse planet
  throughput at small separations (~0.006 at 30 mas vs KILO's 0.35).
- `kilo_keck.toml` sets `throughput_penalty_alpha=5e-2`,
  `throughput_penalty_seps_mas=[3.0, 10.0, 30.0]`.

### DetectionCurve ao_piston_scale and scintillation_index_per_sub (commits 1764ae0, 7063adf, 1407e01)

- `DetectionCurve.Params` gains `ao_piston_scale` (float, default 1.0). Multiplies the
  per-sub-aperture piston RMS returned by `_find_ao_piston_rms_per_sub_m` before
  passing it to `realized_null_depth_mc`. The raw Conan 1995 formula can overestimate
  KILO's tuned-servo piston by ~2-3x at Keck J=4; set to 0.4 to match KILO's regime.
- `DetectionCurve.Params` gains `scintillation_index_per_sub` (float, default 0.0).
  Passed to `realized_null_depth_mc` as per-sub multiplicative amplitude jitter drawn
  log-normal with E[I]=1, Var[I]=sigma_I^2 (Roddier 1981, KILO transport L1639). This
  is the post-amplitude-servo residual, not open-loop scintillation.
- `realized_null_depth_mc` gains a matching `scintillation_index_per_sub` parameter.
- `kilo_keck.toml` sets `ao_piston_scale=0.4`, `scintillation_index_per_sub=0.02`.

### _find_ao_piston_rms_per_sub_m priority correction

- `_find_ao_piston_rms_per_sub_m` now prefers the raw AO-piston residual
  (`state.wavefront.components['ao_residual'].piston_opd_nm.total`), matching KILO's
  `transport.realized_null_depth` convention. The fringe tracker corrects only the
  common inter-telescope piston; the within-telescope differential piston that drives
  the chip null floor comes from the AO stage. Falls back to the FT closed-loop residual
  for space pipelines with no AO stage.
