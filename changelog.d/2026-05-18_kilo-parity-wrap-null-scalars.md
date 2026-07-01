---
date: 2026-05-18
---
### KILO parity pass: torch-path wrap fix, KILO-convention null scalars, fringe-tracking default

Four commits closing the remaining KILO parity gap and resolving a null-depth
reporting convention mismatch.

**chip_optimization: phase-wrap location moved (d31d64b)**

`_quantize_phases` now only rounds to the DAC grid. The `[0, 2π)` wrap is
applied inside `_build_unitary_stack_numpy` at the same call site as the
per-wavelength rescaling. The torch/mlx backends optimize in unwrapped phase
space; wrapping the iterate post-hoc shifted the deployed chip away from the
autograd operating point, causing `kilo_keck` `loss_final` to balloon from
~1e-3 to ~3.3 and the realized chip null to degrade from ~3e-4 to ~1e-2.

**chip_optimization._require_field strips upstream phases (167d5e3)**

`_require_field` collapses `state.field.amplitudes` to the wavelength-mean
per-mode magnitude and broadcasts as a real-valued wavelength-independent
field, matching KILO `injection.py:stellar_input_field`. Pre-fix, upstream
AO-residual and transport phases drove the broadband optimizer into a shallow
basin (mean dark/bright ~1e-3). Post-fix: ~7.5e-5, within 1.08× of KILO's
1.4e-4 reference at J=4 Keck.

**KILO-convention null scalars added (d31d64b + 3fb003b)**

`RealizedNullResult` gains two new fields:
- `ideal_null_depth_kilo`: chip-only ideal mean(dark)/mean(bright).
- `mean_null_depth_kilo`: AO-realized MC mean, same convention.

Both are surfaced in the sensitivity payload (`ideal_chip_null_depth_kilo`,
`realized_null_depth_kilo`) and in sweep parquet. The existing dark-power-
fraction fields stay as the downstream contrast pipeline's input. Measured on
kilo_keck J=4: `realized_null_depth_kilo` = 1.84e-4 vs KILO 1.4e-4 (1.3×).

**fringe_tracking default predictive_lag_reduction changed to 0.05 (c137256)**

`ClosedLoopFringeTracker.Params.predictive_lag_reduction` default changed
from 1.0 to 0.05, matching KILO production convention
(`KILO_PISTON_PRED_LAG_REDUCTION = 0.05`). Pure-physics regression tests are
pinned to 1.0 explicitly.
