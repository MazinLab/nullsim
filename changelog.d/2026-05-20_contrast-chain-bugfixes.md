---
date: 2026-05-20
---
### Contrast-chain correctness fixes (five-commit series)

A series of five correctness fixes addressed semantic errors throughout the
null-depth / contrast computation chain:

**sensitivity.detection_curve** now supports `deployment_mode="per_pol_arm"`
from `chip.kernel_mzi_mesh`, incoherently combining per-arm planet throughputs
weighted by injected flux. The realized-null computation is per-arm and then
flux-averaged, matching the NAYRA dual-PBS front end.

**chip_optimization (scipy + torch)** fixed the target-null early-exit
comparison: the optimizer objective is KILO-convention absolute leakage, but
`target_null_depth` is a dimensionless null depth. The exit now normalizes by
total input power before comparing, preventing spurious early stops when input
power is not unity. A new `optimization_normalized_null_final` field is
published to results for downstream diagnostics.

**performance.standard** received several fixes:
- `eps_used` (null-instability systematic variance) now combines
  `eps_calibration_residual` and the sensitivity-stage `null_instability` in
  quadrature (`math.hypot`) instead of ignoring the config residual when a
  sensitivity stage runs.
- Total stellar rate for leakage normalization now includes spectro-port rates
  alongside bright and leakage ports.
- A new `contrast_curve_reference_time_s` Params field (default 3600 s)
  controls the reference integration time for the contrast curve, replacing the
  implicit 1-hour assumption.
- Calibration floor is applied in `performance.standard` when it exceeds the
  sensitivity-stage effective null, with a `RuntimeWarning`.

**detector.snspd** now separates stellar signal rates from dark-count rates:
`port_rates` carries signal-only counts (matching the null-depth estimator
convention); `dark_rate_hz_per_port` and `total_port_rates_with_dark` are
published as explicit diagnostic fields. The stage now also produces
`throughput` by folding detector QE into the throughput budget, matching
`detector.mkid`.

**detection_space plot** corrected contrast-masking logic for multi-band and
swept runs.
