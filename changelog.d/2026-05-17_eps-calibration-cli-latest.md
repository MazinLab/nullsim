---
date: 2026-05-17
---

### chip_optimization.torch RNG pin (commit 0e52d8f2)

- `ChipOptimizationTorch.apply` now pins `torch.manual_seed(seed_base)` immediately
  after constructing the numpy master RNG. Without this, PyTorch's internal RNG was
  seeded from system entropy each run, making restart initializations non-reproducible
  even when `params.seed` was fixed. The CPU MKL thread-reduction non-determinism is
  intentionally not pinned to avoid the ~10× perf cost; it is small enough that BFGS
  basins are stable from identical inits.

### Port KILO eps_calibration noise model (commits c9feec43, d968b86e, fbcf59de)

- `DetectionCurve.Params` gains `effective_systematic_bandwidth_hz_override`
  (`float | None`, default `None`, gt=0). When set, forwards as
  `effective_systematic_bandwidth_hz` in the sensitivity component payload so
  `StandardPerformance._systematic_variance` applies
  `eps² × leak² × t / (2 f_eff)` with the supplied f_eff instead of the
  performance stage's own `default_servo_bw_hz`. Port of KILO
  `studies/natural_design.py` L1240 (`_effective_systematic_bandwidth_hz`;
  typical Keck values 1223 Hz at J=4, 206 Hz at J=8).
- The bandwidth override also controls `effective_null_override` semantics: when
  absent, MC `null_instability` is zeroed (KILO detection_space parity, matching
  `contrast_curve` default `null_instability=0`); when present, the MC CV is
  propagated so the systematic floor activates.
- `StandardPerformance.apply` reads `sens_payload["effective_systematic_bandwidth_hz"]`
  and overrides `servo_bw_hz` before SNR/contrast-curve computation when non-None.
  `_resolve_servo_bw` is unchanged.
- `kilo_keck.toml`: `effective_null_override` bumped 1.5e-5 → 1.95e-5 for KILO
  13/14-point parity; `ao_piston_scale` revised 0.4 → 0.064 to match KILO servo
  σ_OPD ≈ 0.67 nm at Keck; `scintillation_index_per_sub` set to 0.0 (disabled,
  since `effective_null_override` absorbs the realized-null floor).

### CLI results/latest symlink (commit 8278022)

- `_handle_run` calls `_update_latest_symlink(output_dir)` after writing all outputs.
  Creates a relative symlink `results/<name>/latest → <run-hash-dir>` so users have
  a stable handle alongside the hex-hash directory name. Relative so the parent
  results directory can be moved. Best-effort: non-symlink occupants of `latest` are
  left untouched; `OSError` is swallowed since the run artifacts are already on disk.
