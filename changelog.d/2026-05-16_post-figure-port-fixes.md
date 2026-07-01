---
date: 2026-05-16
---

## Post-figure-port fixes: Fig 4, Fig 9, chip-optimizer tricks, injection wavefront carry, detector null depth (commits c61192f, a22770d, f2b3d2b, e4d84f0, ba5095d)

**plots: loss_budget — paperv3 Fig 4 (c61192f):**
`nullsim/outputs/plots/loss_budget.py` — per-band stacked-dB pre-chip loss bars. Scalar adaptation of KILO's dual-polarization figure: drops the scalar-vs-dual comparison axis, reads per-stage decomposition from `state.throughput.components`.

**plots: calibration_time — paperv3 Fig 9 scoped (a22770d):**
`nullsim/outputs/plots/calibration_time.py` — chip optimizer convergence curve (null depth vs objective evaluations). Single-panel port; right panel (null vs calibration budget at multiple J-mags) deferred until photon-noise-aware optimization ships. Requires `capture_convergence_history = true` on `ChipOptimizationParams`.

**perf: chip-optimizer KILO tricks (f2b3d2b):**
`ChipOptimizationParams` gains four fields: `n_restarts` (default 5, was hardcoded 16), `restart_patience` (default 3, early-exit when K consecutive restarts fail to improve by `restart_patience_reltol`), `restart_patience_reltol` (default 0.01), and `n_workers` (default 1, chip-restart ProcessPoolExecutor). `examples/kilo_keck.toml` updated to `n_restarts=5`, `restart_patience=3`, `n_workers=2`.

**fix(injection): carry forward upstream wavefront components (e4d84f0):**
`injection.ideal` was constructing a fresh `Wavefront()` without propagating upstream `wavefront.components`, silently dropping OPD and amplitude contributions from atmosphere/AO stages. Fixed by passing `components=dict(state.wavefront.components)` to the constructor.

**fix(detector): compute null_depth_raw from pre-saturation rates (ba5095d):**
`detector.mkid` and `detector.snspd` were computing `null_depth_raw` from per-port rates after the saturation cap, QE, and dark-count steps. When both bright and dark ports saturated, the ratio collapsed to the pixel-count ratio (~1e-3) regardless of the actual chip null. Fixed by computing `null_depth_raw` from `port_rate_hz_per_wavelength` before saturation/QE/darks, matching KILO's `null_depth()` convention (`kilo/performance.py` L164).
