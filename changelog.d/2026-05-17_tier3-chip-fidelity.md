---
date: 2026-05-17
---

### Tier 3 — Chip fidelity (BFGS + throughput penalty + Fresnel loss)

Three of the four Tier 3 fixes. Tier 3.1 (per-wavelength phase rescaling
in the deployed chip) is deferred — it requires resolving the layout
mismatch between the torch optimizer's KILO column-parity convention
and the chip stage's physics.clements convention, which the "Ultra-
review High #5" pass intentionally settled with monochromatic
deployment. Re-opening that needs a chip-stage refactor (read the raw
KILO-layout params + per-wavelength rescaling) and is queued.

#### 3.2 Switch chip optimizer from L-BFGS-B to BFGS

- `nullsim/stages/chip_optimization/torch.py:432,472`: both the
  bootstrap and broadband BFGS calls now use scipy's full-Hessian
  `method="BFGS"`, matching KILO `chip.py:1180,1212`. On a 576-dim
  multi-modal kernel-null landscape the low-memory L-BFGS-B
  approximation lands in a different basin from the same init — the
  most common cause of "same algorithm, different null floor" runs.

#### 3.3 Throughput penalty in broadband objective + drop α to KILO default

- The throughput penalty `−α × <planet_throughput>_{sep, PA, wl}` now
  runs inside the broadband objective at every wavelength bin (KILO
  `natural_design.py:KILO_BB_THROUGHPUT_PENALTY_ALPHA` convention).
  Pre-fix nullsim only applied the penalty inside the single-λ
  bootstrap, so `broadband_refine` runs drifted toward narrow-band
  null-only basins.
- `examples/kilo_keck.toml`: `throughput_penalty_alpha` dropped from
  5.0e-2 to KILO's 1.0e-3 default. The previous 50× over-weighting
  was the workaround for the bootstrap-only penalty; with the
  broadband penalty in place that compensation is no longer needed.

#### 3.4 Fresnel reflection loss

- `nullsim/physics/ruilier_cassaing.py` gains `FRESNEL_REFLECTION_LOSS
  = 0.035` and an `apply_fresnel_loss: bool = False` kwarg on
  `coupling_efficiency`. When True, multiplies by `(1 - 0.035)` —
  KILO `injection.py:25,72` convention.
- `DetectionCurve.Params` gains
  `apply_fresnel_injection_loss: bool = False`; when True the
  sensitivity stage's `coupling_per_sub` includes the Fresnel loss
  (applies even when `apply_strehl_coupling` is False, since Fresnel
  is an optical loss separate from wavefront quality).
- The exact annular-overlap `geometric_coupling_limit` is kept
  (more accurate than KILO's `0.81 × (1 − ε²)²` Taylor approximation).
  For Keck ε=0.24 this gives 0.756 vs KILO 0.7197 — a documented ~5%
  geometric-throughput difference flagged in the next Tier 4 commit.

#### Regression tests

- `test_coupling_efficiency_fresnel_loss_applied_when_flagged`,
  `test_coupling_efficiency_fresnel_loss_default_off` —
  pin the Fresnel-loss kwarg behavior in
  `nullsim/physics/ruilier_cassaing.py`.

642 non-parity tests pass.
