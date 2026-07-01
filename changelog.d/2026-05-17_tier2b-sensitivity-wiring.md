---
date: 2026-05-17
---

### Tier 2b — Sensitivity-stage wiring: auto σ_I, f_eff blend, AO+FT quadrature

Pipeline-side wiring for the Tier 2a physics ports. After this pass the
sensitivity stage can derive both the amp-servo scintillation residual
and the variance-weighted effective bandwidth from physics, with the
manual override stack as a fallback.

#### 2.3b Auto-derived σ_I from physics

- `DetectionCurve.Params` gains
  `auto_amplitude_servo_residual: bool = False`. When True, σ_I is
  derived per-band from
  `nullsim.physics.scintillation.amplitude_servo_residual_sigma_i`
  (KILO `kilo/atmosphere.py:311-481` port) and replaces the manual
  `scintillation_index_per_sub` scalar passed to the realized-null MC.
- Companion params with KILO Maunakea defaults:
  `auto_scintillation_anchor_h5m_zenith=0.041`,
  `auto_scintillation_height_km=10.0`,
  `auto_high_altitude_wind_ms=35.0`,
  `auto_amp_servo_bandwidth_hz=5000.0`,
  `auto_amp_predictive_gain_variance=20.0`,
  `auto_amp_servo_photon_rate_per_sub_hz` (required when auto mode is on).
- New `amplitude_servo_residual` payload field carries the full
  `AmplitudeServoResidual` breakdown (open-loop, temporal, photon-floor)
  for diagnostics.

#### 2.5b Auto-derived f_eff blend

- `DetectionCurve.Params` gains
  `auto_f_piston_hz: float = 100.0` (AO piston servo bandwidth used by
  the blend; KILO Keck typical 30-250 Hz at J=4).
- In `apply()`, the f_eff propagated to the performance stage now uses
  this precedence:
  1. explicit `effective_systematic_bandwidth_hz_override`,
  2. variance-weighted blend
     (`nullsim.physics.systematic_bandwidth.effective_systematic_bandwidth_hz`)
     when `auto_amplitude_servo_residual=True`,
  3. `None` — performance stage falls back to its own default.

#### 2.4 `_find_ao_piston_rms_per_sub_m` combines AO + FT in quadrature

- When both `ao_residual.piston_opd_nm.total` and a
  `fringe_tracking.closed_loop_residual` OPD component are present, the
  per-sub piston returned to the MC is now the quadrature sum:
  σ_per_sub = √(σ²_AO_within_tel + σ²_FT_common). This matches the
  physical model: the FT corrects only the common inter-telescope
  piston, so the chip null sees AO-differential ⊕ FT-residual.
- Falls back to AO-only when no FT stage has run (e.g. space configs)
  and to FT-only when no AO stage has run (back-compat).
- Full per-telescope `(M,)` array support — KILO
  `compute_servo_corrected_piston` (`kilo/fringe_tracking.py:473-580`) —
  is deferred until pyramid_wfs emits per-telescope piston OPD.

#### Regression tests

- `test_detection_curve_auto_amplitude_servo_derives_sigma_i_and_f_eff`
  pins both σ_I and the f_eff blend against direct physics-function
  calls.
- `test_detection_curve_auto_amp_servo_requires_photon_rate` pins the
  clear error when auto mode is requested without a per-sub photon rate.

#### Not yet flipped on for kilo_keck.toml

`examples/kilo_keck.toml` still uses the manual override stack
(`effective_null_override`, `scintillation_index_per_sub=0.0`,
`ao_piston_scale=0.064`). Flipping it to the auto-mode physics requires
choosing the right per-sub photon rate from the upstream chain; that
choice is a config decision tracked as a Tier 4 followup. The pipeline
support is in place.
