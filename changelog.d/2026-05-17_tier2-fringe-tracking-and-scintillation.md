---
date: 2026-05-17
---

### Tier 2a — Fringe-tracking formulas + scintillation/f_eff physics ports

Three bugs in `nullsim/stages/fringe_tracking/closed_loop.py` and two
missing physics modules that block KILO Keck-only reproduction. The
formula fixes land here; pipeline wiring of the new physics modules
into the sensitivity stage (Tier 2.3b, 2.4, 2.5b) is the next commit.

#### 2.1 Conan servo-lag formula (closed_loop.py)

- `_conan_servo_lag_variance_rad2` no longer multiplies by the upstream
  atmospheric piston variance. KILO `fringe_tracking_phase_noise`
  (`kilo/fringe_tracking.py`:180-182) uses the standalone
  `(f_piston / f_servo)^(5/3)` form — the bandwidth-ratio factor
  already carries the absolute lag variance. Pre-fix nullsim's
  ×σ²_atmos multiplication inflated the lag by 2-3 orders of magnitude
  at J-band Keck and forced `ao_piston_scale = 0.064` in
  `examples/kilo_keck.toml` to compensate downstream.

#### 2.2 Shao-Colavita shot-noise formula (closed_loop.py)

- `_shao_colavita_shot_variance_rad2` now matches KILO
  `shot_noise_phase_variance_per_frame` (`kilo/fringe_tracking.py`:67-95)
  exactly: `sigma2_per_channel = (N_s + N_d) / (V² × N_s²)`.
- Cross-channel averaging (`/n_spectral_channels`) is applied in
  `apply()` to match KILO `fringe_tracking_phase_noise`. The
  pre-fix formula `1 / (V² × n_spectral_eff × N_s × 2π²)` was off by
  ~158× from KILO (mixed `n_spectral_eff` and `n_spectral_channels`
  conventions, spurious `2π²`).
- New `Params.dark_rate_hz_per_channel` (default 0.0 → MKID
  photon-counting limit) carries the FT-readout dark-count term.
- Tolerance on three existing FT scaling tests loosened modestly
  (5e-3) to absorb the small shot residual now that the lag term is
  no longer artificially amplified.

#### 2.3a Scintillation physics module port

- New `nullsim/physics/scintillation.py` with
  `scintillation_index`, `scintillation_timescale_for_servo_design`,
  and `amplitude_servo_residual_sigma_i` — direct ports of KILO
  `kilo/atmosphere.py`:95-481. Returns the residual σ_I per
  sub-aperture used by the realized-null MC.
- `AmplitudeServoResidual` dataclass carries the full breakdown
  (open-loop, temporal-lag, photon-floor) for diagnostics.
- Tests pin the analytic limits (λ scaling, d_sub scaling, airmass
  scaling, Fresnel-radius saturation, photon-limited / temporal-
  dominated regimes, KILO Keck-only J-band order-of-magnitude).

#### 2.5a Effective systematic bandwidth port

- New `nullsim/physics/systematic_bandwidth.py` with
  `effective_systematic_bandwidth_hz` — direct port of KILO
  `_effective_systematic_bandwidth_hz`
  (`kilo/studies/natural_design.py`:180-238). Variance-weighted blend
  of AO piston and amp-servo bandwidths for the `eps² × leak² × t /
  (2 f_eff)` systematic noise term.
- Tests pin both single-process limits (collapses to f_piston or
  f_amp), the balanced-contributor harmonic-mean ballpark, and the
  KILO Keck-only J=4 regime (σ_I dominates → f_eff ≈ f_amp).

#### Regression tests added

- `test_closed_loop_lag_matches_kilo_standalone_formula` — pins the
  bare `(f_piston/f_servo)^(5/3)` lag against the photon-noise-free
  limit so any regression to the ×σ²_atmos multiplication shows up.
- `test_closed_loop_shao_colavita_matches_kilo_formula` — pins the
  shot-noise term against the KILO analytic `(N_s + N_d) / (V² × N_s²)`
  with cross-channel averaging.
- `tests/physics/test_scintillation.py` (11 tests).
- `tests/physics/test_systematic_bandwidth.py` (7 tests).
