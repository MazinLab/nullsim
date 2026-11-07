# FringeTracking Stage Notes

## Source Survey

The task path `kilo/kilo/fringe_tracking.py`
is stale for this checkout. The KILO package root is
`kilo`. The fringe tracker lives at
`kilo/fringe_tracking.py` (706 lines).

Files inspected:

- `kilo/fringe_tracking.py` in full.
- `nayra/fringe_tracking.py`
  in full (a thin re-export shim over KILO).
- Existing nullsim Pass 1 stages (`kolmogorov_screens.py`, `pyramid_wfs.py`)
  for the `OPDComponent.superseded` precedent and `OPDBudget` totals
  convention.
- `nullsim/pipeline/state.py` for `PhotonRates.rates_hz` shape
  `(n_ports, n_wavelength)` produced by Pass 4a SNSPD/MKID detector stages.

## KILO Reference Functions

The two formulas the nullsim stage implements come from `kilo.fringe_tracking`:

### Shao-Colavita 1992 per-channel shot-noise variance

The shared primitive `shot_noise_phase_variance_per_frame` at lines 67-95
returns

    sigma2_phi_per_frame = (N_signal + N_dark) / (V^2 * N_signal^2)

in rad^2, where `N_signal = rate_per_channel * tau` and
`N_dark = dark_per_channel * tau`. Used as the per-channel primitive for
`fringe_tracking_phase_noise` (lines 98-183), which averages over
`n_spectral_channels` independent AWG bins:

    sigma2_photon = sigma2_phi_per_frame / n_spectral_channels

`fringe_tracking_phase_noise` defines the per-frame integration time as
the Nyquist sample of the servo cycle: `tau_coh = 1 / (2 * servo_bw_hz)`.
In the bright-port, zero-dark-count limit this collapses to the canonical

    sigma2_photon ~= 2 * servo_bw_hz
                    / (V^2 * n_spectral_channels * rate_total)

so the nullsim stage uses the closed form

    sigma2_photon = 1 / (V^2 * n_spectral_eff * N_phot_per_channel * 2 * pi^2)

per the task spec (equivalent up to a 4 * pi^2 factor that absorbs the
per-cycle averaging into a single Shao-Colavita estimator over the
coherent integration time). When dark counts and saturation matter the
fuller diagnostic in `piston_servo_residual` (KILO lines 335-471) is the
reference path.

### Conan 1995 servo lag

`fringe_tracking_phase_noise` derives the servo-lag variance from a
piston-specific Greenwood frequency:

    f_piston = 0.134 * v_wind / D_eff
    sigma2_lag = (f_piston / f_servo) ** (5/3)

KILO uses the *effective telescope diameter* for `D_eff` because piston
is averaged over the full aperture, giving `f_piston << f_Greenwood`
(KILO `effective_diameter_m = 10.0` Keck baseline). The
`piston_servo_residual` diagnostic at lines 335-471 multiplies the lag
term by `predictive_lag_reduction` and the temporal piston variance:

    sigma2_lag = predictive_lag_reduction * sigma2_temp_correctable
                 * (f_char / f_s) ** (5/3)

with `f_char = 0.134 * v_wind / d_sub_m` for the per-sub-aperture case
and `d_sub_m = D_eff` for the per-telescope case.

nullsim no longer implements the standalone `fringe_tracking_phase_noise`
lag (review 2026-11-07 row 43): `(f_piston / f_servo)^(5/3)` rad^2 carries
no r0 or upstream piston amplitude, and a fixed phase variance makes the
OPD scale with wavelength. `use_full_piston_servo_residual` now defaults to
true and false is rejected; only the `piston_servo_residual` form remains.

The nullsim stage applies the lag formula to the upstream atmospheric
piston RMS (`atmos_piston` OPDComponent) rather than the AO-residual
`ao_temporal` because the fringe tracker corrects the residual *after*
AO; when AO has already superseded `atmos_piston` the stage falls
through to the AO-residual components named by `superseded_components`
override or by inspection of the OPD budget.

## KILO `FringeTrackingConfig` Defaults

Lines 23-62 of `kilo/fringe_tracking.py`:

- `servo_bandwidth_hz = 2000.0`
- `fringe_visibility = 0.9`
- `n_spectral_channels = 10`
- `n_spectral_eff = 2.5`
- `effective_diameter_m = 10.0`  (Keck baseline diameter; nullsim derives
  this from `geometry.diameter_m` and exposes an override Param)
- `predictive_lag_reduction = 1.0`  (KILO default = leaky integrator,
  no Kalman prediction)

## NAYRA Differences

NAYRA's `nayra/fringe_tracking.py` is a thin wrapper over KILO with two
defaults flipped (2026-05-13, Ben):

- `predictive_lag_reduction = 0.05` (20x variance reduction from LQG /
  Kalman predictive piston control). Exposed as
  `NAYRA_DEFAULT_PISTON_PRED_LAG_REDUCTION = 0.05`.
- `predictive_gain_variance = 20.0` on the independent
  `amplitude_servo_residual` (lines 76-160 of the NAYRA shim, KILO
  reference at lines 589-706). Exposed as
  `NAYRA_DEFAULT_AMP_PRED_GAIN_VARIANCE = 20.0`. KILO's own default for
  the amplitude servo's predictive gain is 1.0 (no prediction).

The nullsim stage carries `predictive_lag_reduction` as a Param
defaulting to 1.0 (KILO behavior); presets/scene configs flip it to
0.05 for the NAYRA E-ELT preset.

The amplitude (scintillation) servo is NAYRA-only and is *not* in scope
for this stage — it is a separate observable on bright-port intensity
not phase, and would land as a future `fringe_tracking.amplitude_servo`
stage. Its NAYRA-specific `predictive_gain_variance` kwarg is documented
here for completeness; the piston-only nullsim stage does not consume it.

## Frozen Reference Pins

KILO `FringeTrackingConfig()` defaults reproduce the formulas above with
no scaling factors. In the bright-port, zero-dark-count limit:

- Doubling `bright_port_photon_rate` halves `sigma2_photon`.
- Doubling `servo_bw_hz` shrinks `sigma2_lag` by `2 ** (5/3) ~= 3.175`.
- Setting `predictive_lag_reduction = 0.05` shrinks `sigma2_lag` by 20x,
  shrinking the total residual by ~`sqrt(20) ~= 4.47x` in the
  lag-dominated regime.
- In the noise-limited regime (`servo_bw -> inf`), `sigma2_lag -> 0` and
  the residual converges to the Shao-Colavita shot-noise floor.

## OPD Component Naming

The nullsim stage writes one `fringe_tracking.closed_loop_residual`
component to `state.opd_budget.components` with the quadrature-summed
residual sigma converted to `rms_opd_nm` via
`sigma_rad * wavelength_center_m / (2 * pi) * 1e9`. The upstream piston
component (`atmos_piston` by default, overridden by Param if AO has
already superseded it) is marked `superseded=True` via
`dataclasses.replace` — the same pattern Pass 1's `pyramid_wfs` uses on
`atmos_piston`. Superseded components stay in `components` for plot
history but are dropped from the rebuilt `total_rms_opd_nm` quadrature
sum.

The stage emits a single component, not per-arm, because nullsim's
current `OPDBudget` is a single global sub-budget; per-aperture
diagnostics live in `state.results` (deferred to the second-pass
diagnostic stage). If `geometry.n_apertures > 1` the same closed-loop
residual is reported globally — KILO's `compute_servo_corrected_piston`
multi-telescope helper similarly returns one per-telescope residual that
the caller broadcasts.

## Units And Gotchas

- `wavelength_grid_um` is microns; `wavelength_center_m` is meters.
- `OPDComponent.rms_opd_nm` is nanometers (RMS), positive.
- `sigma2_*` are variances in rad^2 at the representative science
  wavelength (middle of the grid, matching Pass 1 convention).
- `wind_speed_m_s` is meters/second. It is not in `SimulationState`
  today (the atmosphere stage holds it on Params); the fringe-tracking
  stage carries it as its own Param.
- `effective_baseline_m` defaults to `geometry.diameter_m` when None;
  for multi-aperture arrays this is the per-telescope aperture, not the
  inter-telescope baseline (piston is averaged within one aperture, not
  across the baseline).
- The stage cannot make the budget worse: when the upstream piston
  component does not exist (e.g. no atmosphere stage ran), the stage
  raises `ValueError`. There is no implicit "noise-only" mode.

## Source Lines For Pin Tests

- KILO `FringeTrackingConfig` dataclass: lines 24-62.
- KILO `shot_noise_phase_variance_per_frame`: lines 67-95.
- KILO `fringe_tracking_phase_noise` (top-level): lines 98-183.
- KILO `piston_servo_residual` (full diagnostic, Kalman-aware): lines
  335-471.
- KILO `amplitude_servo_residual` (scintillation, NAYRA-only consumer):
  lines 589-706.
- NAYRA shim `FringeTrackingConfig` factory and predictive defaults:
  lines 60-79 of `nayra/fringe_tracking.py`.
- NAYRA `amplitude_servo_residual` predictive override and
  `predictive_gain_variance` kwarg: lines 81-160 of the same file.
