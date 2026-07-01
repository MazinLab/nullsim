# PyramidWFS Stage Notes

## Source Survey

The task path `kilo/kilo/...` is stale for
this checkout. The KILO package root is `kilo`.

Files inspected:

- `kilo/pwfs.py` in full.
- `kilo/config.py` in full, with the AO
  blocks at `TelescopeConfig`, `AtmosphereConfig`, and `maunakea_telescopes()`.
- `kilo/atmosphere.py` AO/PWFS functions:
  `ao_residual_variance`, `strehl_ratio`, `ao_piston_phase_components`, and
  `ao_piston_opd_rms`.
- `nayra/wfs.py` in full.
- `nayra/config.py` in full,
  with the WFS/AO blocks at `TelescopeConfig`, `AtmosphereConfig`, and
  `eelt_telescope()`.
- `nayra/atmosphere.py`, which is
  a thin KILO atmosphere adapter.
- Existing KILO/NAYRA tests under `kilo/tests`
  and `nayra/tests` were checked for
  reference values and invariants.

## KILO Residual Terms

KILO's AO residual phase variance has exactly three summed terms:

- `fitting`
- `temporal`
- `noise` (the pyramid-WFS measurement-noise term when `telescope.pwfs` is set)

The returned dictionary is `{"fitting", "temporal", "noise", "total"}` where
`total = fitting + temporal + noise`, all in rad^2 at the science wavelength.

KILO does not add anisoplanatism to `ao_residual_variance`. `theta0_500nm_arcsec`
is present on `AtmosphereConfig`, but it is not used in the AO residual formula.
Scintillation exists as separate amplitude-jitter functions in
`kilo.atmosphere`; it is not part of the AO residual phase-variance sum.

## Formula Sketches

All formulas below use KILO names and units.

Reference scaling:

- `r0_zenith = scale_r0(atmosphere.r0_500nm, wavelength_m)`
- `r0 = r0_zenith * airmass**(-0.6)`
- `scale_r0` uses `(wavelength_m / 500e-9)**1.2`.

Fitting error:

- `d_act = telescope.ao_actuator_pitch`
- KILO pitch: `telescope.diameter / int(sqrt(telescope.ao_actuators))`
- `sigma2_fit = C_FIT * (d_act / r0)**(5/3)`
- `C_FIT = 0.28`

Temporal bandwidth error:

- `f_g = greenwood_frequency(atmosphere.v_wind_ms, r0)`
- `greenwood_frequency = 0.426 * v_wind_ms / r0_m`
- `f_servo = telescope.ao_loop_speed_hz / 10.0`
- `sigma2_temp = (f_g / f_servo)**(5/3)`

Pyramid-WFS noise, active when `telescope.pwfs is not None`:

- `sigma2_at_wfs = wfs_noise_phase_variance(guide_star_mag, telescope.diameter, telescope.pwfs, central_obstruction=telescope.central_obstruction, airmass=airmass)`
- `wl_ratio = telescope.pwfs.wavelength_m / wavelength_m`
- `sigma2_noise = sigma2_at_wfs * wl_ratio * wl_ratio`

Backward-compatible fallback when `telescope.pwfs is None`:

- `noise_scale = 10.0**(0.4 * (guide_star_mag - 8.0))`
- `sigma2_noise = 0.01 * noise_scale * (3000 / telescope.ao_actuators)`
- This branch is zenith anchored and does not include WFS photon-flux airmass
  scaling.

Pyramid-WFS internals:

- `pyramid_sensitivity(modulation_lambda_d) = 1.0 / (1.0 + 2.0 * modulation_lambda_d)`
- `n_subap_illuminated = round(n_subap_across**2 * pi * (1 - central_obstruction**2) / 4)`
- `flux_top_atm = 3.01e10 * 10**(-0.4 * guide_star_R_mag) * (wfs.bandwidth_m * 1e6)` photons/s/m^2
- `A_tel = pi * (telescope_diameter_m / 2)**2 * (1 - central_obstruction**2)`
- `atm_T = wfs.atm_throughput**airmass`
- `rate_at_detector = flux_top_atm * A_tel * atm_T * wfs.mirror_throughput * wfs.dichroic_T * wfs.qe`
- `N_ph = rate_at_detector / (n_subap_illuminated * wfs.loop_frequency_hz)`
- `measurement_variance = wfs.excess_noise_sq * (N_ph + wfs.readout_noise_e_rms**2)`
- `sigma2_at_wfs = measurement_variance / (N_ph**2 * sensitivity**2)`

## WFS Photon Rate

KILO and NAYRA both compute guide-star WFS photons with their own hard-coded
R-band Vega zero point:

- `_VEGA_R_ZEROPOINT_PH_S_M2_UM = 3.01e10`

They do not call `nullsim.physics.photometry`; the current nullsim photometry
primitive only exposes the 2MASS J-band zero point. The future stage should
either carry an explicit R-band WFS zero point locally or add an R-band
photometry primitive deliberately.

The KILO `ao_residual_variance` docstring calls `guide_star_mag` V-band for
historical reasons, but the active PWFS branch interprets it as R-band. NAYRA's
WFS functions name the parameter `guide_star_R_mag`.

## Strehl Convention

KILO uses the Marechal/extended-Marechal scalar convention:

- `strehl_ratio(sigma2_rad2) = exp(-sigma2_rad2)`

`sigma2_rad2` is the summed AO residual phase variance at the science
wavelength. This is not the Noll full-aperture Kolmogorov variance and not the
piston-filtered phase variance; it is the full AO residual wavefront variance
for injection/coupling.

## OPD Components

KILO does not write a named OPD budget object. It exposes phase diagnostics:

- `ao_residual_variance(...)["fitting"]`
- `ao_residual_variance(...)["temporal"]`
- `ao_residual_variance(...)["noise"]`
- `ao_residual_variance(...)["total"]`

For coupled-mode piston, KILO exposes:

- `AOPistonComponents.temporal`
- `AOPistonComponents.fitting`
- `AOPistonComponents.wfs_noise`
- `AOPistonComponents.total`
- `AOPistonComponents.frac_temporal`
- `AOPistonComponents.frac_fitting`
- `AOPistonComponents.frac_wfs_noise`

The nullsim stage should map these to explicit OPD budget names:

- `ao_temporal`
- `ao_fitting`
- `ao_wfs_noise`

The component OPD RMS conversion is `sqrt(phase_variance_rad2) * wavelength_m /
(2*pi)`, stored in nanometers in `OPDComponent.rms_opd_nm`. The quadrature
sum of those three components should match KILO's `ao_piston_opd_rms`.

## Units And Gotchas

- KILO and NAYRA store `r0_500nm`, `tau0_500nm`, and `theta0_500nm_arcsec` at
  500 nm.
- `theta0_500nm_arcsec` is arcseconds, but it is not used by KILO's AO
  residual phase-variance calculation.
- `wavelength_m` and `wfs.wavelength_m` are meters.
- `wfs.bandwidth_m` is meters and is converted to microns with `* 1e6` before
  multiplying the R-band zero point.
- `central_obstruction` is a linear diameter fraction. Area uses
  `1 - central_obstruction**2`.
- KILO accepts any finite R magnitude, including negative values. NAYRA rejects
  `guide_star_R_mag` outside `(-30, 50)`.
- The WFS noise returned by `wfs_noise_phase_variance` is at the WFS wavelength
  and must be scaled to the science wavelength by `(lambda_wfs/lambda_science)^2`.
- The residual terms are variances, so no sign convention enters the AO budget.
  OPD is a positive RMS.
- With `d_sub_m=None`, KILO uses conservative piston fractions:
  `fitting=0.03`, `temporal=1.0`, `wfs_noise=1.0`. With `d_sub_m` set, it
  integrates the Bessel piston filter against each term's PSD.

## NAYRA Differences

NAYRA's `nayra.wfs` and KILO's `kilo.pwfs` use the same closed-form pyramid-WFS
model in the current checkout. The main differences are defaults and config
shape:

- NAYRA `PyramidWFSConfig`: `n_subap_across=80`, `loop_frequency_hz=1000.0`.
- KILO Maunakea defaults: Keck and Gemini use `n_subap_across=50`,
  Subaru uses `n_subap_across=56`, all at `loop_frequency_hz=2000.0`.
- NAYRA stores the WFS as `TelescopeConfig.wfs` and exposes a `pwfs` property
  so KILO's atmosphere code can consume it.
- NAYRA's actuator pitch is `diameter / sqrt(ao_actuators * 4/pi)`, while
  KILO uses `diameter / int(sqrt(ao_actuators))`.
- NAYRA's atmosphere module delegates to KILO and wraps tuple-form Strehl
  outputs for the single-telescope case.

## Frozen Reference For Pinning Tests

The xfail-strict tests pin this KILO reference:

- Telescope: `maunakea_telescopes()[0]` (Keck I)
- Atmosphere: KILO default `AtmosphereConfig()`
- Science wavelength: `1.65e-6 m`
- Guide star: `guide_star_R_mag = 8.0`
- Airmass: `1.0`
- Piston sub-aperture diameter: `d_sub_m = 1.8`

Reference values from current KILO source:

- `wfs_photons_per_subap_per_frame = 73.60971154412574`
- `wfs_noise_phase_variance_at_wfs_rad2 = 0.02726260624964504`
- AO residual phase variance:
  - `fitting = 0.036530458245298755 rad^2`
  - `temporal = 0.0035491091904672413 rad^2`
  - `noise = 0.004906768434279547 rad^2`
  - `total = 0.044986335870045545 rad^2`
- `strehl = 0.9560105447961746`
- Piston-filtered phase variance:
  - `temporal = 0.0016348878294317967 rad^2`
  - `fitting = 4.3048451547883666e-06 rad^2`
  - `wfs_noise = 8.075372145221129e-05 rad^2`
  - `total = 0.0017199463960387964 rad^2`
- Piston OPD RMS:
  - `ao_temporal = 10.618130441917643 nm`
  - `ao_fitting = 0.5448573867633121 nm`
  - `ao_wfs_noise = 2.3598551618844352 nm`
  - total `ao_piston_opd_rms = 10.89084386255543 nm`

