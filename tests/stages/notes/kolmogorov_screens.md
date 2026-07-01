# KolmogorovScreens Stage Notes

## Source Survey

Searched for `scale_r0`, `r0`, `atmos`, `piston`, `strehl`, `FIXME`,
`NOTE`, `TODO`, and `theta0` in the KILO and NAYRA source trees.

Files inspected:

- `kilo/atmosphere.py` (905 lines).
- `kilo/config.py`, especially
  `TelescopeConfig` and `AtmosphereConfig`.
- `nayra/atmosphere.py`
  (101 lines).
- `nayra/config.py`,
  especially `TelescopeConfig`, `AtmosphereConfig`, and `eelt_telescope`.

The SPEC path `kilo/kilo/...` is stale
for this checkout. The actual import package is rooted at
`kilo`.

## KILO To nullsim Mapping

KILO has one atmosphere module that combines Fried-parameter scaling,
scintillation, AO residual phase variance, Marechal Strehl, and the
fiber-coupled piston OPD estimate. NAYRA does not maintain a divergent copy:
`nayra/atmosphere.py` imports and re-exports `kilo.atmosphere`, then wraps
`compute_strehl_per_telescope((telescope,), ...)` as scalar
`compute_strehl(...)`.

Function mapping:

- `kilo.atmosphere.scale_r0(r0_500nm, wavelength_m)` maps directly to
  `nullsim.physics.kolmogorov.r0_scaled`.
- `kilo.atmosphere.strehl_ratio(sigma2_rad2)` maps to
  `nullsim.physics.kolmogorov.strehl_marechal` after converting
  `sigma2_rad2` to an OPD RMS, but the KILO stage path usually keeps the
  phase variance and uses `exp(-sigma2)`.
- `kilo.atmosphere.ao_piston_opd_rms(...)` uses the same conversion as
  `nullsim.physics.kolmogorov.opd_rms_m`: `sqrt(sigma2_phase) * lambda /
  (2*pi)`. The source of `sigma2_phase` is not the Noll full-aperture
  variance; it is KILO's AO residual piston budget.
- `kilo.atmosphere.ao_residual_variance(...)` has no one-line nullsim
  physics primitive. It is a stage-level AO budget: fitting, temporal, and
  WFS-noise terms in rad^2.
- `kilo.atmosphere.scintillation_index(...)` should populate the stage's
  wavefront amplitude-jitter field until a dedicated scintillation primitive
  exists.

## Wavelength And Unit Conventions

KILO and NAYRA store atmospheric reference values at 500 nm:

- KILO defaults: `r0_500nm=0.15 m`, `tau0_500nm=2.83e-3 s`,
  `theta0_500nm_arcsec=1.19`, `v_wind_ms=10.0`.
- NAYRA defaults: `r0_500nm=0.157 m`, `tau0_500nm=5.5e-3 s`,
  `theta0_500nm_arcsec=2.5`, `v_wind_ms=9.0`.

Scaling is `value(lambda) = value(500 nm) * (lambda / 500e-9)**1.2`.
This is already pinned in `nullsim.physics.kolmogorov.r0_scaled`.

KILO's AO residual and piston phase budgets are in rad^2 at the observing
wavelength. KILO converts piston phase variance to OPD RMS in meters with
`sqrt(sigma2_phase) * wavelength_m / (2*pi)`. `OPDComponent.rms_opd_nm`
should store that result in nanometers.

## Noll Piston Convention

KILO does not use the Noll 1976 full-aperture phase-variance coefficients in
the AO path. The stage should not approximate KILO's `ao_residual_variance`
with `phase_variance(D, r0, piston_removed=...)`.

Instead:

- Strehl comes from KILO's three-term AO residual total:
  `sigma2_fit + sigma2_temporal + sigma2_noise`.
- Coupled-mode piston comes from KILO's Bessel-filtered AO residual terms via
  `ao_piston_phase_components`.
- `nullsim.physics.kolmogorov.phase_variance(..., piston_removed=False)`
  remains the correct raw uncorrected full-aperture default. Use
  `piston_removed=True` only for a deliberately modeled piston-removed raw
  Kolmogorov case, not for the KILO AO-residual stage.

For the OPD budget vocabulary, use the SPEC key `atmos_piston` for the
coupled-mode atmosphere piston component. Keep the per-term KILO names
`temporal`, `fitting`, and `wfs_noise` in diagnostics or the wavefront
component payload.

## Non-Obvious Source Invariants

- Airmass is load-bearing. KILO degrades `r0` as `airmass**(-0.6)`, which
  increases fitting and temporal variance at airmass 1.5. A KILO comment says
  pre-2026-05-05 zenith-only AO budgets understated on-sky degradation.
- Greenwood frequency is `0.426 * v_wind_ms / r0_m`.
- The AO servo bandwidth heuristic is `ao_loop_speed_hz / 10.0`.
- If `telescope.pwfs` exists, KILO dispatches to `kilo.pwfs` and interprets
  `guide_star_mag` as R band. It then rescales WFS phase noise from the WFS
  wavelength to the science wavelength by `(lambda_wfs/lambda_science)**2`.
- If `telescope.pwfs is None`, KILO uses the historical fallback
  `0.01 * 10**(0.4 * (guide_star_mag - 8.0)) * (3000 / n_actuators)`.
  The pinning tests use this fallback so they are independent of PWFS details.
- Piston filtering uses different spatial PSD exponents:
  `-11/3` for fitting above the DM cutoff, `-5/3` for temporal below the
  cutoff, and `0` for WFS noise below the cutoff.
- If no sub-aperture diameter is passed to KILO's piston calculation, the
  fallback fractions are fixed at fitting 0.03, temporal 1.0, WFS noise 1.0.
  The pinning tests pass `d_sub_m=4.0` and therefore pin the integrated
  Bessel-filter fractions.
- Scintillation is anchored at H band, 5 m sub-aperture, zenith; the default
  KILO anchor is `0.041`. KILO scales as `lambda^(-7/12)`,
  `d_sub^(-7/6)`, and `airmass^1.5`, clipping at `0.5` for sub-Fresnel
  apertures.
- No explicit `FIXME` or `TODO` markers were present in the inspected
  atmosphere files. The important inline notes are historical comments about
  airmass, WFS wavelength rescaling, small-aperture scintillation saturation,
  and the jet-stream scintillation timescale used for amplitude-servo sizing.

## Numeric Pinning Anchor

The xfail-strict stage tests pin a one-aperture Maunakea-like fallback AO case:

- `r0_500nm_m = 0.16`
- `tau0_500nm_s = 2.83e-3`
- `theta0_500nm_arcsec = 1.19`
- `v_wind_ms = 10.0`
- `wavelength_um = 1.65`
- `diameter_m = d_sub_m = 4.0`
- `ao_actuators = 1024`, so `d_act = 0.125 m`
- `ao_loop_speed_hz = 2000`, so `f_servo = 200 Hz`
- `guide_star_mag = 8.0`
- `scintillation_index_h_5m_zenith = 0.041`
- `scintillation_height_km = 10.0`

KILO outputs at zenith:

- `r0_scaled_m = 0.6704041754203222`
- `tau0_scaled_s = 0.011857773852746948`
- `greenwood_hz = 6.354375697808139`
- AO residual rad^2: fitting `0.017039108517852841`, temporal
  `0.0031871670108394854`, noise `0.029296875`, total
  `0.049523150528692328`
- Strehl `0.9516831259139219`
- Piston filter fractions: temporal `0.30598354006453549`, fitting
  `3.4062331548558047e-06`, WFS noise `0.0015632911822947319`
- Piston phase variance `0.0010210782302862232 rad^2`
- Piston OPD RMS `8.3913837866849246 nm`
- Open-loop scintillation amplitude jitter `0.05319190429725594`

At airmass 1.5, with all else identical, KILO outputs:

- AO residual rad^2: fitting `0.025558662776779262`, temporal
  `0.0047807505162592284`, noise `0.029296875`, total
  `0.059636288293038492`
- Strehl `0.94210712666909557`
- Piston phase variance `0.0015087175722511893 rad^2`
- Piston OPD RMS `10.20018503724644 nm`
- Open-loop scintillation amplitude jitter `0.097719767981424663`
