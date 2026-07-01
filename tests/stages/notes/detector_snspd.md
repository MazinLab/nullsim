# SNSPD Detector Stage Notes

## KILO source surveyed

Files inspected:

- `kilo/detectors.py` lines 289-309:
  `detect_snspd(power_fraction, total_photon_rate, detector)`. Computes
  `rate = power_fraction * total_photon_rate * detector.snspd_efficiency`,
  caps at `detector.snspd_max_count_rate`, returns
  `SNSPDDetection(count_rate=min(rate, cap),
  is_saturated=rate > cap)`. Notable: KILO's `detect_snspd` does NOT add
  `snspd_dark_rate` — dark counts are tracked in the noise budget
  separately. The nullsim stage adds dark counts at the count-rate level
  to give a single per-port observable that downstream null-depth
  estimators can use without a parallel dark-rate channel.
- `kilo/config.py` lines 451-459
  (`DetectorConfig` fields): `snspd_efficiency=0.85`,
  `snspd_max_count_rate=1.0e8` (100 MHz), `snspd_dark_rate=10.0` Hz,
  `snspd_timing_resolution_s=50.0e-12` (50 ps). KILO and NAYRA agree on
  all four SNSPD defaults.

## NAYRA source surveyed

- `nayra/detectors.py`
  lines 1-39: thin re-export of `kilo.detectors`. `detect_snspd` is
  re-exported unchanged.
- `nayra/config.py` lines
  320-335: NAYRA `DetectorConfig` matches KILO for the SNSPD fields:
  `snspd_efficiency=0.85`, `snspd_max_count_rate=1.0e8`,
  `snspd_dark_rate=10.0`. NAYRA does not override
  `snspd_timing_resolution_s` because it is unused in the count-rate path
  (it is a documented timing parameter for future coincidence-window
  modeling).

## Non-obvious invariants

- Fixed cap invariant: SNSPDs use a single device-level
  `snspd_max_count_rate`. No per-pixel notion (KILO models the SNSPD as a
  single-element detector per port; nullsim mirrors this and treats
  `n_bright_ports` / `n_dark_ports` as the port count, with one SNSPD per
  port).
- Dark count is per port: `snspd_dark_rate_hz` is added once per port
  after QE and saturation. The dark contribution does not itself
  saturate the detector in the modeled regime (10 Hz << 1e8 Hz cap).
- Timing-resolution is documentation only: `snspd_timing_resolution_s` is
  stored on the Params model and emitted into the results component for
  downstream coincidence-window stages, but it does not enter the count
  rate computation in this pass.

## Sign / dtype conventions

- All rates in Hz. `qe` and `snspd_efficiency` are dimensionless in (0, 1].
- `snspd_dark_rate_hz` adds to every port (bright and dark) after the
  signal cap.
- `integration_time_s` multiplies the post-cap, post-dark rate to yield
  the integrated counts diagnostic.

## Throughput and state wiring

- `state.photon_rates` (`PhotonRates(wavelength_grid_um=...,
  port_labels=..., rates_hz=...)`) is populated with the per-port,
  per-wavelength rate array of shape `(n_ports, n_wavelengths)`. Dark
  counts are distributed evenly across wavelength bins (uniform spectral
  density model — SNSPD dark counts are broadband and not
  energy-resolved).
- `state.results.components['detector.snspd']` carries `null_depth_raw`,
  `port_rates` (1D float64 over ports, bright-first), `detector_qe`,
  `snspd_dark_rate_hz`, `snspd_timing_resolution_s`,
  `integration_time_s`, and `source_model`.

## Open questions deferred

- Coincidence-window / timing-jitter modeling using
  `snspd_timing_resolution_s` is deferred.
- Polarization-aware dual-arm SNSPD pairs are deferred.
- Stray-light, blackbody-fluctuation, and cosmic-ray backgrounds are
  deferred.
