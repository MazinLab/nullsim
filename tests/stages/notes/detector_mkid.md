# MKID Detector Stage Notes

## KILO source surveyed

Files inspected:

- `kilo/detectors.py` lines 256-287:
  `detect_mkid(power_fraction, total_photon_rate, detector, n_pixels=1)`.
  Computes `total_port_rate = power_fraction * total_photon_rate *
  detector.mkid_efficiency`, then splits across `n_pixels` MKIDs that each
  saturate independently at `detector.mkid_max_count_rate`. Returns
  `MKIDDetection(count_rate=min(per_pixel, cap) * n_pixels,
  is_saturated=per_pixel_rate > cap)`.
- `kilo/config.py` lines 18-32:
  `_default_mkid_max_count_rate()` reads `KILO_MKID_MAX_COUNT_RATE` env var
  or returns `50_000.0` (50 kHz default for linear range; the array can go
  higher with per-pixel dead-time correction but SNR degrades above this).
- `kilo/config.py` lines 451-459
  (`DetectorConfig` fields): `mkid_max_count_rate` via factory,
  `mkid_efficiency=0.7`, `snspd_efficiency=0.85`,
  `snspd_max_count_rate=1.0e8`, `snspd_dark_rate=10.0`,
  `snspd_timing_resolution_s=50.0e-12`. No `mkid_dark_rate` field exists —
  MKID dark counts are not modeled at the detector layer.
- `kilo/detectors.py` lines 79-127:
  `stellar_photon_rate(mag, wavelength_m, bandwidth_m, collecting_area_m2,
  band=None)`. Reference for the `magnitude → photons/s` conversion used by
  the detector stage; nullsim wraps `nullsim.physics.photometry`
  (J-band Vega zero point from Cohen+2003) rather than the KILO Vega
  table.

## NAYRA source surveyed

- `nayra/detectors.py`
  lines 1-39: thin re-export of `kilo.detectors`; `detect_mkid`,
  `detect_snspd`, `MKIDDetection`, `SNSPDDetection` are all re-exported
  unchanged.
- `nayra/config.py` lines
  320-335: NAYRA `DetectorConfig` overrides
  `mkid_max_count_rate=1.0e5` (100 kHz, "NAYRA E-ELT bump (was 5_000); see
  `_kilo_adapter.NAYRA_MKID_MAX_COUNT_RATE`"). KILO default is 50 kHz;
  NAYRA bumps it to 100 kHz. The nullsim stage exposes
  `mkid_max_count_rate_hz` so each config (Mauna Kea KILO vs E-ELT NAYRA)
  can pin its own value.

## Non-obvious invariants

- Per-pixel saturation invariant: `n_pixels_bright` and `n_pixels_dark` set
  how many MKIDs read each port type (Params validates `ge=1`; default 1
  is backward-compatible with the original single-pixel-per-port stage).
  The per-port photon rate is divided across the port's pixels
  (`rate_per_pixel = rate_per_port / n_pixels`), the saturation cap
  applies per-pixel (`rate_per_pixel = min(rate_per_pixel, cap)`), and the
  result is summed back to a port rate (equivalent to
  `min(rate_per_port, n_pixels * cap)`). A 1000-pixel bright port has an
  effective ceiling of `1000 * mkid_max_count_rate_hz` — but each pixel's
  saturation is tracked independently and `saturated_per_port` fires if any
  pixel exceeds the cap. KILO-class configs need ~1000 pixels per bright
  port to handle J=8 host stars; dark ports need ~100. See
  `examples/test/kilo_maunakea.toml` and `examples/nayra_eelt.toml`.
- No dark counts invariant: KILO's MKID model has no `mkid_dark_rate`.
  Dark counts are modeled only for SNSPDs (`snspd_dark_rate`). The nullsim
  MKID stage matches: no dark-count addition.
- Efficiency placement invariant: KILO applies `mkid_efficiency` once,
  multiplied into `total_port_rate` before saturation. The nullsim stage
  applies `qe` to the per-wavelength rate before summing and capping.
- Power-fraction lineage: KILO `detect_mkid` takes a scalar `power_fraction`
  derived from chip output; nullsim derives per-port,
  per-wavelength fractions from `|field|²` summed over polarizations and
  normalized to total port power.

## Sign / dtype conventions

- All rates in Hz (counts/s). `qe` is dimensionless in (0, 1].
- `integration_time_s` is seconds and multiplies rate to yield total
  counts (the integrated-counts diagnostic in `state.results.components`).
- `port_rates` written to `state.results.components` is float64 array
  of shape `(n_bright_ports + n_dark_ports,)` ordered bright-first.

## Throughput and state wiring

- `state.photon_rates` schema (`PhotonRates(wavelength_grid_um=...,
  port_labels=..., rates_hz=...)`) is populated with the per-port,
  per-wavelength rate array of shape `(n_ports, n_wavelengths)`. The
  per-port total (summed over wavelength, capped, integrated for diagnostics)
  goes into `state.results.components['detector.mkid']` as `port_rates`
  along with `null_depth_raw`, `detector_qe`, `integration_time_s`, and
  `source_model`.
- No throughput component is added: detectors consume throughput but do
  not modify the optical-transmission spectrum.

## Open questions deferred

- Polarization-aware multi-pixel arrays (dual-pol KILOPIC chips writing to
  two MKID arrays) are deferred — the stage is scalar over polarization
  for now and sums `|field|²` across the polarization axis.
- MKID detector spectral resolution (R≈100 typical) and energy-bin
  reassignment of photon counts is deferred to a later pass that adds an
  energy-resolved port-counts product.
- Cosmic ray and stray-light backgrounds are deferred.
