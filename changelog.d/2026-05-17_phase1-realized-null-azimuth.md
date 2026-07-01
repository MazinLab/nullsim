---
date: 2026-05-17
---
### Phase 1 — realized-null MC: azimuth marginalization + transport + dark-only stats

Per the KILO parity audit (2026-05-17), nullsim's realized-null MC drew
iid per-mode OPD/fiber-phase samples at a single azimuth. KILO
`azimuth_averaged_realized_null_depth` (`transport.py` L1739) averages
over 6 azimuths × 200 realizations each, applies the per-azimuth
delay-line transmission, and reports dark-only statistics. Without
azimuth marginalization, nullsim was systematically underestimating
transport-loss imbalance that an Earth-rotation track sweeps through.

- `nullsim/physics/realized_null.py` now accepts `stellar_field` of
  shape `(M,)` (back-compat) or `(n_az, M)` (pre-transported per-azimuth
  fields). Per-wavelength CV is computed from the pooled
  `(n_azimuths × n_realizations_per_azimuth)` sample at that wavelength,
  then averaged across wavelengths. `RealizedNullResult` exposes a
  `null_depth_dark` alias (the paperv3 reporting name) and reports
  `n_azimuths` / `n_realizations_per_azimuth` for traceability.
- `nullsim/stages/delay_lines/geometric.per_aperture_transmission()`
  exposes the geometry → loss → transmission pipeline for reuse by the
  realized-null MC at arbitrary `(altitude_deg, azimuth_deg)`.
- `nullsim/stages/sensitivity/detection_curve.py` reads the upstream
  delay-line diagnostics, rebuilds per-aperture transmissions at 6
  uniformly-spaced azimuths, applies `sqrt(transmission)` to the
  chip-input stellar field per azimuth, and passes the `(n_az, M)`
  stack to `realized_null_depth_mc`. New params `n_azimuths` (default 6)
  and `n_realizations_per_azimuth` (default `None` → distributes
  `n_mc_realizations` across azimuths for back-compat).
- `examples/kilo_keck.toml` pins `n_azimuths = 6` and
  `n_realizations_per_azimuth = 200` to match KILO paperv3 defaults.
- Three new regression tests in `tests/physics/test_realized_null.py`
  pin (a) pooled sample count, (b) per-azimuth transport imbalance
  inflating the null, (c) the validation that one of
  `n_realizations` / `n_realizations_per_azimuth` is required.

`null_depth_dark` xfail in `test_kilo_keck_paperv3_parity_metric` is
left in place pending Phase 4 (planet/optimizer obscuration); azimuth
marginalization on its own is necessary but not sufficient for full
parity on that metric.
