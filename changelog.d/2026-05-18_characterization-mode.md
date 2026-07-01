---
date: 2026-05-18
---
### Added characterization mode

- Added `characterization.spectro`, a known-position spectro-port
  characterization stage that retunes the chip, reuses the upstream
  `sensitivity.detection_curve` physical floors, and reports spectro
  throughput, leakage, SNR, and maximum spectral resolving power.
- Added pure physics helpers for characterization optimization, resolved-disk
  spectro leakage, and high-resolution spectrograph per-bin SNR.
- Added characterization metrics to sweep results and table outputs, plus a
  dedicated `characterization_summary` table.
- `per_bin_snr` splits sky background by `bin_fraction` (per-port,
  band-integrated semantics), matching `_compute_snr` in
  `performance.standard` so the same `n_sky_background_hz` parameter has
  one consistent meaning across broadband and per-bin SNR paths.
- `optimize_characterization` now takes `dac_bits` and quantizes the
  best continuous solution before computing returned metrics, so the
  reported throughput/null depth/disk leak correspond to the deployed
  quantized chip rather than the optimizer output.
- `spectro_leak_fraction` uses the on-axis stellar null (not the
  row-averaged optimizer-objective value) summed with the resolved-disk
  leak, avoiding double-counting when disk samples are added to the
  constraint set. The on-axis null is exposed as
  `spectro_on_axis_null_depth` in the payload.
- Single-wavelength grids require an explicit `band_bandwidth_um`; the
  prior fallback silently used the wavelength itself as the bin width.
- Added a torch analytic-Jacobian backend
  (`nullsim/physics/characterization_torch.py`) for
  `optimize_characterization`. The new `backend` parameter selects
  `"scipy"` (FD-SLSQP, no extras), `"torch"` (SLSQP with autograd
  Jacobians for both the objective and the null constraint), or
  `"auto"`. Matches KILO's `chip_torch` characterization fast-path. At
  M=24 the torch path is ~10-100× faster per gradient than 2-point FD
  and reaches a deep enough basin to reproduce KILO paperv3's
  `chip_diagnostics_characterization.pdf` numbers.
- `examples/kilo_keck.toml` characterization stage is now configured
  to track KILO's `chip_diagnostics` settings (`backend="torch"`,
  `null_floor=3e-5`, `n_restarts=3`, `max_iter=10000`). At the J=4
  fiducial cell the optimizer reproduces KILO's published
  characterization metrics to within SLSQP run-to-run variation:
  spectro throughput 0.395 (KILO 0.363), bright fraction 0.9999
  (KILO 0.9997), spectro disk leak 1.7e-5 (KILO 2.8e-5), on-axis
  spectro null 1.8e-5 (KILO 3.2e-5).
