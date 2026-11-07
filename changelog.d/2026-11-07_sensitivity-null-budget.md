---
date: 2026-11-07
---
### Fixed: sensitivity null budget (review 2026-11-07 Issues 17, 18, 19; rows 27, 30, 34, 36-39)

- **Disk floor (Issue 17, rows 38, 30).** The realized null already holds the
  on-axis leakage T(0), so `disk_null_floor` is now the disk excess
  `<T(r) - T(0)>`. T is evaluated on 16 Gauss-Legendre nodes inside the
  stellar radius instead of the separation grid with an invented (0, 0)
  anchor. The separation grid no longer has to reach the stellar radius.
  Pointing-jitter smoothing now integrates on a fine grid with trapezoid
  weights, so a log-spaced grid no longer biases it toward small separations.
- **Band statistics (Issue 18, rows 34, 39).** Every band average uses the
  detected stellar photon rate per bin: realized null, planet throughput, and
  residual starlight. The MC CV is now the CV of the photon-weighted band
  leakage of paired draws, with the variance taken within each azimuth.
- **Null fluctuation (Issue 19).** New `null_fluctuation_rms` (null-depth
  units) is the MC RMS plus the cross term between the trim bias and the
  random piston. The disk floor adds no variance. `null_instability` is that
  RMS over the effective null, and `null_instability_mc` is the MC band CV.
- **f_eff (row 27).** `effective_systematic_bandwidth_hz` now takes the
  leakage variances of piston-only and amplitude-only MC reruns. The
  `n_modes` and input-sigma arguments are gone (API change).
- **Planet throughput (rows 36, 37).** It is averaged over the realized-null
  piston covariance. Position angles are sampled at least
  `ceil(4 pi B_max theta / lambda_min)` times, up to 4096.
  `n_pa_samples` is now the minimum.

Measured on cell 0 with the cache off, before → after:

- `examples/test/kilo_maunakea.toml`: disk floor 4.713e-2 → 4.534e-2;
  `null_instability` 0.938 → 0.0012 (MC band CV 0.191,
  `null_fluctuation_rms` 5.5e-5); f_eff 1978 → 1933 Hz; planet throughput at
  10 mas 0.1968 → 0.2097; SNR 1.6e-8 → 7.4e-7.
- `examples/test/hdc_detection_smoke.toml`: disk floor 1.741e-3 → 7.30e-5
  (its grid starts at 5 mas against a 0.4 mas stellar radius); effective null
  2.154e-3 → 4.857e-4; SNR 3.87 → 9.10.
