---
date: 2026-05-17
---

### Tier 4 — Config mismatches + realized-null MC physics

Two config slices and two MC-physics ports.

#### 4.1 Integration time 1 h → 4 h

- `examples/kilo_keck.toml`: `[observation].integration_time_s` and
  `[detector.mkid].integration_time_s` bumped from 3600.0 to 14400.0
  to match the KILO paperv3 headline figures (Y/J/H null-depth tables
  and contrast-curve panels are at 4 h on-source).

#### 4.2 MKID per-pixel cap + dark-pixel count

- `mkid_max_count_rate_hz` dropped from `5.0e15` (unit-test override
  that effectively disabled the cap) to KILO's physical 50 kHz.
- `n_pixels_dark` raised from 100 to 360
  (`n_dark_ports × n_wavelength_bins = 18 × 20`), so each (port, bin)
  MKID is its own pixel — KILO's per-bin readout convention.

#### 4.3 DAC bit depth 20 → 16

- `examples/kilo_keck.toml`: `[chip.kernel_mzi_mesh].dac_bits` dropped
  from 20 to 16. KILO's paperv3 chip-diagnostics figure uses 16-bit
  DACs; 20 bits is ~100× tighter phase quantization than KILO and
  artificially lowers the quantization-floor on the null depth.

#### 4.4 Fiber phase noise in realized-null MC

- `realized_null_depth_mc` gains
  `fiber_phase_noise_rms_rad: float = 0.0` — IID per-mode-per-
  realization residual fiber phase noise (KILO
  `FiberConfig.phase_noise_rms_rad` convention; typical 0.01 rad
  for SMF28 + thermal drift). Added in quadrature to the AO-derived
  phase inside the MC loop.
- `DetectionCurve.Params.fiber_phase_noise_rms_rad` plumbs it through;
  `examples/kilo_keck.toml` enables it with 0.01 rad.

#### 4.5 Per-λ CV and independent per-λ OPD draws

- The MC now draws independent AO-piston OPD perturbations PER
  wavelength bin (pre-fix the draws were reused across wavelengths,
  artificially correlating samples and underestimating CV).
- `null_instability` is now the **average of per-wavelength CVs**
  (KILO `natural_design.py:1100-1106` convention) rather than the
  pooled `std/mean` of `n_realizations × n_wl` samples. The pooled
  version mixed band-to-band mean drift into the CV; the per-λ-then-
  average version isolates per-realization fluctuation.

#### Regression tests

- `test_fiber_phase_noise_increases_null_floor` — pins that adding
  IID fiber phase noise to a perfect-null chip strictly raises the
  realized null floor.
- `test_fiber_phase_noise_rejects_negative_value` — pins the input
  validation.
- `test_per_wavelength_cv_average_does_not_pool_samples` — pins the
  per-λ-then-averaged CV against the average of single-λ runs.

645 non-parity tests pass; 40 parity (1 pre-existing kilo_maunakea
`null_depth_calibrated` failure unrelated to Tier 1-4).
