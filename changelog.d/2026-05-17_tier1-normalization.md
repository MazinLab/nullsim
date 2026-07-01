---
date: 2026-05-17
---

### Tier 1 — Stellar/contrast normalization fixes (KILO Keck-only parity blockers)

Three tightly coupled bugs in the detector → contrast-curve → sensitivity
path were producing inconsistent absolute photon rates and silently dropping
the dominant noise term on the Keck-only target. Fixed together because they
partially cancel; fixing any one alone moves the headline numbers in the
wrong direction.

#### 1.1 Detector stellar-rate over-normalization

- New `Geometry.collecting_area_m2` field: total light-collecting area
  summed over all apertures. `telescope.fixed_aperture` publishes
  `collecting_area_m2 = n_apertures × area_m2`.
- `detector.mkid`, `detector.snspd`, and `detector.ideal_counter` now use
  the total collecting area (with a fallback chain
  `collecting_area_m2 → n_apertures × area_m2 → area_m2`).
  `_port_rate_hz_per_wavelength` switches from
  `power_fraction × area_m2 × stellar_rate` to
  `power_per_mode × (collecting_area_m2 / n_modes) × stellar_rate`,
  matching KILO's `effective_area = Σ(coupling × sub_areas)` convention
  (`kilo/studies/natural_design.py`).
- Pre-fix: for N_apertures × N_subs/aperture unit-amplitude inputs through
  a unitary chip, total port rate was `n_modes × A_per_aperture × stellar`,
  i.e. ~12× too high for Keck-only (M=24 modes on 2 telescopes).

#### 1.2 Contrast-curve denominator

- `StandardPerformance.apply` no longer divides `bright_rate_hz` by
  `n_dark_ports` before passing it to `_contrast_curve`. KILO
  `contrast_curve` (`kilo/performance.py` L968-1008) uses the aggregate
  stellar photon rate; nullsim's planet/leakage rates are already
  aggregated over dark ports, so the per-port stellar division reported
  contrasts `n_dark_ports` × tighter than KILO (~18× on Keck-only with
  N_dark = 18).

#### 1.3 Decouple `null_instability` from `effective_null_override`

- `sensitivity.detection_curve` always propagates the MC
  `null_instability` downstream regardless of whether
  `effective_null_override` is set. Pre-fix, the override silently zeroed
  the systematic-noise term unless
  `effective_systematic_bandwidth_hz_override` was also set, dropping the
  dominant speckle noise on Keck-only at every J-mag. The override now
  acts strictly as a null-depth replacement; f_eff handling remains the
  responsibility of the bandwidth override / performance stage default
  (Tier 2 will land KILO's blended f_eff).

#### Regression tests

- `test_mkid_multi_aperture_uses_collecting_area`,
  `test_mkid_collecting_area_falls_back_to_n_apertures_times_area`
  pin total port-rate = `collecting_area × Σ_wl stellar_rate × qe`.
- `test_performance_contrast_curve_uses_aggregate_stellar_rate` pins the
  photon-noise-limited analytic contrast against the KILO formula.
- `test_detection_curve_override_preserves_mc_null_instability` pins that
  setting `effective_null_override` does not zero `null_instability`.
- `EXPECTED_EMPTY_STATE_DIGEST_HEX` updated to reflect the new
  `Geometry.collecting_area_m2` field.
