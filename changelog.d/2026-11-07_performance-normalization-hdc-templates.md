---
date: 2026-11-07
---
### Fixed: performance rates normalized to the physical stellar rate (review 2026-11-07 Issue 9)

`performance.standard` built its planet and leakage rates from the sum of the
detector's saturation-capped `port_rates`. When bright ports saturate that
total understates the star, so both rates were low by the uncapped/capped
ratio. Every rate is now a fraction of the physical stellar rate summed from
`uncapped_port_rates`. Saturation is treated as a per-port detector effect:
when the dark ports saturate, the planet and leakage both carry the dark-port
recorded/physical ratio (`dark_detected_fraction`, with a `RuntimeWarning`).
Bright-port saturation no longer changes dark-port science.

`stellar_rate_hz_total` and `bright_rate_hz` are now physical rates; the
readout total moves to `stellar_rate_hz_total_capped`. `performance.hdc`
applies `dark_detected_fraction` to its planet reference, and the
`stellar_normalization_mismatch_note` it used to emit is replaced by
`dark_saturation_note`, because the two curves now share one normalization.

`examples/test/hdc_detection_smoke.toml` (J = 5, bright ports clipped 3.80x):

- systematics knobs off: planet 4199 → 15972 Hz, SNR 241.4 → 470.8, 10 mas
  broadband contrast 2.43e-6 → 1.24e-6, `ccf_contrast_gain` 0.626 → 0.1645.
- shipped knobs: SNR 3.8700 → 3.8703, 10 mas CCF contrast 5.38e-6 → 1.63e-5,
  `ccf_contrast_gain` 28.199 → 9.327. The CCF curve was 2.97x optimistic,
  plus 1.8% from the telluric edge fix below.

### Changed: systematic amplitude interface for `null_fluctuation_rms` (Issue 19)

When sensitivity publishes `null_fluctuation_rms` (σ_L), the systematic
amplitude is hypot(σ_L·S / `nsc_systematic_reduction`,
`residual_speckle_stability_factor`·L), with S the detected stellar rate.
Without it the stage keeps the legacy `null_instability`·L. The contrast curve
evaluates the amplitude per separation and publishes
`contrast_curve_eps_systematic`; `performance.hdc` uses it per anchor.
`eps_systematic_used` still satisfies eps × `leakage_rate_hz` = amplitude.

### Fixed: calibration residual sets only the systematic (review row 71)

Without a sensitivity stage, the calibration-subtracted residual
max(raw − floor, 0) replaced the detected leakage, so subtraction also removed
Poisson noise. The detected dark rate now sets `leakage_rate_hz` and the
Poisson term; the residual, bounded below by the floor, sets the new
`systematic_leakage_rate_hz`.

### Fixed: newest component wins in `performance.standard` (Issue 25, partial)

The detector, calibration, sensitivity, and fringe-tracking selectors took the
first matching component while the background selector took the last. All now
take the newest. The cache-key side of Issue 25 is not addressed here.

### Fixed: HDC template broadening (review rows 50, 51)

`convolve_to_resolving_power` gains `outside="edge"`. Transmission and template
shapes are continued at their edge values instead of zero padded, so a flat
T = 0.9 no longer drops to about 0.77 in the edge channels and gets masked.
In `hdc_detection_smoke` this unmasks channel 0 (T 0.798 → 0.954). Every
template is now degraded to the lowest content resolving power before the
products are formed, so when the instrument R reaches that value the sharper
templates are no longer left unbroadened. The shipped templates all carry
lsf R = 30000, so `hdc_ccf_smoke` and `hdc_ccf_systematics` are unchanged
(CCF SNR 70.2177 and 31.2859).
