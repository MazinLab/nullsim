Fixed several ways the HDC cross-correlation could report a number it had not
earned, and made the radial-velocity scan say when its peak is meaningless.

- **A flat spectral island is no longer a detection.** The continuum filter
  annihilates a constant only to a few ulp, and every quantity built on it is
  scale-free in the template, so the residue was amplified into an ordinary
  SNR: three flat channels at 1.03 Hz against a 100 Hz leak reported
  `ccf_snr = 13.1` at 3600 s. Both `matched_filter_ccf` and
  `ccf_contrast_at_target_snr` now zero any island whose filtered template is
  within `1024 eps` of that island's own template magnitude.
- **The RV scan no longer erases real structure on a faint island.** The round-3
  guard was a global absolute threshold, which zeroed a genuine `1 Hz` island
  sitting beside a `1e12 Hz` one. The filter is now built from per-island pair
  differences, which are exactly zero on a flat island and exactly the real
  structure on a faint one, so no threshold is needed at all.
- **`ccf_contrast_at_target_snr` handles an island that mixes noiseless and
  ordinary channels.** Island membership is decided on "has variance or has
  template signal" and never on `v0`, so a zero-background channel between two
  ordinary ones stays in their island instead of becoming a singleton with
  `H = 0`. That case used to raise; it now returns the limit of the same solve
  as the background goes to zero.
- **Diagnostics and roots survive extreme but representable inputs.** The
  reported weight, `A`, `B2`, `C2` and `V_sys` are assembled with exponent
  bookkeeping rather than by materializing the weight scale, and the closing
  contrast product is too. Previously a `1e-300` planet against a subnormal leak
  reported `inf`/`NaN` for every field, and a subnormal target SNR returned a
  spurious "not representable" error on a root of `3.2e-57`. The RV scan
  likewise no longer returns `NaN` for a variance span wider than float64's
  exponent range.

New payload keys on `characterization.hdc_ccf`: `rv_leak_contamination` and
`rv_leak_contamination_effective` (how much of the RV score a leak could
contribute, relative to the planet), `rv_scan_localization_grade` /
`rv_scan_localization_valid` (graded at stated conventional cuts), and
`ccf_island_channel_counts` / `ccf_island_self_subtraction_factors` (per-island
retention, which the global `self_subtraction_factor` averages over a
factor-of-8 spread). The `hdc_ccf` plot annotates its RV panel when the
localization grade is not `reliable`. Rejected payloads now publish every array
key an accepted payload does, so the schema no longer depends on whether the
upstream optimizer converged.

No shipped configuration's headline numbers moved: `hdc_ccf_smoke`,
`hdc_ccf_systematics` and `hdc_detection_smoke` are bit-identical before and
after. The fixes are confined to degenerate and extreme-magnitude inputs.
