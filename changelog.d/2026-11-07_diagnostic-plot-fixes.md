---
date: 2026-11-07
---
### Fixed: diagnostic plot physics (review 2026-11-07 rows 60 to 66, Issue 25)

Figure-only fixes. No stage result, cache key, or published scalar changes.

- **`chip_diagnostics`, `chip_diagnostics_characterization`**
  - The DAC sweep starts from `optimal_chip_params_unquantized` when the
    optimizer publishes it. Otherwise its reference line is labeled
    "published (DAC quantized)", because the published params already carry
    the configured DAC quantization (row 60).
  - The physical null, port power, clean input, and DAC panels sum power
    incoherently over polarization (PBS) arms. Before, they read arm A only
    (row 61).
  - Without a deployed stack, the chip input at each bin is recovered with that
    bin's own rebuilt `U(λ)`, not the design-wavelength unitary. The port
    power and unitary panels now show `U` at the grid bin nearest the design
    wavelength (row 62).
  - The optimizer payload is paired with the parameters of the stage that
    published it, matched by component name. Before, it took the first
    scipy or torch optimizer in the config (Issue 25).
- **`cophasing_unitary`**: the displayed ABCD analysis matrix puts
  `e^{-iθ}` on the second mode of each pair. This matches the measurement
  model `1 + V cos(φ_a − φ_b + θ)`. Rows B and D of the phase panel were
  swapped (row 63).
- **`debug_report`**: "Cumulative pre-detector transmission" divides out the
  `detector.*_qe` components. It was low by the QE: ×0.7 for KILO and NAYRA,
  ×0.9 for HWO (row 64).
- **`uv_coverage_grid`**: only telescope origins rotate with hour angle.
  Intra-telescope sub-aperture offsets stay rigid in the tracking pupil, the
  same convention the science stages use. Intra-telescope baselines are now
  fixed points instead of foreshortened tracks (row 65). New physics helper:
  `nullsim.physics.uv_coverage.baselines_to_uv_with_origins`.
- **`hdc_ccf`**
  - The SNR² share panel is `u_i² v_i / Σ u² v` with the published
    `noise_weights_per_channel` `u = Hᵀw`, not the diagonal `w_i² v_i`.
    Payloads without `u` get no share curve (row 66).
  - Every HDC payload in a record is considered. Rejected ones are skipped,
    and each accepted one is plotted: `hdc_ccf.<fmt>` for the first, then
    `hdc_ccf_<id>.<fmt>`. Before, a rejected first payload hid the rest of
    the record (Issue 25).
