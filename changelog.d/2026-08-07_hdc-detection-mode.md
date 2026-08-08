---
date: 2026-08-07
---
### HDC phase 2: detection-mode CCF contrast curves

Phase 1 scored a *known* planet on the spectro port. Phase 2 turns the same
matched filter around on the *dark* ports and asks the detection question: how
faint a planet would this observation still find? The answer, evaluated across
the separation anchors `sensitivity.detection_curve` already publishes, is a
CCF contrast curve — the HDC counterpart of the broadband curve
`performance.standard` publishes, and directly comparable to it.

- **New stage `performance.hdc`** (`results`, `scene` → `results`). Consumes
  `performance.standard` and the `sensitivity.detection_curve` component that
  produced it, following the recorded link rather than recency so the
  throughputs, the null, and the leak all describe one solution. Publishes
  `ccf_contrast_5sigma`, `broadband_contrast_5sigma`, their ratio
  `ccf_contrast_gain`, and the per-anchor diagnostics behind them (effective
  line count, systematic and photon variances, solver optimism and residual,
  non-detection reasons, leak-shape fallbacks). Reuses the phase 1 spectral
  machinery — templates, telluric shape and mask, analysis grid, channel
  builder — rather than restating it. Deterministic.

  `integration_time_s` and `target_snr` both default to `None` and are
  inherited from the broadband curve's own metadata
  (`contrast_curve_reference_time_s`, `target_snr`). That inheritance is the
  fairness guarantee behind `ccf_contrast_gain`: the ratio is only meaningful
  if both curves describe the same observation, and the broadband curve's
  reference time is *not* the `integration_time_for_target_snr_s` the same
  payload also publishes. An explicit override that disagrees is refused, not
  reconciled.
- **New contrast solver `ccf_contrast_at_target_snr`** in `physics/hdc.py`:
  inverts the matched-filter SNR for the faintest contrast reaching a target
  SNR. Scale safe, with one matched-filter refinement pass and the
  post-correction residual published so the caller can see how good the root
  was.
- **`detection_space` overlays HDC**: `performance.hdc` curves are drawn as
  dash-dot lines in the matching record colour, labelled with that record's
  J magnitude, integration time, and resolving power so several records stay
  distinguishable. Legend entries are derived from the curves actually drawn
  and selected by artist gid.
- **New example config `examples/test/hdc_detection_smoke.toml`**: the
  detection-mode counterpart of `hdc_ccf_smoke.toml`, four modes around an M2V
  host, seven separation anchors, dark ports dispersed at R = 3000, with the
  `detection_space` overlay. It carries the same quasi-static leak instability
  as `hdc_ccf_systematics.toml`, and for the same reason — `ccf_contrast_gain`
  is 7.5 with the instability and 0.16 without it, so a photon-limited example
  would have shown an overlay demonstrating that HDC loses. Runs in under four
  seconds.
