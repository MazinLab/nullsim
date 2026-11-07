---
date: 2026-11-07
---
### Fixed: the setpoint bias cross term in the null fluctuation and f_eff

- **Cross term moments (Issue 19 follow-up).** The setpoint residual and
  dither are a static phase offset. They beat with the random phase (piston
  and fiber), not with scintillation, which sits in the orthogonal quadrature.
  The cross term is now `2 (sigma_phase^2 / L_phase,dyn) L_b`, with both
  moments from a phase-only MC leg (scintillation off).
  - The earlier `2 CV_MC^2 L_dyn L_b` used the full-MC CV, normalized by the
    total mean. On two-input toys with a 0.02 rad bias it was 6.5x low in RMS
    (phase noise) and 4.1x high (amplitude noise only).
  - The new form matches a direct biased MC to 0.2%.
  - New payload key: `null_fluctuation_bias_cross_variance`.
- **f_eff (row 27 follow-up).** The phase-driven cross term averages at the
  piston bandwidth, so it is added to the piston leg's variance in the blend.
  The reviewer's toy (2000 Hz piston, 11.1 Hz amplitude) goes from
  f_eff 12.5 Hz to 73.9 Hz, removing a 5.9x overstatement of the averaged
  systematic variance.
- `sensitivity.detection_curve` now reads the newest `fringe_tracking.*`
  component's published bandwidth, consistent with the residual lookup.
