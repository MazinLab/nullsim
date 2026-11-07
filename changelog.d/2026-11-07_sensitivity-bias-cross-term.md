---
date: 2026-11-07
---
### Fixed: the setpoint bias cross term in the null fluctuation and f_eff

- **Cross term (Issue 19 follow-up).** The setpoint residual and dither are
  now modeled as an unknown static phase offset, i.i.d. across chip input
  modes. Its linear dark field beats with each MC draw's static-subtracted
  dark field, giving `cross = 4 L_b S / D` (`bias_cross_moments` and
  `bias_cross_variance` in `physics/realized_null.py`).
  - This is exact to linear order and has no denominator that can go to
    zero. Phase and amplitude couple as the output rows dictate.
  - Against a bias-averaged direct MC it agrees to 4e-4 on three toys: a
    Bracewell pair, a 0.02 rad static phase error, and a three-input DFT row
    with amplitude noise.
  - It replaces two earlier forms. `2 CV_MC^2 L_dyn L_b` was 6.5x low and
    4.1x high in RMS on two-input toys. The phase-only
    `2 (sigma_phase^2 / L_phase,dyn) L_b` was 5.6x high in RMS with a static
    phase error and 5.7x low on a complex row with amplitude noise.
  - `characterization.spectro` uses the same primitive on the retuned
    spectro rows.
  - New payload key: `null_fluctuation_bias_cross_variance`.
- **f_eff (row 27 follow-up).** The cross term is split by leg, each part
  computed from that leg's own MC draws and added to that leg's variance. The
  first version of this fix put a phase-only cross term on the piston leg.
  On the reviewer's toy (2000 Hz piston, 11.1 Hz amplitude) that already
  moved f_eff from 12.5 Hz to 73.9 Hz, removing a 5.9x overstatement of the
  averaged systematic variance.
- `sensitivity.detection_curve` now reads the newest `fringe_tracking.*`
  component's published bandwidth, consistent with the residual lookup.
