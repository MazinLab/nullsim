### Fixed

- `ao.pyramid_wfs` publishes `subaperture_strehl_ratio`, the residual
  Strehl with the subaperture piston removed,
  `exp(-(sigma2_total - sigma2_piston,sub))`, and `injection.ideal` uses
  it for SMF coupling (and republishes it as `wavefront.strehl_ratio`).
  A piston uniform over the subaperture changes the coupled phase, not
  the coupled power, and the OPD budget already books it for the
  realized null. The full-pupil `strehl_ratio` in the AO payload and the
  OPD-budget piston are unchanged. Physics review 2026-11-07, row 32.
  Mean coupling moves 0.71678 -> 0.72143 (+0.65%) for kilo_keck_6 at
  J = 4 and 0.63937 -> 0.63945 (+0.012%) for nayra_eelt at J = 3.
- `ao.pyramid_wfs` piston filter fractions integrate from `k_low` itself
  with a closed-form unfiltered denominator (`3 k_c^(1/3)` for the
  temporal term) instead of clamping the lower limit at `1e-6 m^-1`.
  The temporal piston fraction moves 0.38146 -> 0.38590 for kilo_keck_6
  (AO piston OPD 15.89 -> 15.98 nm) and 0.36028 -> 0.36618 for
  nayra_eelt. Row 33.
- `nullsim.physics.kolmogorov`: the Noll (1976) residual coefficients
  were mislabeled. 1.0299 is Delta_1, the piston-removed variance;
  0.134 is Delta_3, tip/tilt removed; the former 0.582 "piston removed"
  is Delta_2. The piston-included variance diverges for pure Kolmogorov
  turbulence and has no constant. `NOLL_TOTAL_VARIANCE_COEFFICIENT` is
  removed, `NOLL_PISTON_REMOVED_COEFFICIENT` is now 1.0299,
  `NOLL_TIP_TILT_REMOVED_COEFFICIENT = 0.134` is added, and
  `phase_variance` / `opd_rms_m` replace `piston_removed=` with
  `tip_tilt_removed=` (default piston removed). Row 69.
- `atmosphere.kolmogorov` scales the zenith r0 by `airmass^(-3/5)`, as
  `ao.pyramid_wfs` does, and records `airmass` in its payload. Row 70.
  With the Noll fix, the kilo_keck_6 atmosphere phase variance at the
  first J bin moves 114.8 -> 234.5 rad^2 (x1.770 Noll, x1.155 airmass).
  Diagnostics only: shipped configs run AO, which overwrites the Strehl
  and supersedes the atmosphere piston.
