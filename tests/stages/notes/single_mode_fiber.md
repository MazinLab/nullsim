# SingleModeFiber Stage Notes

## KILO source surveyed

The task path `kilo/tests/...` is stale
for this checkout. The KILO package root is
`kilo`; tests live under
`kilo/tests`.

Files inspected:

- `kilo/injection.py` lines 1-223.
- `kilo/injection.py` lines 20-29:
  coupling constants.
- `kilo/injection.py` lines 32-73:
  `optimal_coupling_efficiency` and `coupling_efficiency`.
- `kilo/injection.py` lines 76-129:
  angular coupling and half-angle helpers.
- `kilo/injection.py` lines 132-191:
  per-sub-aperture coupling expansion.
- `kilo/injection.py` lines 194-223:
  `stellar_input_field`.
- `kilo/tests/test_injection.py` lines 24-105:
  coupling and stellar-field regression tests.

## NAYRA source surveyed

Files inspected:

- `nayra/injection.py`
  lines 1-114.
- `nayra/injection.py`
  lines 39-78: single-pupil coupling helpers requested by the work package.
- `nayra/injection.py`
  lines 81-98: single-pupil coupling broadcast.
- `nayra/injection.py`
  lines 101-114: `stellar_input_field`.
- `nayra/tests/test_injection.py`
  lines 22-99: matching coupling and stellar-field tests.

## Non-obvious invariants

- Coupling power invariant: `coupling_efficiency` is a power efficiency:
  `eta = strehl * eta_opt * (1 - FRESNEL_LOSS)` by default. KILO and NAYRA
  both use `ETA_OPT_UNOBSTRUCTED = 0.81`, `FRESNEL_LOSS = 0.035`, and
  `eta_opt = 0.81 * (1 - central_obstruction**2)**2`.
- Fresnel default invariant: the default Fresnel-inclusive perfect-clear coupling is `0.78165`; with
  `strehl = 0.5` it is `0.39082500000000003`. The no-Fresnel pin is
  exactly `0.81` in the reference tests.
- Clear-sub-aperture invariant: sub-apertures are assumed to sit inside clear pupil. KILO's
  `compute_coupling_per_subaperture` indexes the Strehl by
  `sub.telescope_index` but passes `central_obstruction=0.0` for every
  sub-aperture.
- Shape-broadcast invariant: KILO maps Strehl shape `(n_tel,)` to coupling shape `(n_sub,)` and Strehl
  shape `(n_wavelength, n_tel)` to `(n_wavelength, n_sub)`. NAYRA has one
  pupil, so scalar Strehl broadcasts to `(n_sub,)` and 1D wavelength Strehl
  broadcasts to `(n_wavelength, n_sub)`.
- Amplitude-square-root invariant: the stellar field is an amplitude, not a power:
  `E_k = sqrt(coupling_per_sub[k] * area_k)`. Therefore
  `sum(abs(E_k)**2) = sum(coupling_per_sub[k] * area_k)`. The KILO/NAYRA
  tests pin `coupling=1.0` versus `0.25` as a `4.0` power ratio.
- Angular-coupling invariant: `angular_coupling_factor` returns a power factor. Any stage applying it to
  amplitudes must use `sqrt(angular_coupling_factor(...))`.
- Half-angle invariant: the angular coupling half-angle is `wavelength_m / (pi * w)` with
  `w = 0.6436 * sub_diameter_m / 2`. The H-band, 3.6 m reference is
  `4.533624337090452e-07 rad = 93.51280238899622 mas`.
- Throughput-ownership invariant: KILO optionally folds pre-chip transmission into coupling when
  `wavelength_m` is supplied to `compute_coupling_per_subaperture`; this is a
  throughput ownership question for the nullsim stage, not part of the pure
  coupling primitive.

## Sign / dtype conventions

- Wavelengths are meters, separations are radians, position angles are
  radians, and sub-aperture areas are square meters.
- On-axis stellar fields have zero phase. The returned array is complex dtype
  even though the imaginary part is zero to `1e-15` in the reference tests.
- Coupling arrays are real power efficiencies. The field layer owns the
  conversion from power efficiency to complex amplitude by square root.
- KILO and NAYRA use the approximate `0.81` overlap constant in injection.
  `nullsim.physics.ruilier_cassaing` exposes a more exact unobstructed limit;
  parity tests should deliberately decide which constant a stage is pinning.

## Edge cases

- Zero Strehl gives zero coupling. A full obstruction ratio of `1.0` gives
  zero optimal coupling in KILO/NAYRA.
- KILO/NAYRA tests only assert bounded outputs for Strehl values in `[0, 1]`;
  the reference functions do not validate out-of-range Strehl or obstruction.
- Scalar and wavelength-vector coupling shapes must be preserved. In NAYRA the
  broadcasted wavelength coupling is copied so later mutation cannot alias the
  scalar source.
- On-axis planet and on-axis star fields match in the reference tests; off-axis
  planet power is reduced by the angular coupling power factor.

## Open questions deferred to Pass 3

- Decide whether the stage exposes a KILO-parity mode using `0.81` and Fresnel
  by default, or wraps `nullsim.physics.ruilier_cassaing` directly and pins a
  tolerance against the exact `0.8145287551781475` primitive.
- Decide whether pre-chip atmosphere/mirror/AO-bench transmission belongs in
  the injection stage or in a separate throughput stage.
- Dual-polarization injection and PBS/HWP routing are out of scope for this
  single-mode-fiber pass and should be handled with the Pass 3 polarization
  work.
