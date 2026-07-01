# SMF28 Stage Notes

## KILO source surveyed

The task path `kilo/tests/...` is stale
for this checkout. The KILO package root is
`kilo`; tests live under
`kilo/tests`.

Files inspected:

- `kilo/transport.py` lines 1-48:
  SMF-28e constants and attenuation interpolation.
- `kilo/transport.py` lines 172-270:
  optical-loss and static-loss dataclasses.
- `kilo/transport.py` lines 275-308:
  `FiberConfig` fields and units.
- `kilo/transport.py` lines 311-370:
  `fiber_phase`.
- `kilo/transport.py` lines 373-428:
  `chromatic_dispersion_phase`.
- `kilo/transport.py` lines 434-486:
  `compute_loss_budget`.
- `kilo/tests/test_transport.py` lines 46-101:
  fiber phase and chromatic dispersion tests.
- `kilo/tests/test_transport.py` lines 349-421:
  SMF-28e attenuation and static loss tests.
- `kilo/tests/test_transport.py` lines 1274-1315:
  KILO common-mode fiber phase tests.

## NAYRA source surveyed

Files inspected:

- `nayra/transport.py`
  lines 1-20: single-bench scope and dropped KILO features.
- `nayra/transport.py`
  lines 35-53: SMF-28e table and KILO re-export.
- `nayra/transport.py`
  lines 140-215: NAYRA static loss budget.
- `nayra/transport.py`
  lines 253-325: NAYRA fiber phase and chromatic dispersion.
- `nayra/config.py`
  lines 357-388: NAYRA `FiberConfig` fields and units.
- `nayra/config.py`
  lines 390-406: NAYRA `FiberConfig` validation.
- `nayra/tests/test_transport.py`
  lines 23-95: phase, dispersion, and attenuation tests.
- `nayra/tests/test_transport.py`
  lines 131-248: loss and apply-transport tests.
- `nayra/tests/test_transport.py`
  lines 258-308: seed reproducibility tests.

## Non-obvious invariants

- KILO fiber-config invariant: KILO `FiberConfig` defaults are four equal `lengths_m=(780.0, 780.0, 780.0,
  780.0)`, `length_matching_error_m=0.001`,
  `dispersion_ps_nm_km=-40.0`, `phase_noise_rms_rad=0.01`, `n_eff=1.4504`,
  `connector_loss_db=0.6`, `delay_line_loss_db_per_m=0.034`, and no fixed
  delay-line override.
- NAYRA fiber-config invariant: NAYRA `FiberConfig` is a single-bench scalar: `length_m=30.0`,
  `length_matching_error_m=0.001`, `dispersion_ps_nm_km=-29.5`,
  `n_eff=1.4682`, `phase_noise_rms_rad=0.01`, and `connector_loss_db=0.6`.
  NAYRA validates positive length and `n_eff`, and non-negative length
  matching error, connector loss, and phase noise.
- Attenuation-table invariant: SMF-28e attenuation is interpolated in wavelength microns and returned in
  `dB/km`. Regression anchors from both KILO and NAYRA are `0.33 dB/km` at
  `1.31 um`, `0.19 dB/km` at `1.55 um`, `1.3 dB/km` at `1.0 um`, and
  `0.21666666666666679 dB/km` at `1.65 um`.
- Static-loss invariant: static fiber loss is power loss in dB:
  `fiber_loss_db = attenuation_db_km * length_km`,
  `total_loss_db = fiber_loss_db + connector_loss_db`, and
  `total_transmission = 10**(-total_loss_db / 10)`.
- Loss-shape invariant: KILO static loss is per telescope: scalar wavelength shapes are `(4,)`, and
  wavelength arrays shape as `(n_wavelength, 4)`. NAYRA static loss is one
  representative fiber: scalar wavelength gives scalar-shaped arrays, and
  wavelength arrays shape as `(n_wavelength,)`.
- Loss-ownership invariant: NAYRA's `compute_loss_budget` additionally includes Cerro Armazones
  atmospheric loss and optional bare-Al mirror-train loss. KILO's
  `compute_loss_budget` is fiber plus connectors only; KILO pre-chip
  atmosphere, mirror, and AO-bench losses are separate.
- Phase-to-OPD invariant: fiber phase converts configured RMS phase at the center wavelength into OPD:
  `sigma_opd_m = phase_noise_rms_rad * wavelength_center_m / (2*pi)`.
  Phase at each wavelength is then `2*pi*opd_noise_m / wavelength_m`.
- KILO common-mode invariant: KILO draws one OPD per telescope with `np.random.default_rng(seed)` and
  repeats it across that telescope's sub-apertures. NAYRA draws one OPD per
  sub-aperture because the single-bench metrology is per fiber.
- Dispersion-formula invariant: `chromatic_dispersion_phase` uses `c_light = 2.998e8`, converts
  `D ps/(nm km)` to SI with `* 1e-6`, then applies
  `beta2 = -lambda0**2 * D_si / (2*pi*c)` and
  `delta_omega = -2*pi*c*delta_lambda/lambda0**2`.
- Center-wavelength invariant: chromatic dispersion phase is identically zero at the center wavelength
  because `delta_lambda = 0`. This is pinned at `atol=1e-10`.

## Sign / dtype conventions

- Wavelengths are meters. Fiber attenuation is `dB/km`; fiber lengths are
  meters and converted to kilometers for static attenuation.
- `phase_noise_rms_rad` is a phase RMS at `wavelength_center_m`; the stored
  random variable is OPD in meters.
- `fiber_phase` and `chromatic_dispersion_phase` return real float arrays in
  radians. `apply_transport` converts them to complex amplitudes through
  `exp(1j * phase)`.
- `compute_loss_budget` returns power transmission. Any field-stage attenuation
  must multiply amplitudes by `sqrt(total_transmission)`.
- Seed `None` means stochastic draws. Integer seeds must reproduce exact arrays
  for the same reference implementation.

## Edge cases

- KILO scalar `fiber_phase` shape is `(M,)`; array wavelength shape is
  `(n_wavelength, M)`. With default four telescopes and four subs each, the
  pins are `(16,)` and `(5, 16)`.
- KILO heterogeneous sub-aperture counts are supported for `fiber_phase`; the
  test pins shape `(20,)` for `(6, 6, 4, 4)` with each telescope block
  internally identical.
- NAYRA scalar `fiber_phase` shape is `(n_sub,)`; array wavelength shape is
  `(n_wavelength, n_sub)`. The tests pin `(4,)` and `(5, 4)`.
- A 1D transported field with more than one wavelength is invalid in NAYRA and
  raises `ValueError` matching `"1D field requires single-element"`.
- Zero phase noise should leave phase-only transport at unity amplitude. Loss
  changes amplitudes but preserves uniformity across identical NAYRA fibers.

## Regression-pinning tests

KILO fiber phase and dispersion pins:

- `TestFiberPhase.test_shape_scalar_wavelength`: `fiber_phase(FiberConfig(),
  1.0e-6, n_sub_per_telescope=4).shape == (16,)`.
- `TestFiberPhase.test_shape_array_wavelength`: `np.linspace(0.95e-6,
  1.05e-6, 5)` gives shape `(5, 16)`.
- `TestFiberPhase.test_small_noise`: 100 stochastic calls with
  `phase_noise_rms_rad=0.01` have `np.std(phases) < 0.05`.
- `TestFiberPhase.test_wavelength_dependent_phase`: with
  `phase_noise_rms_rad=0.1`, wavelengths `[0.9e-6, 1.0e-6, 1.1e-6]`,
  `wavelength_center_m=1.0e-6`, and `seed=42`, `phase[0]/phase[1] == 1/0.9`
  and `phase[2]/phase[1] == 1/1.1` at `rtol=1e-10`.
- The seeded KILO center-wavelength telescope phases for that test are
  `[0.03047170797544314, -0.10399841062404956, 0.07504511958064572,
  0.0940564716391214]`, each repeated four times.
- `TestFiberPhaseCommonMode.test_common_mode_within_telescope`: for
  `1.25e-6`, `n_sub_per_telescope=3`, and `seed=42`, shape is `(12,)` and
  each telescope block of three values is identical.
- `TestChromaticDispersion.test_zero_at_center_wavelength`: center wavelength
  output is all zero within `atol=1e-10`.
- `TestChromaticDispersion.test_shape`: `np.linspace(0.95e-6, 1.05e-6, 10)`
  gives shape `(10, 16)`.
- `TestChromaticDispersion.test_small_for_matched_fibers`: with
  `length_matching_error_m=0.001`, `max(abs(phase))` is
  `0.09795084871189812`, satisfying `< 1.0`.

NAYRA pinning tests:

- `TestFiberPhase.test_shape_scalar_wavelength`: shape `(4,)`.
- `TestFiberPhase.test_shape_array_wavelength`: shape `(5, 4)`.
- `TestFiberPhase.test_wavelength_dependent_phase`: the same seeded OPD draw
  gives ratios `1/0.9` and `1/1.1`, but without KILO telescope-block repeats.
- NAYRA seeded center-wavelength phases for `n_sub=4`, `seed=42` are
  `[0.03047170797544314, -0.10399841062404956, 0.07504511958064572,
  0.0940564716391214]`.
- `TestChromaticDispersion.test_zero_at_center_wavelength`: all zero within
  `atol=1e-10`.
- `TestChromaticDispersion.test_shape`: shape `(10, 4)`.
- `TestChromaticDispersion.test_small_for_matched_fibers`: with
  `length_matching_error_m=0.001`, `max(abs(phase))` is
  `0.07223875092502485`, satisfying `< 1.0`.
- `TestApplyTransportSeed`: same seed gives exact array equality, different
  seeds produce different output, and `phase_noise_seed=None` is stochastic.

## Open questions deferred to Pass 3

- PM980 attenuation, `jones_pm_fiber`, and `DualPolField` polarization details
  are explicitly deferred to Pass 3.
- Decide whether nullsim's transport stage owns NAYRA-style atmosphere and
  mirror losses or only fiber/connector losses, with separate throughput stages
  for sky and telescope.
- Decide how the stage maps reference seeds onto nullsim's pipeline RNG
  contract without introducing custom cache-key logic.
