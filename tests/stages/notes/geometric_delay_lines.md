# GeometricDelayLines Stage Notes

## KILO source surveyed

The task path `kilo/tests/...` is stale
for this checkout. The KILO package root is
`kilo`; tests live under
`kilo/tests`.

Files inspected:

- `kilo/transport.py` lines 275-308:
  `FiberConfig` delay-line fields and units.
- `kilo/transport.py` lines 520-539:
  `DelayLineLoss` dataclass.
- `kilo/transport.py` lines 541-581:
  `compute_delay_line_lengths`.
- `kilo/transport.py` lines 584-651:
  `delay_line_loss`.
- `kilo/transport.py` lines 654-765:
  default altitude and mean/azimuth-averaged transmission helpers.
- `kilo/tests/test_transport.py` lines 820-1018:
  delay length, loss, and mean-transmission tests.
- `kilo/tests/test_transport.py` lines 1435-1520:
  fixed flat-dB delay-line override tests.

## NAYRA source surveyed

NAYRA is a single-bench instrument with no geometric delay-line equivalent.

Files inspected:

- `nayra/transport.py`
  lines 1-20: explicitly drops geometric delay lines, pointing-dependent loss,
  and azimuth-averaged delay-line helpers.
- `nayra/tests/test_eelt.py`
  lines 48-70: asserts KILO-only delay-line symbols are absent from
  `nayra.transport`.
- `nayra/tests/test_transport.py`
  lines 1-481: no delay-line tests or helpers.

## Non-obvious invariants

- ENU-pointing invariant: telescope positions are ENU meters, ordered `(east, north, up)`. Source
  altitude and azimuth are degrees; azimuth convention is `N=0`, `E=90`.
- Source-vector invariant: the source unit vector is
  `[cos(alt) * sin(az), cos(alt) * cos(az), sin(alt)]`.
- Effective-path invariant: KILO's geometric term is `-telescope_enu_m @ source_unit_vector`. The
  effective path at the combiner is
  `geometric_term + fiber_lengths_m * n_eff`.
- Nonnegative-delay invariant: the telescope with the maximum effective path is the slowest path and needs
  zero delay. All others receive
  `delay_lengths_m = max(effective_delay) - effective_delay`, so at least one
  delay is zero and all delays are non-negative.
- OPD-length invariant: `delay_lengths_m` is an optical-path-equivalent length because the fiber
  term included `n_eff`. Propagation loss is applied per physical waveguide
  meter: `physical_length_m = delay_lengths_m / n_eff`.
- Photonic-loss invariant: the photonic loss model is only `physical_length_m * delay_line_loss_db_per_m`
  plus the optional fixed override. It does not include bend loss, connector
  loss, dispersion, or any richer spool model.
- Transmission invariant: `delay_line_loss` returns power transmission as
  `10**(-loss_db / 10)`. Any field stage must convert to amplitude with
  `sqrt(transmission)`.
- Fixed-override invariant: `fixed_delay_line_loss_db_per_telescope` replaces the photonic model per
  telescope when an entry is a float. For those telescopes, reported delay
  length is set to `0.0` and loss is the fixed dB value. `None` entries keep
  the photonic model. With `fixed_delay_line_loss_mode = "attenuator"`, fixed
  entries instead apply the fixed dB value while preserving the geometric delay
  length for chromatic phase and replay diagnostics.
- State-contract exception: `delay_lines.geometric` declares and reads
  `scene` specifically for `altitude_deg` and `azimuth_deg` science-target
  pointing. This is the documented exception to INS-8-008 for WP3; the
  pointing direction changes delay-line lengths and therefore throughput even
  though the stage does not otherwise inspect astrophysical scene fields.
- Averaging invariant: `mean_delay_line_transmission` expands telescope transmission to
  sub-apertures with `np.repeat` and averages uniformly. The azimuth-averaged
  helper samples `np.linspace(0, 360, n_azimuths, endpoint=False)`.

## Sign / dtype conventions

- Altitude and azimuth inputs are degrees and are converted with `np.radians`.
- Delay arrays are float NumPy arrays in meters; losses are float arrays in
  dB; transmission is dimensionless power transmission.
- East-positive baselines with a source at azimuth `90 deg` make the eastern
  telescope geometrically earlier, so it receives positive delay when fiber
  lengths are equal.
- The KILO comments call the returned delay a length in meters, but the loss
  test makes the convention explicit: divide by `n_eff` before applying
  `dB/m`.

## Edge cases

- Tests allow a small numerical tolerance for non-negative delays:
  all delays must be `>= -1e-10`.
- At least one telescope must have delay `< 1e-10` for every pointing.
- At zenith with equal 780 m fibers, only the Maunakea altitude spread
  contributes; the max delay is pinned below `100 m`.
- At lower altitude, mean delay-line transmission decreases because geometric
  path differences grow.
- Heterogeneous sub-aperture counts such as `(6, 6, 4, 4)` are accepted by the
  mean/azimuth-averaged transmission helpers.
- A fixed override tuple with the wrong telescope count raises `ValueError`
  matching `"does not match telescope count"`.

## Regression-pinning tests

KILO delay-line length pins from `TestDelayLineLength`:

- `test_lengths_always_non_negative`: for altitudes `[10.0, 45.0, 90.0]` and
  azimuths `[0.0, 90.0, 180.0, 270.0]`, all lengths are `>= -1e-10`.
- `test_at_least_one_zero`: at `(alt=45.0, az=90.0)`,
  `min(lengths) < 1e-10`.
- With default 780 m fibers, `(alt=45.0, az=90.0)` gives delay lengths
  `[100.60927971567116, 138.5469787838142, 0.0, 567.2982020179873]`.
- At zenith with equal fibers, lengths are
  `[6.003939523862982, 6.003834084448954, 0.0, 73.98589885564297]`,
  and the max is below `100.0`.
- At zenith with `lengths_m=(300.0, 300.0, 350.0, 650.0)`, lengths are
  `[439.6580406682201, 439.6579352288061, 361.13410114435703, 0.0]`.
  The Gemini-minus-Keck altitude difference is `67.98195933178002 m`;
  expected and actual Keck/Gemini delay difference are both
  `439.65804066822 m` within the test tolerance of `10.0 m`.
- `test_shape`: default four-telescope output shape is `(4,)`.

KILO loss and transmission pins from `TestDelayLineLoss` and
`TestMeanDelayLineTransmission`:

- `test_loss_matches_physical_length_times_rate`: `loss_db` equals
  `(delay_lengths_m / n_eff) * delay_line_loss_db_per_m` exactly under
  `np.testing.assert_allclose`.
- At `(alt=45.0, az=90.0)`, default loss dB is
  `[2.3584635344269307, 3.247791835803698, 0.0, 13.298496186301412]`
  and transmission is
  `[0.580969919187712, 0.4733918932247486, 1.0, 0.04678971300610677]`.
- At `(alt=60.0, az=90.0)`, default loss dB is
  `[1.7192011466326498, 2.348050314535714, 0.0, 10.03827786787422]`
  and transmission is
  `[0.6731004568708244, 0.5823646010069443, 1.0, 0.09912249229185875]`.
- `test_zero_delay_has_unity_transmission`: the minimum-delay telescope has
  transmission within `1e-14` of `1.0`.
- Zenith mean transmission with seven sub-apertures per telescope is
  `0.9017448758245363`, satisfying `> 0.80`.
- At azimuth `180 deg`, mean transmission is `0.778871594017755` for
  altitude `70 deg` and `0.6287915206505753` for altitude `30 deg`, pinning
  the lower-elevation degradation.
- At `(alt=60.0, az=90.0)`, mean transmission with seven sub-apertures per
  telescope is `0.5886468875424068`.
- `azimuth_averaged_delay_line_transmission(..., altitude=90.0, n_sub=7)` is
  `0.9017448758245367`, satisfying `> 0.80`.
- Heterogeneous `(6, 6, 4, 4)` azimuth-averaged transmission at `60 deg` is
  `0.6278835656762611`.

Fixed flat-dB override pins from `TestFixedDelayLineLossOverride`:

- Default `FiberConfig().fixed_delay_line_loss_db_per_telescope is None`.
- With two telescopes at `[[0, 0, 0], [85.6, 0, 0]]`, equal `64 m` fibers,
  `(alt=60.0, az=90.0)`, and no fixed override, telescope 0 has zero loss and
  telescope 1 has nonzero photonic loss.
- With override `(0.0, 5.1)`, loss is `[0.0, 5.1]`, delay lengths are
  `[0.0, 0.0]`, and transmission is
  `[1.0, 0.30902954325135906]`.
- With partial override `(None, 5.1)` at `(alt=45.0, az=270.0)`, telescope 0
  keeps positive photonic loss and telescope 1 is fixed at `5.1 dB`.

## Open questions deferred to Pass 3

- The future nullsim stage needs an upstream geometry convention for converting
  target coordinates and observation time into KILO-style altitude and azimuth.
- Decide whether fixed commercial delay-line losses should be a delay-line
  stage parameter or a transport hardware preset.
- Decide how delay-line loss contributes to the throughput sub-budget while
  keeping static fiber loss in the SMF28 stage.
