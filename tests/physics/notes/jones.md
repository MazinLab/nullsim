# Jones Matrix Notes

## Source files reviewed

Candidate search:

```text
grep -rilE 'jones|stokes|polariz|waveplate|retarder|birefring|half_wave|quarter_wave' kilo nayra --include='*.py'
```

The search returned these package files, all reviewed:

- `kilo/injection.py`
- `kilo/config.py`
- `kilo/studies/paperv3_plots.py`
- `kilo/studies/pdi_sensitivity.py`
- `kilo/studies/pdi_exoplanet_table.py`
- `kilo/studies/dualpol_loss_budget.py`
- `kilo/studies/keck_only_dualpol.py`
- `kilo/studies/pdi_planet_table.py`
- `kilo/transport.py`
- `kilo/chip.py`
- `kilo/exoplanets.py`
- `kilo/performance.py`
- `nayra/multi_band.py`
- `nayra/transport.py`
- `nayra/frontend.py`
- `nayra/polarization.py`
- `nayra/performance.py`
- `nayra/_kilo_adapter.py`

Primary behavior sources:

- KILO `transport.py`: `jones_identity`, `jones_random_birefringence`,
  `jones_linear_retarder`, `jones_hwp`, `jones_pbs_arm`,
  `jones_pm_fiber`, `DualPolField`, `apply_pbs`, `apply_hwp`, and the
  scalar `apply_transport(..., jones_matrices=...)` projection path.
- NAYRA `polarization.py`: updated dual-pol primitives, especially
  `jones_linear_retarder`, `jones_hwp`, `jones_pbs_arm`,
  `jones_pm_fiber`, `apply_pbs`, `apply_hwp`, `pm_fiber_per_at_length`,
  and `pdi_imperfection_factor`.
- NAYRA `frontend.py`: warm front-end order, including scalar unpolarized
  HWP pass-through and polarized `DualPolField` Jones projection through
  HWP plus PBS arms.
- KILO `injection.py`: Stokes-style polarized planet projection into
  the PBS arms.
- KILO `exoplanets.py`, KILO `performance.py`, and NAYRA `performance.py`:
  planet polarization angle/fraction and PDI Q/U recovery conventions.

Reference tests reviewed:

- `kilo/tests/test_transport.py`
- `kilo/tests/test_injection.py`
- `nayra/tests/test_polarization.py`
- `nayra/tests/test_frontend.py`
- `nayra/tests/test_multi_band.py`
- `nayra/tests/test_eelt.py`

The remaining hits are study/config/reporting files or downstream callers.
They do not define additional Jones algebra, but they do record why the
dual-polarization path exists and how PDI outputs are interpreted.

## Time convention adopted: exp(-iwt)

nullsim adopts the optical time dependence `exp(-iwt)`. Under this
convention, a positive complex phase factor advances the optical phase.

KILO and NAYRA do not state an explicit `exp(-iwt)` or `exp(+iwt)` convention
in their source comments. The implemented sign convention is nevertheless
clear from `jones_linear_retarder(theta, delta)`: in the retarder eigenbasis
the fast-axis term receives `exp(+i delta / 2)` and the orthogonal term
receives `exp(-i delta / 2)`. Therefore an ideal HWP with fast axis along
`+x` is equivalent to `diag(i, -i)`, up to global phase.

Pinning tests should preserve source-observable behavior rather than depend
on a physically irrelevant global phase:

- HWP at `0` has opposite phases on horizontal and vertical components.
- HWP at `pi / 8` rotates linear polarization by `pi / 4`.
- HWP at `pi / 4` flips horizontal to vertical and vertical to horizontal
  up to global phase.

## Fast-axis convention for waveplates

KILO and NAYRA both use a Jones basis ordered as horizontal/vertical,
`(+x, +y)`, in the instrument/PBS frame. A waveplate fast-axis angle is a
finite real angle in radians, measured from `+x` toward `+y`.

The source retarder implementation is algebraically:

```text
J(theta, delta) =
[[cos(theta)^2 exp(+i delta/2) + sin(theta)^2 exp(-i delta/2),
  cos(theta) sin(theta) (exp(+i delta/2) - exp(-i delta/2))],
 [cos(theta) sin(theta) (exp(+i delta/2) - exp(-i delta/2)),
  sin(theta)^2 exp(+i delta/2) + cos(theta)^2 exp(-i delta/2)]]
```

This is equivalent to rotating an ideal diagonal retarder into the lab frame.
For an ideal HWP, the physical plate angle `theta` rotates a linear
polarization angle by `2 theta`. Source tests explicitly pin:

- `theta = 0`: HWP is a diagonal phase plate.
- `theta = pi / 8`: horizontal input exits with equal horizontal/vertical
  power and common component phase.
- `theta = pi / 4`: horizontal input exits as vertical; KILO injection tests
  also pin the corresponding Stokes-Q arm swap.

## Angle units: radians

All Jones and Stokes angles in the reviewed source are radians:

- HWP angles use `0`, `pi / 8`, `pi / 4`, `3 pi / 8`.
- PBS axis offsets are radians (`1e-3` rad defaults in config).
- Planet polarization angles are radians in the PBS/instrument frame.
- KILO `planet_polarization_angle` wraps the angle modulo `pi` because
  linear polarization has no arrow direction.

The only degree-valued quantities in adjacent code are unrelated sky or
pointing angles such as altitude, azimuth, and position angle labels.

## Dtype: complex128 for matrices, float64 for angles

nullsim should use `complex128` for Jones matrices and `float64` for angle
inputs. This is the explicit dtype contract for the pure physics primitive.

KILO and NAYRA matrix builders use `dtype=complex`, which maps to
`complex128` in the target NumPy runtime. One exception is KILO
`jones_identity`, which returns tiled `np.eye(2)` and is therefore `float64`.
That identity matrix is deliberately bit-identical to the no-Jones scalar
transport path. In nullsim, the identity-like rotator/polarizer/waveplate
constructors should still return `complex128`; bit-identical transport is a
stage-level concern, not a pure-math dtype exception.

## Stokes convention: I,Q,U,V order with right-handed observer-facing frame

KILO and NAYRA do not expose a full Stokes-vector conversion primitive.
Their polarized-planet and PDI code imply the linear Stokes convention:

```text
Q = |P| cos(2 chi)
U = |P| sin(2 chi)
```

where `chi` is measured in the PBS/instrument frame. The arm fractions for
an ideal HWP followed by an ideal PBS are:

```text
f_A = T_PBS * (1 + |P| cos(2 (chi - 2 theta_HWP))) / 2
f_B = T_PBS * (1 - |P| cos(2 (chi - 2 theta_HWP))) / 2
```

nullsim adopts Stokes order `(I, Q, U, V)` in a right-handed observer-facing
frame. The pinning tests therefore use:

- horizontal `(+x)` -> `[1, +1, 0, 0]`
- vertical `(+y)` -> `[1, -1, 0, 0]`
- `+45 deg` linear -> `[1, 0, +1, 0]`
- Jones vector `[1, i] / sqrt(2)` -> `[1, 0, 0, +1]`

The sign of `V` is the nullsim convention; KILO/NAYRA do not exercise
circular polarization.

## Edge cases captured from source comments

- Zero retardance is identity for `jones_linear_retarder`.
- HWP behavior is invariant to a global phase; tests should compare output
  states up to global phase where phase is not physically observable.
- PBS arm names are exactly `"A"` and `"B"` in the reference code. Arm A
  transmits `+x`; arm B transmits `+y`.
- PBS extinction ratio is specified as a power leakage in dB. A 30 dB
  extinction ratio means blocked power `1e-3` and blocked amplitude
  `sqrt(1e-3)`.
- PBS insertion loss is an amplitude factor `10 ** (-loss_db / 20)`, so
  transmitted power is reduced by `10 ** (-loss_db / 10)`.
- NAYRA corrected the warm front-end ordering: for unpolarized scalar input,
  HWP is a no-op before the PBS split; applying PBS first and then an HWP to
  coherent equal arms incorrectly routes unpolarized power as if it were a
  coherent polarization vector.
- KILO `apply_hwp` rejects scalar input, while NAYRA `apply_hwp` passes scalar
  unpolarized input through. This is a stage/interface decision, not a pure
  Jones-matrix primitive.
- KILO's older PM-fiber Jones model made the in-frame coupling symmetric so
  slow-axis misalignment did not commute away. NAYRA's newer PM-fiber model
  adds a random retardance phase and an intrinsic PER floor, keeping the
  matrices unitary and preserving the same "alignment must matter" invariant.
- Stress birefringence retardance in KILO scales as `1 / wavelength_m` for a
  fixed physical OPD.

## References

- Collett, E. *Field Guide to Polarization*, SPIE FG05 (2005), Eqs. 4.1-4.12.
- Born, M. and Wolf, E. *Principles of Optics*, Section 1.4.2.
