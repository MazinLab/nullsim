# Clements Decomposition Notes

## Source files reviewed

Candidate search:

```bash
grep -rilE 'clements|reck|mzi|mesh|givens|interferometer|unitary' kilo nayra --include='*.py'
```

Targeted symbol search:

```bash
grep -nE 'clements|Clements|Reck|reck|Givens|givens' kilo/chip.py
```

KILO files reviewed in full (Clements / MZI core):

- `kilo/chip.py` — defines
  `mzi_transfer_matrix(theta, phi)`, `clements_to_unitary(params, m)`,
  `quantize_phases`, `build_chip_unitary`,
  `build_chip_unitary_at_wavelength`, `n_params_for_unitary`,
  `random_unitary`, and the scipy/torch/mlx null optimizers that
  consume the Clements parameter vector.
- `kilo/chip_torch.py` — torch port of
  the same Clements-to-unitary forward map, used as the broadband
  optimizer fast path.
- `kilo/chip_mlx.py` — Apple-Silicon
  mlx port of the same forward map.
- `kilo/config.py` — chip-stage config
  surface (M, dac_bits, optimizer choice).

NAYRA files reviewed in full:

- `nayra/chip.py` — explicit
  comment: "This module mirrors KILO's chip layer; NAYRA operates on a
  single E-ELT pupil sampled into M sub-apertures, but the chip math is
  geometry-agnostic (M-dimensional unitaries acting on M-vectors)." The
  same `clements_to_unitary` / `mzi_transfer_matrix` / parameter-vector
  layout is reused.
- `nayra/chip_torch.py`
- `nayra/chip_mlx.py`
- `nayra/multi_band.py`
- `nayra/config.py`

Reference tests reviewed:

- `kilo/tests/test_chip.py` — pins MZI
  unitarity, Clements unitarity over random params, deterministic
  reproducibility for fixed parameter vectors, total-power conservation
  through the mesh.
- `kilo/tests/test_chip_backends.py` —
  cross-backend agreement (numpy vs torch vs mlx) on the same parameter
  vector.

The remaining grep hits are study/plotting/optimizer files that consume
`clements_to_unitary` but do not redefine the algebra. Neither codebase
contains a *decomposition* routine that factors a target unitary into
Clements parameters — both repositories only implement the forward map
(parameters → unitary) and rely on numerical optimization to fit the
parameters. There is therefore no upstream decomposition test to lift.

The keyword "Reck" returns no hits in either tree; the keyword "Givens"
returns no hits. KILO and NAYRA chose Clements 2016 directly and never
implemented the Reck 1994 triangular mesh.

## Adopted conventions

### Mesh topology: rectangular Clements 2016, not triangular Reck 1994

nullsim adopts the rectangular Clements mesh of Clements *et al.*,
"Optimal design for universal multiport interferometers," Optica 3, 1460
(2016). KILO and NAYRA both call this out explicitly in `chip.py`
docstrings and parameter layouts. The rectangular topology has
half the optical depth of the Reck 1994 triangular mesh and matches the
fabricated chip layout that the KILO instrument is built around.

### Parameter-vector layout

KILO and NAYRA both flatten the Clements parameters into a single 1D
vector of length `m**2`:

```text
[ thetas (length n_mzi),
  phis   (length n_mzi),
  diag   (length m)     ]   with n_mzi = m * (m - 1) // 2
```

`thetas` are the MZI internal beam-splitter angles, `phis` are the
external relative phases, and `diag` is the output phase screen. A
`(theta, phi)` pair sets one MZI; `m` extra phases set the right-hand
output screen. Total parameter count is `m * (m - 1) + m = m**2`.

The reference chip-at-wavelength routine
(`build_chip_unitary_at_wavelength`) further documents that `thetas` are
set by physical directional couplers and are wavelength-independent to
leading order, while `phis` and `diag` are heater-driven path-length
phases that scale as `wl_design / wl_target`. nullsim preserves that
split when it grows a wavelength-dispersive chip stage; the pure-math
primitive does not need to know about wavelength.

### MZI rotation block

KILO `mzi_transfer_matrix(theta, phi)` is:

```text
T(theta, phi) = [[ exp(i phi) sin(theta),  cos(theta) ],
                 [ exp(i phi) cos(theta), -sin(theta) ]]
```

This is unitary for any real `(theta, phi)` and reduces to a permutation
plus a phase at the canonical fixed points (e.g. `theta = 0` swaps the
two arms up to a sign). nullsim's `_rotation_block(theta, phi)` plays
the equivalent role inside the column sweep: it is the 2×2 rotation
that, when applied from the left to two adjacent rows, zeroes the lower
of the two leading entries in the working column.

### Column-sweep order in the forward map

KILO `clements_to_unitary` applies MZIs in the rectangular Clements
ordering:

```text
for col in range(m):
    start = col % 2
    for row in range(start, m - 1, 2):
        apply MZI on rows (row, row+1)
```

even and odd columns alternate which row pairs they touch, producing
the rectangular brick-wall pattern of the Clements 2016 mesh. The final
diagonal phase screen is applied last:
`U = diag(exp(i diag_phases)) @ (mesh applied to I)`.

nullsim's `decompose_clements` runs the *inverse* sweep: for each
column from the leftmost to the rightmost, work from the bottom row
upward, choosing a `(theta, phi)` for each adjacent row pair so that
the rotation block left-applied to those two rows zeroes the lower of
the two entries in the current column. After all columns are processed
the working matrix is diagonal; its phases are returned as `alpha`.
`reconstruct_clements` replays the rotations in reverse order.

### Dtype: complex128 for unitaries, float64 for angles

nullsim coerces the input matrix to `complex128` before decomposition
and returns `complex128` from `reconstruct_clements`. Angle arrays
(`theta`, `phi`, `alpha`) are `float64`. KILO and NAYRA use `dtype=complex`
throughout, which maps to `complex128` in the target NumPy runtime;
parameter vectors are `float64` (numpy default). The pinning test for
dtype invariance pins `complex64` input → `complex128` output for the
decomposition surface, matching the same dtype-promotion rule used by
the Jones primitives.

### Identity-case branch and the theta=0 degeneracy

When the leading entry to be zeroed is already zero (within numerical
precision), the rotation that zeroes it is not unique — `theta = 0`
combined with any `phi` leaves the column unchanged. nullsim adopts the
deterministic branch `(theta, phi) = (0, 0)` in this case so that
decomposing the identity returns all-zero `theta` and `phi` arrays and
all-zero `alpha`. The pinning test `test_clements_identity_reconstructs_to_identity`
exercises this branch.

For the same reason, when the upper entry is zero but the lower is not,
the rotation must be a pure swap; the branch picks `theta = pi/2` with a
`phi` chosen to cancel the residual phase of the lower entry. This
keeps `reconstruct_clements(decompose_clements(I_n)) = I_n` exact at
machine precision rather than matched only within tolerance.

### Mesh sizing: arbitrary N, no odd-N special case

KILO `n_params_for_unitary(m) = m * m` accepts any `m >= 1`. NAYRA uses
the same formula. Neither codebase carves out a special case for odd
`m`. nullsim follows: `decompose_clements` accepts any square unitary
of side `n_modes >= 1`, and the `(theta, phi)` count is always
`n_modes * (n_modes - 1) // 2`. There is no separate triangular fallback
for odd `n_modes`.

## Architect-Spec Divergence

### 1. Algorithm: Givens-rotation factorisation, not the original Clements null sweep

The Clements 2016 paper presents the algorithm as an alternating series
of left- and right-multiplications by MZIs that null one entry at a time
across the upper and lower triangles, terminating with a residual
diagonal. The nullsim implementation factors the matrix purely from the
left, using a sequence of complex Givens-style 2×2 rotations on
adjacent rows, and absorbs the entire residual diagonal into `alpha` at
the end.

The two routes are mathematically equivalent for the *count* of
parameters (`m**2`) and for the rectangular brick-wall layout of the
forward map, but the internal sweep order differs from the
paper-literal Clements sweep. The nullsim sweep is column-major from
the leftmost column outward, bottom-row-up within each column. This
matches the forward map convention used by KILO/NAYRA `clements_to_unitary`
when the parameter list is consumed by their nested
`for col / for row` loops.

This is recorded as a divergence so that anyone cross-referencing
`decompose_clements` against the Clements 2016 paper sees the
substitution flagged. The behavioral contract — `reconstruct(decompose(U)) == U`
to machine precision for any unitary `U`, parameter count `m**2`,
identity round-trips exactly — is invariant under the substitution and
is what the pinning tests check.

### 2. No forward-map parity wrapper for the KILO `params` vector

KILO and NAYRA expose only the forward direction (parameters →
unitary). nullsim ships both directions but does not provide a thin
wrapper that takes a flat `params` vector in KILO's `[thetas, phis,
diag]` layout and returns a unitary. If a chip stage later needs to
consume a KILO-format parameter vector, it should construct a
`ClementsDecomposition` explicitly (with `order` derived from the
column-sweep iteration above) and call `reconstruct_clements`. This
keeps the public API typed (`ClementsDecomposition` is a frozen
dataclass) and prevents silent layout drift.

### 3. Output phase screen on the diagonal, not split fore/aft

Some Clements-style implementations split the global phase screen into
two halves, one before and one after the mesh, to symmetrize the
hardware layout. nullsim follows KILO/NAYRA: a single phase screen on
the *output* side, length `m`, returned as `alpha`. There is no
input-side phase screen.

## Pinning test provenance

`tests/physics/test_clements_pinning.py` is **derived from scratch**.
KILO and NAYRA test files (`tests/test_chip.py`,
`tests/test_chip_backends.py`) only exercise the forward map
(parameters → unitary), the unitarity of the result, and cross-backend
parity of the forward map. They do not test a decomposition routine
because neither package implements one.

The pinning suite therefore pins the *behavioral contract* required by
SPEC §15.2 directly:

- Haar-random `N=4` round trip: `reconstruct(decompose(U)) == U` to
  `atol=1e-12` for a Haar-distributed unitary drawn via
  Mezzadri 2007 Eq. (5.12).
- Identity case: `decompose(I_4)` reconstructs to `I_4` exactly.
- Dtype invariance: `complex64` input promotes to `complex128` output.
- Non-unitary input raises `ValueError` mentioning "unitary".

These are the invariants that downstream chip-stage code (the
broadband nulling optimizer, the wavelength-dispersive chip-at-wavelength
forward map, the DAC-quantization wrapper) relies on. They are also the
invariants that survive any future swap of the internal sweep
algorithm (paper-literal Clements vs. Givens vs. QR-flavored variants).

## References

- Clements, W. R., Humphreys, P. C., Metcalf, B. J., Kolthammer, W. S.,
  and Walmsley, I. A. "Optimal design for universal multiport
  interferometers," Optica 3, 1460 (2016), Eqs. (1)-(3).
- Reck, M., Zeilinger, A., Bernstein, H. J., and Bertani, P.
  "Experimental realization of any discrete unitary operator,"
  Phys. Rev. Lett. 73, 58 (1994). Cited for completeness; not adopted.
- Mezzadri, F. "How to generate random matrices from the classical
  compact groups," Notices Amer. Math. Soc. 54, 592 (2007), Eq. (5.12),
  for the Haar-distributed test fixture.
