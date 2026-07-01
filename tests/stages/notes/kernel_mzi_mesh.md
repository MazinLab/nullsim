# Kernel MZI Mesh Stage Notes

## KILO source surveyed

Files inspected:

- `kilo/chip.py` lines 72-157:
  `mzi_transfer_matrix`, `quantize_phases`, `clements_to_unitary`.
- `kilo/chip.py` lines 160-177:
  `build_chip_unitary` with DAC quantization.
- `kilo/chip.py` lines 180-232:
  `build_chip_unitary_at_wavelength` and wavelength-rescaling model.
- `kilo/chip.py` lines 96-111:
  `quantize_phases` helper and bit-width convention.
- `kilo/chip_torch.py` lines 63-134:
  `_clements_unitary_ri` real/imag-split Clements builder.
- `nayra/chip.py` lines 1-11:
  module docstring, scope statement.
- `nayra/chip.py` lines 118-161:
  `clements_to_unitary` — identical to KILO.
- `nayra/chip.py` lines 164-181:
  `build_chip_unitary` — identical to KILO; no `build_chip_unitary_at_wavelength`.
- `nayra/chip_remote.py`:
  confirmed as Ray-cluster dispatch layer; excluded from nullsim scope.

## Non-obvious invariants

### 1. Parameter vector layout

`build_chip_unitary` (chip.py:160-177) accepts `params` of length
`2*n_mzi + m` where `n_mzi = m*(m-1)//2` (chip.py:130).

Exact slice layout (chip.py:131-133, mirrored in chip_torch.py:79-81 and
chip_mlx.py:49-52):

```
theta = params[0 : n_mzi]           # MZI internal (beam-splitter) phases
phi   = params[n_mzi : 2*n_mzi]     # MZI external (arm) phases
alpha = params[2*n_mzi : 2*n_mzi+m] # diagonal output phases
```

`n_params_for_unitary(m)` (chip.py:268-280) returns exactly `2*n_mzi + m`.
At m=24: n_mzi=276, total params=576.

### 2. Clements column ordering (the near-neighbour MZI sequence)

`clements_to_unitary` (chip.py:114-157) iterates:

```python
idx = 0
for col in range(m):
    start = col % 2          # alternates 0, 1, 0, 1, ...
    for row in range(start, m - 1, 2):
        if idx >= n_mzi:
            break
        # apply MZI at (row, row+1)
        idx += 1
```

Column 0 touches rows 0,2,4,...; column 1 touches rows 1,3,5,.... The
sequence fills the `n_mzi` MZI indices in column-major order until exhausted.
This is a rectangular nearest-neighbour Clements mesh, NOT the diagonal
Reck decomposition. The vectorised torch/MLX backends (chip_torch.py:88-128,
chip_mlx.py:58-98) reproduce this column loop exactly with `start_row = col % 2`
and `n_active = (m - start_row) // 2`.

### 3. MZI transfer matrix convention

`mzi_transfer_matrix(theta, phi)` (chip.py:72-93):

```
T = [[exp(i*phi)*sin(theta),  cos(theta)],
     [exp(i*phi)*cos(theta), -sin(theta)]]
```

Applied to rows `(row, row+1)` of U on the LEFT (chip.py:148-151):
`u[row]   = T[0,0]*old_row + T[0,1]*old_row+1`
`u[row+1] = T[1,0]*old_row + T[1,1]*old_row+1`

Diagonal phases applied last as `diag(exp(i*alpha)) @ U` (chip.py:155).

### 4. Wavelength variant (chip.py:180-232)

`build_chip_unitary_at_wavelength` applies **DAC quantization first** (at the
design wavelength), then rescales only `phi` and `alpha`; `theta` is left
unchanged. Exact implementation (chip.py:223-231):

```python
if dac_bits is not None:
    params = quantize_phases(params, m, dac_bits)  # step 1: quantize at design wl
if wl_target_m != wl_design_m:
    n_mzi = m * (m - 1) // 2
    scale = wl_design_m / wl_target_m
    scaled[n_mzi:2*n_mzi] *= scale    # phis rescaled
    scaled[2*n_mzi:2*n_mzi+m] *= scale # diag rescaled
    params = scaled
return clements_to_unitary(params, m)
```

Physical model: `phi` and `alpha` are set by physical path-length differences
whose accumulated phase is proportional to `1/lambda`. `theta` is set by the
directional-coupler geometry, which is wavelength-independent to leading order
(see chip.py:204-221 for the achromatic-DC model assumption and its limits).

The broadband optimizer uses `wl_scales[i] = wl_center / wl_i` and rescales
identically (chip_torch.py:319-321):
`params_i = torch.cat([thetas, phis * scale, diag * scale], dim=0)`

**Ordering rule:** DAC quantize is applied BEFORE wavelength rescaling.
Calling `build_chip_unitary_at_wavelength(params, m, wl_design, wl_design)`
reproduces `build_chip_unitary(params, m)` exactly (chip.py:200).

### 5. DAC quantization helper

`quantize_phases(params, m, bits)` (chip.py:96-111):

```python
step = 2 * np.pi / (1 << bits)
return step * np.round(params / step)
```

Default bit-width: no default; caller must supply `dac_bits`. KILO tests use
16-bit as a representative high-precision case and 4-bit for coarse
degradation. There is no default `dac_bits` in any optimizer call.

Quantization is uniform across all three parameter blocks (theta, phi, diag);
there is no per-block bit-depth control.

### 6. Nullsim field tensor and einsum convention

KILO operates on fields of shape `(M,)` for monochromatic or `(n_wl, M)` for
broadband. The chip stage applies `U @ field` independently for each
wavelength row.

Nullsim's internal field tensor is `(n_modes, n_wl, n_pol)`. The chip stage
must apply the `(m, m)` unitary to the `n_modes` axis for each `(wl, pol)`
pair. The correct contraction is:

```python
# U: (m, m); field: (m, n_wl, n_pol)
output = np.einsum("ij,jwp->iwp", U, field)
# or equivalently per wl,pol slice: output[:, wl, pol] = U @ field[:, wl, pol]
```

Do NOT apply a single broadband U built at one wavelength to the whole
`(n_modes, n_wl, n_pol)` tensor; each wavelength requires its own rescaled U.

### 7. NAYRA diff vs KILO

- `chip.py`: structurally identical Clements math. No `build_chip_unitary_at_wavelength`
  (NAYRA's single-bench scope had no need until the 2026 broadband resync).
  NAYRA's `clements_to_unitary` and `build_chip_unitary` are byte-for-byte
  copies of KILO's (confirmed nayra/chip.py:118-181).
- `null_depth_from_unitary` (nayra/chip.py:184-204): NAYRA's version returns
  `0.0` for zero-power fields rather than raising `ValueError` as KILO does
  (KILO chip.py:259-265). Nullsim should follow KILO's raise convention to
  avoid silently masking mis-shaped inputs.
- `chip_torch.py` (nayra): thin re-export of `kilo.chip_torch` — no
  independent implementation. Sets `torch.set_default_dtype(float64)` at
  module load (nayra/chip_torch.py:21-22).
- `chip_mlx.py` (nayra): thin re-export of `kilo.chip_mlx` — no
  independent implementation (nayra/chip_mlx.py:13-17).
- `chip_remote.py`: Ray-cluster dispatch. Excluded from nullsim scope.
  All remote calls guarded by `_should_use_remote()` which checks
  `NAYRA_OPT_BACKEND=remote` (nayra/chip.py:278-303).

### 8. chip_params storage (architect plan)

Per the Pass 3 architect plan: chip params stored as a pydantic field on the
chip stage dataclass. The optimizer writes optimized params to
`state.results.components`; there is no state flowthrough in Pass 3. The
stage simply reads its `params` field and calls `build_chip_unitary` or
`build_chip_unitary_at_wavelength` per wavelength.

## Sign / dtype conventions

- All phase parameters are in radians.
- `clements_to_unitary` returns complex128 (`dtype=complex` which numpy
  defaults to complex128). The torch backend uses float64 for real/imag
  components. The MLX backend uses float32.
- Wavelengths are always meters. `wl_design_m`, `wl_target_m` in meters.
- `scale = wl_design / wl_target`: scale > 1 when target is shorter than
  design (blue shift of phase), scale < 1 for red shift.

## Edge cases

- `build_chip_unitary(params, m, dac_bits=None)` is identical to
  `clements_to_unitary(params, m)` — no copy, no quantization.
- `build_chip_unitary_at_wavelength(params, m, wl_design, wl_target=wl_design)`
  is identical to `build_chip_unitary(params, m)` (conditional at chip.py:225
  skips rescaling when `wl_target == wl_design`).
- When `m` is odd, the last column of the mesh has `n_active = 0` (or 1 MZI
  on alternating parity columns); the `if n_active == 0: continue` guard in
  chip_torch.py:93 / chip_mlx.py:63 handles this.
- The parameter vector must have exactly `2*n_mzi + m` elements; callers are
  responsible for shape checking. KILO does not validate this at runtime.
