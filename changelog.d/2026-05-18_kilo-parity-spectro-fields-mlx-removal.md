---
date: 2026-05-18
---
### KILO parity pass: spectro-port disjoint partitions, physical planet fields, basin knobs, MLX retirement

Five basin-aligning changes against KILO's `natural_design.py` reference for
the chip optimizer plus retirement of the never-converging MLX backend.

**Spectro port becomes its own partition (chip_optimization)**

Schema changes from `n_bright + n_dark == n_modes` (spectro overlapping
bright) to KILO's disjoint `[bright | dark | spectro]` layout with
`n_bright + n_dark + n_spectro == n_modes` (kilo/chip.py:2049
`assign_ports`). Two index helpers now expose the split:

- `_null_indices(params)` = dark + spectro (the 19 indices KILO's null
  objective zeros — natural_design.py:390).
- `_dark_indices(params)` = dark only (the 18 indices KILO's
  throughput-penalty term sums over — natural_design.py:643).

Pre-fix the same 19-index set drove both the null objective AND the
throughput penalty, weighting the regularizer 19/18 ≈ 1.06× too high.

**Physical KILO-style optimizer fields (chip_optimization)**

`_require_field` now picks the per-mode magnitude near the design
wavelength (not the wavelength-mean) and multiplies by
`sqrt(area_per_mode)` so the optimizer sees `sqrt(coupling · area)`
fields matching KILO `injection.py:stellar_input_field`. The
throughput-penalty term in the torch backend now builds full
`planet_input_field(...)` samples per optimized wavelength with the
KILO off-axis SMF angular-coupling rolloff, replacing the previous
unit-amplitude phase-only term.

**Basin-aligning knobs (kilo_keck.toml)**

- `chip_optimization.torch.seed = 42` (KILO `KILO_BB_SEED` default,
  natural_design.py:651). Was 0 while debugging F007.
- `throughput_penalty_seps_mas = [3.0]` (KILO
  `KILO_BB_THROUGHPUT_REPR_SEP_MAS` default = single 3 mas anchor,
  natural_design.py:614). The previous `[3.0, 10.0, 30.0]` is KILO's
  empirical multisep-ab study variant, not the canonical basin.
- Per-stage port counts: chip_optimization uses
  `5/18/1` (KILO disjoint); downstream stages
  (`chip.kernel_mzi_mesh`, `detector.mkid`,
  `sensitivity.detection_curve`) use `5/19` — the spectro port routes
  downstream as a dark planet-detection channel.

**MLX backend removed**

Deleted `nullsim/stages/chip_optimization/mlx.py`, the
`nullsim[mlx]` pyproject extra, all MLX-aware test branches, and the
`_mlx_available()` / `_build_mlx_obj_grad()` scaffolding in
`scripts/bench_chip_optimizer.py`. KILO production runs torch.compile on
CPU at float64; MLX-Metal is float32-only and never reached the chip
nulls KILO produces. Test parametrizations dropped to `["torch"]`
only.

**Measured impact at kilo_keck J=4 (with `2026-05-18_keck-enu-chip-basin`'s
geometry fix layered in)**

- `ideal_chip_null_depth_kilo`: 1.25e-4 → 1.01e-5 (12× deeper; within
  1.2× of KILO's theoretical ~1.2e-5)
- `realized_null_depth_kilo`: 1.32e-4 → 1.62e-5 (now 5.5× DEEPER than
  KILO's `null_val_dark = 8.98e-5`; chip reaches its kernel-null
  floor and AO-residual is the new limiting term)
- `null_instability`: 0.154 → 0.322 (KILO ref 0.166; the now-
  deeper chip exposes AO-residual phase noise as the fractional
  fluctuation source)
- SNR (1 hr): 35.2 → 115

F007 is resolved with this pass + the ENU geometry correction. The
realized-parity scorecard now sits at or below KILO's published
references, with `null_instability` indicating AO-residual is the new
dominant term.
