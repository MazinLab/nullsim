---
date: 2026-05-17
---

### Tier 3.1 — Chromatic chip pass-through (option 2)

Closes the deferred chromaticity gap from the Tier 3 commit. The torch
and MLX optimizers now publish the chromatic `U_stack` they actually
evaluated; the chip stage uses it verbatim. The chip is now genuinely
broadband — its deployed unitary varies across wavelength the way a
real physical chip's does — and matches the optimizer at every
detector bin instead of collapsing to the design-wavelength constant.

#### Why the bug existed

`_canonicalize_clements_params` decomposes the optimizer's design-
wavelength unitary into `nullsim.physics.clements` (lexicographic)
angles. The torch/MLX optimizers internally use KILO's column-parity
layout (`_clements_unitary_ri`). Both layouts produce the same `U` at
`wl_design`, so the canonicalization is correct *at the design
wavelength*. But the two layouts compose MZIs in different orders, so
applying the same phase rescaling `φ → φ × wl_design/wl` to the two
parameter sets produces **different** `U(λ)` at off-design
wavelengths. Only the KILO-layout `(θ, φ)` represent the physical MZI
phase shifters directly; the physics.clements decomposition is a math
reparameterization that loses the direct mapping to optical-path
phases.

The "Ultra-review High #5" pass settled this by deploying
monochromatically at `wl_design` (chip stage skipped the rescale) —
correct at one bin, off by `max|ΔU| ≈ 0.6` at band edges.

#### What changed

- `nullsim/stages/chip_optimization/__init__.py:_write_results` accepts
  an optional `optimal_u_stack: NDArray[np.complex128]` and persists it
  in the payload.
- `nullsim/stages/chip_optimization/torch.py` and
  `nullsim/stages/chip_optimization/mlx.py` build the chromatic
  `U(λ)` at every detector wavelength using their native KILO column-
  parity builders, with `φ × wl_design/wl` scaling per bin, after DAC
  quantization. The result is shipped through `_write_results`.
- `nullsim/stages/chip/kernel_mzi_mesh.py:_resolve_chip_params` now
  also returns the optional `precomputed_u_stack`. When present,
  `KernelMZIMesh.apply` uses it directly — DAC quantization,
  canonicalization, per-λ rebuild, and the `rescale_per_wavelength`
  flag are all bypassed on this path (the optimizer already did the
  quantization and the rescaling in the right layout).
- The chip-stage payload now reports `precomputed_u_stack_used: bool`
  for diagnostics.

#### Test changes

- `test_optimizer_to_kernel_handoff_broadband_uses_monochromatic_kernel`
  renamed in spirit (kept the existing test ID for git blame) and
  rewritten to pin the new chromatic behavior:
  1. the optimizer publishes `optimal_u_stack`,
  2. the chip stage's `u_stack` component is bit-for-bit identical to
     the optimizer's `optimal_u_stack` (torch path), and
  3. per-wavelength chip outputs **vary** across the band (the
     chromatic invariant) — pre-fix they were identical to machine
     precision.

#### Coverage

685 tests pass (1 pre-existing kilo_maunakea parity failure unrelated
to this work). The Tier 3.1 gap noted in
`changelog.d/2026-05-17_tier3-chip-fidelity.md` is now closed.
