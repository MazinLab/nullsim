---
date: 2026-05-17
---
### Phase 8 — MLX backend to parity + torch bootstrap normalization

Per the KILO parity audit (2026-05-17):

1. The MLX backend minimized `mean(dark) / mean(bright)` while the
   scipy/torch backends minimized the absolute dark-port leakage sum.
   Different loss landscape, silent under-delivery on Apple Silicon.
   MLX also used `L-BFGS-B` (torch uses full `BFGS` for the multi-modal
   576-dim basin), and ignored `params.n_restarts` (single-shot solve
   regardless of the user's request).
2. The torch broadband refine path normalized the throughput penalty
   by `n_dark_ports` (correct, matching KILO `chip_torch.py:330`) but
   the bootstrap path divided by `n_modes`. For Keck (6 bright /
   18 dark) the penalty scale was off by ~33% between warm-start and
   final, biasing the bootstrap init the broadband refinement starts
   from.

- `nullsim/stages/chip_optimization/mlx.py`:
  - Objective rewritten to the broadband absolute-leakage sum
    `sum_{i in dark, p} |U_k @ field_k|^2 / n_wl` (matches
    `_loss_numpy`).
  - `scipy.optimize.minimize` switched from `L-BFGS-B` to `BFGS`.
  - KILO master-RNG restart loop ported from the scipy backend so
    `params.n_restarts` is actually honoured. Bails early when
    `best_loss < target_null_depth`.
  - Module docstring updated to reflect the new contract.
  - Remaining gap: `kernel_order > 0` still silently collapsed to
    `kernel_order = 0`; tracked separately.
- `nullsim/stages/chip_optimization/torch.py:480`: bootstrap
  throughput-penalty normalization changed from `n_modes` to
  `n_dark_ports`.

The existing `test_backends_agree_on_final_null_depth` parametrization
(m in {4, 8}, 1% relative tolerance) now passes for the MLX backend
too — it was previously only running scipy + torch because MLX's
divergent objective would have failed the assertion.
