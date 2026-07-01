---
date: 2026-05-18
---
### Chip-optimizer depth gap: investigation + open finding

Surfaced a real-but-bounded gap between nullsim's chip optimizer and KILO's
same-algorithm port. Documented in `.claude/spec/FINDINGS.md` as **F007**.

**What we found:** at kilo_keck J=4 (M=24, kernel_order=3, n_restarts=5, broadband
refine, torch.compile on), nullsim's `ideal_chip_null_depth_kilo ≈ 1.25e-4`.
Plugging KILO's cached chip params from
`kilo/results/cache/warm_start_keck_only_j.npz` into nullsim's pipeline
(rebuild U via the KILO column-parity Clements builder, nullsim's
`[0..5]` bright / `[6..23]` dark convention) gives `ideal = 1.19e-5` — so the
evaluation is correct and the entire gap lives in the optimizer trajectory.

**Six hypotheses tested and ruled out:**

1. *Broadband loss normalization (`1/n_wl` vs `1/(n_wl × n_rows)`)*: mathematically equivalent up to uniform rescaling. Test patch was a no-op (~6% noise).
2. *Restart budget (5 → 20)*: no improvement; if anything slightly worse seed.
3. *Aperture-position origin (Keck-I at 0 vs centroid-referenced ENU)*: centroid recentering made nullsim 10× *worse*. Opposite of what the agent's read of KILO's `_geodetic_to_enu` predicted.
4. *Per-mode field non-uniformity*: verified at chip-opt time `_require_field`'s output is uniform 0.7958 across all 24 modes. Earlier agent claim of std=0.56 came from reading post-chip state.
5. *Single-λ bootstrap on/off*: KILO's `optimize_null_broadband` does NOT bootstrap by default (gated by `KILO_BB_SINGLE_LAMBDA_BOOTSTRAP`). Disabling nullsim's bootstrap made the chip **worse** (1.25e-4 → 1.70e-4), so the bootstrap is helping nullsim, not hurting. Bootstrap is not the divergence.
6. *BFGS options*: both codes call scipy `minimize(method="BFGS", options={"maxiter": ...})`. KILO's `KILO_BB_GTOL` env knob is opt-in and undefaulted. Identical scipy defaults (gtol = 1e-5).

**What this means for parity:** realized parity is in hand. `realized_null_depth_kilo`
matches KILO `null_val_dark` within 1.5×; `null_instability` within 7%. The
remaining sub-mas detection-space gap traces to this chip-optimizer depth
delta but doesn't propagate proportionally because realized null is
AO+disk-floor-dominated in the J=4 regime.

**Suggested next moves (not done in this pass):**
- Wire a `chip_opt_params_override` parameter on the chip_optimization stage that bypasses BFGS and pins a pre-trained chip from `.npz`. Quick workaround.
- Instrument both codes to dump per-restart loss trajectories side-by-side. Slowest but most diagnostic.

**Also delivered this session (committed under separate fragments):**
- `examples/kilo_keck.toml:422` zeroed `fiber_phase_noise_rms_rad`. The 0.01 rad default was double-counting the FT-servo residual (already in `ao_piston_opd_rms_per_sub_m`). Realized null at J=4 dropped from 1.80e-4 → 1.32e-4. Null instability moved 0.115 → 0.154 (closer to KILO 0.166).
- `chip_diagnostics.py` broadband-null panel now uses a clean nominal input field, not post-chip output. Title now matches what's plotted.
- `output.dir` template gained `{timestamp}` (default `results/<name>/YYYYMMDD-HHMMSS`); `latest` symlink continues to repoint.
