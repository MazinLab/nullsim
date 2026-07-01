---
date: 2026-05-18
---
### Pipeline contracts, CI cleanup, MLX retirement, and chip_diagnostics correctness

**Per-instance `required_consumes()` hook** (`nullsim/pipeline/stage.py`): stages can
now declare consumed state keys that depend on instance params, not just
class-level `consumes`. `validate_dependencies` and `ContentCache.key_for` both
route through `consumed_state_keys()`. `chip.kernel_mzi_mesh` overrides it to
include `results` only when reading from an upstream chip_optimization payload.
`--no-cache` runs now skip cache-key computation entirely.

**`detector.ideal_counter` port layout**: preserves `state.results`, publishes a
`detector.ideal_counter` component via `unique_component_name`, and emits
`bright_0` + `dark_0` ports with proper per-port rates and integrated counts.
`fringe_tracking.closed_loop` now matches ports starting with `"bright"`, fixing
~5× SNR underestimate with the new port layout.

**chip_diagnostics correctness**: `_plot_broadband_null` now uses `optimal_u_stack`
from the torch backend (the actual deployed chromatic chip) instead of rebuilding
from `optimal_chip_params`. `null_idx` updated to cover dark + spectro ports
(19 indices) rather than dark only (18 indices).

**MLX backend retired**: `nullsim/stages/chip_optimization/mlx.py` deleted;
`nullsim[mlx]` pyproject extra and all MLX-aware test/bench code removed.
KILO production runs torch.compile on CPU at float64; MLX-Metal never reached
the chip nulls KILO produces.

**CI parity tests deselected by default**: `pytest` default marker filter set to
`-m 'not parity'`; secondary `parity` CI job removed. Parity tests remain in the
tree as KILO historical reference.
