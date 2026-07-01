---
date: 2026-05-17
---
### Phase 9 — loss-budget silent NaN + test fraud cleanup

Two unrelated diagnostic bugs from the KILO parity audit (2026-05-17).

**Loss-budget plot silently zeroed all-NaN components.** The pre-Phase-9
code computed `finite = [v for v in vals if np.isfinite(v)]` and used
`np.median(finite)` when non-empty, falling through to `0.0` otherwise.
A broken stage that reported all-NaN transmission was indistinguishable
from a perfectly transmitting one — the worst-possible failure mode for
a diagnostic plot.

- `nullsim/outputs/plots/loss_budget.py` now (a) emits a
  `logging.warning` whenever a (component, band) cell has only
  non-finite values, and (b) renders an explicit hatched "n/a: <name>"
  marker stacked above the band's bar. The marker is drawn in the
  component's color so the legend mapping remains clear.
- New regression test in `tests/outputs/plots/test_loss_budget.py`
  asserts both the warning message and the on-axis marker for an
  all-NaN component.

**Test fraud: `test_restart_patience_disabled_runs_all_restarts`.** The
test name claimed "patience=0 disables early exit even when restarts
don't improve," but the only assertion was
`payload["optimal_chip_params"] is not None` — a single restart would
have passed. The other two audit-flagged tests
(`test_kernel_mzi_mesh_wavelength_rescaling_matches_clements` and
`test_kernel_mzi_mesh_optimization_source_auto_finds_upstream`) were
investigated and found to have genuine `np.testing.assert_allclose`
end-to-end checks; the `assert ... is not None` lines in those tests
are defensive guards before unwrapping `Optional` returns, not the
main assertion. Audit was inaccurate; no change required.

- `scipy.py` restart loop now counts `n_restarts_completed` in both
  the serial and parallel paths.
- `_write_results` accepts an optional `n_restarts_completed` and
  surfaces it on the payload.
- The fraud test is rewritten to assert
  `payload["n_restarts_completed"] == 3`. A companion
  `test_restart_patience_enabled_can_short_circuit` test pins the
  early-exit contract from the opposite side.
