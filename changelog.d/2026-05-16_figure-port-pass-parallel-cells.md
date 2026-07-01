---
date: 2026-05-16
---

## Figure-port pass + parallel sweep cells (commits 40a9782–4ae96bc)

Four paperv3 figures ported as registered plot modules; sweep parallelism shipped; DM actuator default corrected.

**Perf: parallel sweep cells (40a9782):**
`nullsim/cli.py` now parallelizes independent sweep cells via `concurrent.futures.ProcessPoolExecutor`. New `[run] n_workers` TOML key (default: CPU count) and matching `--workers N` CLI flag. `nullsim/config/schema.py` updated with the `n_workers` field. `nullsim/stages/ao/pyramid_wfs.py` default `n_actuators` corrected from 3000 → 3025 (55², the smallest perfect square ≥ 3000; the validator already requires a perfect square).

**KILO matplotlib defaults (ef5b681):**
`nullsim/outputs/styles.py` now contains the KILO publication rcParams verbatim (font sizes, line widths, figure dimensions) rather than a stub.

**Fig 3 — subaperture layout (ef5b681):**
`nullsim/outputs/plots/subaperture_layout.py` — sub-aperture positions on the pupil (paperv3 Fig 3). Physics primitive `nullsim/physics/uv_coverage.py` added to compute per-baseline (u, v) pairs used by this and subsequent figures.

**Fig 5 — UV coverage grid (ef5b681):**
`nullsim/outputs/plots/uv_coverage_grid.py` — (u, v) coverage across hour angles for the array and observation (paperv3 Fig 5).

**Fig 7 — chip diagnostics (00b34d8):**
`nullsim/outputs/plots/chip_diagnostics.py` — MZI mesh transmission map and null-depth histogram (paperv3 Fig 7).

**Fig 10 — contrast curves multiband (4ae96bc):**
`nullsim/outputs/plots/contrast_curves_multiband.py` — contrast curve family across wavelength bands with planet population overlay (paperv3 Fig 10).

Test count: 570 → 594 (+24 tests covering the new plot modules and physics primitive).
