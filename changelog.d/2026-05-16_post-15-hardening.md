---
date: 2026-05-16
---

## Post-§15 hardening pass (commits 3dc8458–9eefbcc)

Six correctness bugs fixed from independent code review; new physics primitive ported from KILO.

**Bug fixes:**
- `injection.single_mode_fiber` + `transport.smf28` double-counted optical loss: both stages scaled field amplitude by `sqrt(loss)` AND multiplied `throughput.transmission_fraction` by the same loss, so the detector (computing `|field|² × throughput`) received rates wrong by `loss²`. Fixed asymmetrically to KILO convention: SMF coupling stays in field amplitude, transport loss in throughput.
- `cli._handle_run` computed `run_hash` from the pre-materialized resolved config but wrote `cells[0].config.config_hash` into the manifest — output directory and manifest disagreed about which config ran. Fixed by pinning all downstream state to `cells[0].config`.
- `chip_optimization/torch.py` + `chip_optimization/mlx.py` optimized in KILO's column-parity Clements ordering while `chip.kernel_mzi_mesh` reconstructed via the lexicographic `physics.clements` ordering — same params produced different unitaries. Fixed by adding `_canonicalize_clements_params` that decomposes the optimized unitary in the canonical ordering before storage.
- `calibration.floor_lookup` read its lookup-table file at runtime but did not declare it as an `external_dependencies()` entry — table edits reused the stale cache and the same `run_hash`. Fixed by adding `external_dependencies()` that hashes the file bytes.
- `exoplanets._autocompute_snapshot_hash` short-circuited when a hash was supplied, skipping byte-level verification. Now recomputes and raises on mismatch.
- `ChipOptimizationParams.n_dark_ports` accepted 0, producing NaN loss from an empty-slice `mean`. Tightened to `ge=1`.

**New physics primitive:**
- `nullsim/physics/pupil_geometry.py`: `PupilGeometry` dataclass, `optimize_subaperture_positions`, and `keck_pupil()` — a line-for-line port of KILO's `kilo.array` sub-aperture placement optimizer and `kilo.pupil` constraint helpers. `PyramidWFS` gains `n_subapertures_per_aperture` + `pupil_template` params that derive `piston_subaperture_diameter_m` from the actual pupil geometry (result cached via `lru_cache`).

**New example config:**
- `examples/kilo_keck.toml`: Keck-only paper-baseline (Keck I + Keck II, N=12 sub-apertures per telescope, 276-MZI Clements mesh). Declares pupil geometry instead of pinning a constant; derives d_sub = 1.929 m reproducing KILO's `keck_only_config` to <1 mm.

Test count: 555 passed + 3 xfailed (was 523 + 3; +32 tests for six regression fixes and the pupil-geometry primitive).
