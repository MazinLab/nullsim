---
date: 2026-05-16
---
## Tier 0–4 code-review hardening pass (commits 9a857f9–770b0a8)

Full-sweep code review of 73 production files (~13.2k LOC) closed 43 of 78 flagged issues across five severity tiers. Summary of what changed:

**Tier 0 — silently-corrupted headline outputs:**
- `performance.standard`: contrast curve divided stellar counts by `planet_dark_throughput` — a category error that biased every published contrast number. Fixed to match KILO convention (`contrast = signal_counts / (star_rate · t)`).
- `performance.standard` + `fringe_tracking.closed_loop`: planet rate was per-dark-port but photon-noise variances were not scaled by `n_dark_ports`, so SNR did not scale as `sqrt(n_dark_ports)`. Fixed to sum signal across dark ports (KILO convention).
- `fringe_tracking.closed_loop`: FT photon SNR summed all ports; now masks to bright ports only.

**Tier 1 — reproducibility / `run_hash` identity:**
- `hashes.py` `runtime_python_version()`: returned `sys.version` (includes compile date, OS, compiler banner) — two machines running the same Python point release produced different `run_hash`. Changed to `f"{platform.python_implementation()} {platform.python_version()}"`.
- `pipeline/runner.py`: `_handle_run` applied `pickle.dumps` to already-canonical tuples, re-introducing pickle-protocol drift. Replaced with `canonical_digest`.
- `config/loader.py` + `cli.py`: `resolve_config()` computed `run_hash` without external deps; CLI computed a different value and overwrote the manifest. Library and CLI now derive from the same recipe.
- `pipeline/context.py`: `RunContext` gains `root_seed: int` field (carries `[run].seed` so per-stage RNG derivation matches SPEC §6.5 and the cache key is self-consistent).
- `config/schema.py`: new `ReproducibilityConfig` pydantic model for the `[run]` reproducibility block.

**Tier 2 — physics distortions:**
- `delay_lines/geometric.py`: per-aperture losses were averaged into a scalar and broadcast — physically wrong for unbalanced arrays. Now applied per-mode to `field.amplitudes` with the common-mode scalar kept in `throughput`.
- `ao/pyramid_wfs.py`, `physics/kolmogorov.py`, `stages/transport/smf28.py`: several sign-convention, dtype, and edge-case fixes for WFS noise variance, structure-function normalization, and common-mode phase fallback.
- `physics/fresnel.py`: coupling integral domain and normalisation fixes identified in review.

**Tier 3 — design issues + BUG-tagged cleanups:**
- `chip_optimization/`: all three backends now store `kernel_rescales_per_wavelength: bool` in their result payload so `chip.kernel_mzi_mesh` can apply the correct rescaling convention without guessing.
- `pipeline/cache.py`, `pipeline/runner.py`: stale-cache and missed-invalidation edge cases closed.
- `stages/detector/mkid.py`, `stages/detector/snspd.py`: port-accounting and saturation-arithmetic corrections.
- `stages/calibration/floor_lookup.py`: additional external-dependency declarations for derived lookup tables.
- `stages/chip/kernel_mzi_mesh.py`: reads `kernel_rescales_per_wavelength` from upstream optimizer payload.

**Tier 4 — IMPROVEMENT cleanups:**
- `catalogs/exoplanets.py`, `config/sweeps.py`, `outputs/plots/detection_space.py`, `outputs/styles.py`, `physics/clements.py`, `physics/pupil_geometry.py`, `cli.py`: naming, docstring, type-hint, and robustness improvements from the review's lower-priority findings.
