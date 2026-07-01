---
date: 2026-05-18
---
### Timestamp output directories + chip_diagnostics fix + kilo_keck fiber noise correction

Three targeted fixes shipped in commits ec8d4f3, ce1d4a2, 0c5a1d1.

**Human-readable timestamp output directories (ec8d4f3):** `OutputConfig.dir` default
changed from `"results/{run.name}/{run_hash}"` to `"results/{run.name}/{timestamp}"`.
`ResolvedConfig` gains a `timestamp` field (wall-clock `YYYYMMDD-HHMMSS` stamp, frozen
at load time so CLI and runner resolve to the same path). Both `_run_dir` and
`_resolve_output_dir` now substitute `{timestamp}` alongside `{run_hash}`,
`{config_hash}`, and `{run.<key>}`. The `latest` symlink continues to repoint
after each run. `{run_hash}` remains fully supported for content-addressed directories.
**chip_diagnostics broadband-null panel corrected (ce1d4a2):** `_plot_broadband_null`
now evaluates the chip's intrinsic chromatic response against a clean nominal stellar
field (per-mode magnitudes from the design-wavelength reconstruction, phases stripped,
wavelength-flat) matching `chip_optimization._require_field`. Pre-fix the panel read
`state.field.amplitudes` (post-chip realization), conflating chip design quality with
AO/transport phase noise. Panel title updated to "Broadband null spectrum (chip only,
clean input)".

**kilo_keck fiber_phase_noise zeroed (0c5a1d1):** `examples/kilo_keck.toml` sets
`fiber_phase_noise_rms_rad = 0.0`. The prior 0.01 rad value double-counted the
fringe-tracker servo residual already present in `ao_piston_opd_rms_per_sub_m`.
Realized null at J=4 drops from 1.80e-4 → 1.32e-4; null instability 0.115 → 0.154
(KILO reference: 0.166).
