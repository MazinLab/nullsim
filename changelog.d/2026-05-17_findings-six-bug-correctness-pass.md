---
date: 2026-05-17
---
### Six-bug correctness pass (commit 0e54bf7)

External code-review surfaced six bugs across four subsystems.

**[HIGH] chip_optimization._quantize_phases missing mod-wrap**
Round-to-nearest DAC quantization was not followed by `np.mod(q, 2π)`.
At off-design wavelengths the kernel rescales phi/alpha by
`wl_design/wl_target` before the MZI matrix, so a negative quantized phase
produces a different unitary than its wrapped-positive equivalent.
chip_optimization optimized a different chromatic chip than
`chip.kernel_mzi_mesh` evaluated (~0.47 mismatch at 1.10 µm, two-mode
mesh). Fix shared by scipy/torch/mlx backends. New regression tests pin
on-design and off-design parity.

**[MEDIUM] detector.snspd overcounted per-mode photon rates by ~2.24×**
`_resolve_collecting_area_m2` skipped `state.geometry.effective_collecting_area_m2`
and fell through to `collecting_area_m2`, violating the state.Geometry
contract. Sub-aperture pipelines (kilo_keck, N=12 packed) were affected.
Fixed to mirror `detector.mkid`'s three-step resolver.

**[MEDIUM] realized_null_depth_mc silently accepted negative n_realizations_per_azimuth**
Negative values hit the `n_per_az <= 0` (ideal/no-instability) branch
instead of raising. Now raises `ValueError`; `DetectionCurve.Params`
gains `ge=0` Pydantic constraint. `0` remains the explicit skip-MC toggle.

**[MEDIUM] _chip_input_stellar_rate_hz zeroed rate for single-wavelength grids**
`deltas=[wl[0] * 0.0]` zeroed the chip-input rate and caused a hard raise.
Fixed to `deltas=[wl[0]]` (100% fractional-bandwidth stand-in), mirroring
the detector.mkid/snspd single-bin convention.

**[LOW] ao.pyramid_wfs circular pupil template ignored central obstruction**
`_resolve_pupil_geometry` returned `None` for `pupil_template="circular"`,
placing sub-apertures as if the pupil were unobstructed. Now builds
`PupilGeometry(central_obstruction_diameter_m=…)`.

**[LOW] cli._resolve_output_dir surfaced format errors as raw AttributeError**
Now wraps in `NullsimConfigError`, matching `runner._run_dir` so `main()`'s
friendly "config error:" handler catches CLI-side typos in `output.dir`.

**Parity test extractor fixes**
`strehl_jband` switched to `state.wavefront.strehl_ratio`; `mean_coupling_jband`
and `resolution_mas_jband` XFAIL'd with documented root causes;
`test_kilo_maunakea_parity_metric[null_depth_calibrated]` XFAIL'd (definition
mismatch).
