---
date: 2026-05-16
---

## Post-§15 gap-closure pass (ce63606, 7ccf3c0, 9d2233d)

Three structural gaps documented during the §15.5–§15.7 parity work are now closed; the remaining three are honest fidelity gaps with documented physics root causes. Test count moves from 494+21xfail to 512+14xfail (16 fewer xfails).

**Gap A — Multi-aperture pipeline support (ce63606):**
`telescope.fixed_aperture` gains `n_apertures ≥ 1` with `array_layout` choices (`linear`, `y_array`, `manual`) and populates `geometry.aperture_positions_enu_m`. `injection.ideal` broadcasts the field to `(n_apertures, n_wl, n_pol)` with an optional `n_modes` override. `nullsim/pipeline/registry.py` passes non-Mapping params through `stage_params_from_spec` correctly. Example configs updated to use multi-aperture layout where applicable.

**Gap B — chip_optimization → chip wiring (7ccf3c0):**
`chip.kernel_mzi_mesh` `Params` now accepts either an explicit `params: tuple` or `optimization_source: str`; `"auto"` scans `state.results.components` for the first `chip_optimization.*` entry. MKID multi-pixel saturation arithmetic was already present; example configs now exercise it with realistic pixel counts.

**Gap D — Optimizer loss and SNR partition alignment to KILO convention (9d2233d):**
`chip_optimization` backends compute loss as `mean(dark) / mean(bright)`, matching the detector convention. The scipy backend gains a 16-seed restart bank with target-null early-stop. `performance.standard` SNR, integration time, and contrast curve now use per-dark-port rates matching KILO's `signal_to_noise` convention; new `leakage_rate_hz_total` debug field added.

**Remaining honest gaps (documented in parity tests):**
- KILO `snr_jmag8_3600s_100mas` / `integration_time_5sigma_jmag10_s`: nullsim's optimizer reaches ~8e-7 raw null (10× deeper than the KILO design-spec ~1e-5 used to extract the reference SNR=68). The metric is now planet-shot-noise-limited; closes when the KILO reference is re-extracted at nullsim's achieved null depth.
- NAYRA `null_depth_raw` / `null_depth_calibrated` / `snr_jmag8_3600s_200mas` / `integration_time_5sigma_jmag10_s`: chromatic OPD from `transport.smf28` mode-1 dispersion precludes a broadband null on the single-aperture broadcast input. Closes when a sub-aperture sampler provides chromatically-matched modes.
- `total_rms_opd_nm` (both KILO and NAYRA): nullsim sums all OPD contributors in quadrature; both references expose only the temporal-only piston OPD. Closes once nullsim exposes a temporal-only rollup.
