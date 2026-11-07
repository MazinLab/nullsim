---
date: 2026-11-07
---
### Fixed: characterization spectro sky goes through the retuned chip

The spectro-port sky that background stages publish is the mean over all chip
outputs. A retuned spectro row can add the coherent splitter copies of one
input in phase, so it can collect more than that mean. In a two input, four
copy example, 100 Hz per input became 100 Hz in one output against a published
50 Hz (review 2026-11-07 Issue 11, characterization half).

`characterization.spectro` and `characterization.detection_curve` now route
the per-input sky through their own retuned stack,
`sum_k |(U_char S)_{s,k}|^2 P_k` per wavelength bin, with the background
stage's `_diffuse_port_routing` and splitter isometry. The curve stage does
this per anchor and publishes `sky_rate_hz_per_spectro_port_per_anchor`. The
scalars are anchor medians. The per-bin input rate is recovered from the
background's published per-bin dark sky and routing table. Additive thermal
and exozodi terms carry over, and arm counting is unchanged. A background
payload without `sky_rate_hz_per_input_mode`, such as `background.space_zodi`,
keeps its published value.

### Fixed: characterization epsilon includes the setpoint bias cross term

The spectro floor carried its share of the setpoint and dither leakage, but
the leakage RMS that sets epsilon left out how that static phase offset
couples to the random piston. The RMS now adds
`2 (sigma_phase^2 / L_phase,dyn) L_b`. The phase statistics come from a
phase-only rerun of the retuned MC, with scintillation off and the same draws.
`L_b` is the spectro ports' share of `null_setpoint_leakage +
setpoint_dither_leakage`. See `tests/physics/notes/sensitivity_null.md`.

### Fixed: characterization reads the newest upstream sensitivity and MKID payloads

Both component lookups in `characterization.spectro` now search newest first,
in line with the review 2026-11-07 Issue 25 resolution rule.

None of the HDC smoke configs or `mkid_native_spectro` reach these paths, and
their numbers do not move.
