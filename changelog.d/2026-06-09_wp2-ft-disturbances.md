---
date: 2026-06-09
---
### Fringe tracking: inter-telescope piston + vibration disturbances

`nullsim.physics.fringe_tracking` gains pure primitives for the uncorrected
inter-telescope differential piston and the post-servo vibration residual:
`von_karman_differential_piston_rms_m` (B -> L0 saturation, Absil et al. 2006 /
Linfield et al. 2001), `piston_psd_knee_hz` (outer-scale and aperture-averaging
cutoffs, Conan et al. 1995 / Kellerer & Tokovinin 2007),
`servo_residual_variance_flat_knee` (the closed-form 7/12 flat-knee / f^(-17/3)
rejection integral), and `vibration_residual_rms_nm` (per-line `(f/f_c)^2`
rejection plus a flat broadband floor). Derivations and the citation list live
in `tests/physics/notes/fringe_disturbances.md`.

`fringe_tracking.closed_loop` integrates these disturbances behind opt-in Params
(`inter_telescope_piston`, `vibration_line_frequencies_hz`,
`vibration_line_rms_nm`, `vibration_broadband_rms_nm`, `outer_scale_m`,
`servo_bandwidth_fraction`, `sample_rate_max_hz`). The stage now separates the
detector sample rate from the closed-loop 3 dB bandwidth
(`f_3db = servo_bandwidth_fraction * f_sample`), scales the photon residual by
the closed-loop noise bandwidth (reproducing the prior pi/2 shot term exactly at
`servo_bandwidth_fraction = 0.5`), and, when `servo_bw_hz` is unset, grid-scans
the sample rate (logspace 20 Hz .. `sample_rate_max_hz`, 241 points) over the
full residual budget. The disturbances are tracked as a common differential mode
(`residual_common_differential_nm`) separate from the per-mode IID residual
(`residual_iid_per_mode_nm`), with `outer_scale_m` materialized from
`site.atmosphere.L0_m`. `predictive_lag_reduction` default changed 0.05 -> 0.33.

### Realized-null MC: rank-1 inter-telescope common-mode piston covariance

`nullsim.physics.realized_null` gains `piston_covariance_with_common_mode_m2`,
a per-mode covariance with an IID AO diagonal plus a rank-1 inter-telescope mode
along the +/-1 telescope-sign direction. `sensitivity.detection_curve`'s
`opd_budget` piston path now reads the fringe-tracking IID / common split and,
when the common term is positive, routes a physical-mode covariance through the
existing realized-null covariance MC (mirroring the fringe-observability splitter
handling) so the common differential piston does not average down across modes.
