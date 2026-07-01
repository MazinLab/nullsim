---
date: 2026-06-09
---
### Phase B controls: photon-Kalman FT, setpoint trim, and NSC

`nullsim.physics.fringe_tracking` now includes the Phase B photon-counting
control primitives: `PhotonKalmanResidual`, `piston_structure_function_m2`,
`fusion_effective_broadband_rms_m`, and `photon_kalman_residual_budget`. The
budget stays frequency-domain: it trades continuous-time photon-estimation
noise against structure-function prediction error plus tracked-line residual
floors, without introducing a time-domain simulator.

`fringe_tracking.closed_loop` gained `control_model = "photon_kalman"`. In
that mode the stage derives an equivalent `f_3db_hz` from the photon-Kalman
window, reports the matching `f_sample_hz`, and publishes the estimation,
prediction, and line-floor decomposition alongside the existing IID/common
piston residual split.

`sensitivity.detection_curve` now charges dark-port setpoint trim leakage via
`null_setpoint_residual_nm` and `setpoint_dither_rms_nm`, and publishes the
resulting leakage terms. `performance.standard`,
`characterization.spectro`, and `characterization.detection_curve` now accept
`nsc_systematic_reduction`, which reduces only the null-instability systematic
term; mean-leakage photon noise is unchanged.
