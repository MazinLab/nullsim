---
date: 2026-06-10
---
### Photon-limited auto setpoint trim

The fixed 2.5 nm dark-port trim dither and 1.0 nm setpoint placeholders are
replaced by a derived joint optimum: the 1f lock-in signal scales as
dither x offset, so the held setpoint goes as 1/dither and the total added
leakage L(A) = G A^2/2 + n G (k/A)^2 minimizes at A_opt^4 = 2 n k^2 with
equal dither/bias partition (physics.fringe_tracking.photon_limited_
setpoint_trim). sensitivity.detection_curve gains auto_setpoint_trim /
trim_loop_bandwidth_hz / n_trim_modes and derives both terms from the
run's detected stellar rate and dark-port floor, so the faint-end budget
degrades honestly as 1/sqrt(flux). kilo_keck J=4: trim leakage 4.6e-5 ->
2.8e-6 (0.47 nm RMS dither), SNR back above the pre-realism baseline.
