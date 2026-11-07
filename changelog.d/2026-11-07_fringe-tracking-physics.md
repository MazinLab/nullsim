Fringe tracking and observability fixes from the 2026-11-07 physics review.

The inter-telescope von Karman piston (Issue 3) is now the Conan et al. (1995)
aperture-filtered differential piston, integrated over the von Karman phase
spectrum for the pair's baseline and aperture diameter. The old closed form was
the Kolmogorov structure function at B = L0. At r0 = 0.15 m, L0 = 25 m it gave
14.81 um; the saturated value is 2.354 um for points and 1.688 um for the
Keck-Keck pair (85 m, 10 m primaries), 77 times less variance.
`fringe_tracking.closed_loop` passes the horizontal pair baseline from
`geometry.aperture_positions_enu_m` and `geometry.diameter_m`, averaging over
pairs; `von_karman_differential_piston_rms_m` gains `baseline_m` and
`aperture_diameter_m` (defaults: decorrelated points, the upper bound). This
lowers the kilo_keck and kilo_keck_6 common differential piston residual,
most at the faint end (review replay: 130 to about 70 nm at J=12).

`use_full_piston_servo_residual` now defaults to true, and false raises a
`ValueError`. The legacy servo-lag model it selected had no r0 and an OPD
residual that scaled with wavelength (row 43). hdc_ccf_smoke,
hdc_ccf_systematics, hdc_detection_smoke, kilo_maunakea, and
mkid_native_spectro ran it by default and now use the full model.

Simultaneous servo bands count information-equivalent photons,
N_b (lambda_ref / lambda_b)^2 (row 44). The nayra_eelt J/H/K scale goes from
2.445 to 1.699; nayra_eelt_baseline's Y/J/H scale from 2.999 to 3.204. The
closed_loop payload adds `simultaneous_science_band_information_scale`.
`_resolve_n_bright_ports` counts logical ports, not polarization channels, so
the published `n_bright_ports` halves in dual-arm configs (row 45).

`diagnostics.fringe_observability` builds the tracker response from true
per-wavelength rates, never the MKID energy-redistributed spectrum (Issue 6;
the review measured the NAYRA J=5 CRLB at 0.187 nm before and 0.265 nm after).
QE and rates come from the newest detector component (Issue 25). With
dual-polarization arms, the piston Fisher information (SVD, CRLB, covariance,
linear Monte Carlo, channel sweep) sums each arm's own field and rates (row 49),
published as `piston_fisher_polarization_arms`. The nonlinear, Gauss-Newton,
acquisition, and amplitude diagnostics still model the dominant arm with
arm-summed counts.

Smaller fixes: the temporal Kalman effective mode baseline uses the
structure-function weighting, so apertures with zero coefficient no longer
count (row 46); the cophasing handoff converts phase with the count-weighted
harmonic-mean wavelength (row 48); the Y zero point is Hewett et al. (2006)
Table 7, 5.71e-9 W m^-2 um^-1 at 1.0305 um, down 3.4% (row 41).
