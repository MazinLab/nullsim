Cross-package completion of the 2026-11-07 physics review fixes.

Off-axis coupling (Issue 4) now enters every field with its sign. The planet
input field, planet throughput, imaging beams, characterization disk rows, and
the chip optimizer's disk samples use `angular_coupling_amplitude` instead of
the square root of `angular_coupling_factor`, so modes of unequal diameter past
the overlap's first zero combine correctly. Power-weight sites keep the factor.
The overlap quadrature now sizes its Gauss-Legendre rule from the largest
`theta D / lambda` in the call. The fixed 96 nodes aliased beyond s of about
100 (s = 120 gave -5.5e-3 against the true -7.8e-5); calls with s <= 48 are
unchanged. No shipped path reached that range.

`DIFFUSE_SINGLE_MODE_ACCEPTANCE` (1 - exp(-2 rho^2) = 0.919) moves from
`background.nayra_sky` to `physics.ruilier_cassaing`. A new test pins it by
reciprocity against the point-source overlap: A eta(0) times the integral of
a(theta)^2 over solid angle equals 0.918962, against 0.918964.

Component selection (Issue 25): the remaining first-match selectors take the
newest component. These are the detector lookups in sensitivity,
calibration.floor_lookup, performance.pdi, and imaging; the sensitivity lookup
in fringe observability; the delay-line diagnostics in sensitivity and the chip
optimizer; the amplitude-servo chip lookup; and the plot helper
`maybe_find_component`.

`fringe_tracking.closed_loop` publishes `inter_telescope_uncorrected_piston_rms_m`,
and the debug report shows that value instead of recomputing the saturated
point-aperture limit (2.354 um where the stage uses 1.688 um for Keck-Keck).

Config comments: kilo_keck_6's sky polarization note matches the new
n_polarizations default, hwo_space drops the deprecated zodi `routing` key and
describes the new routing, and the NAYRA configs give the 11.1 m obstruction.
