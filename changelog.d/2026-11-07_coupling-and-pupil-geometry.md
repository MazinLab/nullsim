---
date: 2026-11-07
---
### Fixed: exact off-axis fiber coupling, ELT obstruction, and `pupil_modes` geometry

**Off-axis coupling (review 2026-11-07 Issue 4).** `angular_coupling_factor`
used KILO's Gaussian envelope `exp(-(pi w theta / lambda)^2)`. It now
evaluates the exact overlap of a tilted plane wave on the clear sub-aperture
with the same fiber mode used on axis (`rho = 1.1209`), a one-dimensional
Hankel integral done by Gauss-Legendre quadrature. The signature is
unchanged apart from an optional `central_obscuration_ratio` (default 0,
since packed sub-apertures are clear). The new `angular_coupling_amplitude`
returns the signed field ratio, which goes negative past the first zero at
`theta D / lambda = 1.4948`. For 3.14 m sub-apertures at 1.25 um, power
coupling at 10, 30, 60, 100, and 150 mas moves from 0.985, 0.872, 0.579,
0.220, and 0.033 to 0.971, 0.766, 0.323, 0.0205, and 0.0030. Every planet
throughput curve drops beyond about 0.5 lambda/D_sub. This is a deliberate
break from KILO parity, and `KILO_MODE_FIELD_RADIUS_TO_SUBAP_RADIUS_RATIO`
is gone.

**ELT central obstruction (Issue 7).** `EELT_OBSCURATION_FRAC` is now
11.1 / 39.146 (ESO's 11.1 m) instead of 0.11 (4.3 m). The
`eelt_telescope` preset now has `central_obstruction_diameter_m = 11.1` and
`obscuration_fraction = 0.283554`, so the fraction-driven area and coupling
terms agree with the packing. The NAYRA N = 18 packing moves from
d_sub 7.770 m and 853.4 m^2 to 7.133 m and 719.3 m^2. N = 48 moves from
4.616 m to 4.346 m, and its cold packing time doubles (68 s to 131 s).

**`telescope.pupil_modes` (Issues 22 and 1).** The stage now raises when the
upstream geometry has `n_apertures > 1`. Before, it silently replaced an
array's baselines with modes inside one pupil while keeping the array's
collecting area. It also publishes length-`n_modes` `mode_areas_m2`,
`mode_diameters_m`, and `mode_origin_positions_m` (the single telescope
origin). Readers therefore take the per-mode vector path instead of a
telescope's scalar, and sky projection keeps the pupil offsets rigid.

The four smoke configs that hit Issue 22 (`hdc_ccf_smoke`,
`hdc_ccf_systematics`, `hdc_detection_smoke`, and `mkid_native_spectro`) are
now one 10 m telescope with four pupil modes. They have no delay lines and
use the honest 54 m^2 of mode circles, not the array's 314 m^2. The mode
positions are unchanged. `hdc_ccf_smoke` goes from ccf_snr 70.22 against
band 387.41 to 126.23 against 690.28. `hdc_ccf_systematics` needed
`residual_speckle_stability_factor` raised from 1e-3 to 4e-3 to stay
systematics dominated: ccf_snr 53.50 against band 5.52, a variance ratio of
4.57. The `hdc_detection_smoke` `ccf_contrast_gain` goes from 28.199 to 6.186,
and from 0.626 to 0.163 without the knobs. These are plumbing numbers, not
sensitivity predictions.
