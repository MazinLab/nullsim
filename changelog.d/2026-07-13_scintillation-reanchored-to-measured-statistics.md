### Scintillation model re-anchored to measured aperture-averaged statistics

The 2026-07-13 audit (NullSimTD task #44: literature cross-check plus a
first-principles Fresnel-propagation MC) found the KILO-ported scintillation
law wrong in three ways for meter-class sub-apertures, all inflating
amplitude noise:

- **Anchor ~30x too large.** `scintillation_index` claimed 0.041 as the
  Kornilov+2012 Q1 Maunakea value for a 5 m sub-aperture at zenith; the
  paper's measured S_2^2 = 0.77e-4 m^(7/3) gives sigma_I(5 m) = 1.3e-3.
  Default anchors updated everywhere (function defaults,
  `detection_curve.auto_scintillation_anchor_h5m_zenith`, and the
  `nayra_eelt.toml` example, which carried the same class of error at 0.05).
- **The lambda^(-7/12) factor removed.** The aperture-averaged index is
  wavelength-independent (Kenyon 2006 Eq. 7; Dravins 1998; Kornilov 2012);
  the old factor was a sigma-vs-sigma^2 exponent confusion on the
  point-source Rytov scaling. `wavelength_m` now only sets the Fresnel
  radius saturation guard.
- **Timescale was the point-source Fresnel crossing.**
  `scintillation_timescale_for_servo_design` returned r_F/v (~3.7 ms);
  aperture averaging removes exactly that fast structure, and the
  surviving fluctuations decorrelate on the aperture crossing time. Now
  returns 0.5 d_sub/v_high (measured ACF e-fold 0.47 d/v in the Fresnel
  MC; PSD knee at v/(pi d), Dravins 1998). Signature changes from
  `wavelength_m`/`scintillation_height_km` to `sub_diameter_m`.

Net effect for KILO Keck N=6 (d_sub = 3.14 m, J band, airmass 1.155):
open-loop sigma_I drops from 0.103 to 0.0046, and the servo-design
correlation time rises from 3.2 ms to ~45 ms, so `amplitude_servo_residual`
and the detection-curve amplitude terms shrink accordingly. The d^(-7/6)
and airmass^(3/2) scalings were correct and are unchanged, as is the 0.5
saturation cap below r_F.
