---
date: 2026-11-07
---
### Fixed: diffuse backgrounds use the single-mode étendue and the deployed chip

Review 2026-11-07 Issues 2, 11, and 12 and rows 40 and 42. One convention now
covers all three background stages. A single-mode input collects
`B λ² η_d` per spectral bin from a total-intensity surface brightness B, over
both polarizations (antenna theorem). Here `η_d = 1 − exp(−2ρ²) = 0.919` is the
share of the back-propagated Gaussian mode that the circular aperture passes,
at the Ruilier-Cassaing optimum ρ = 1.1209. Each PBS arm gets half.

Diffuse light carries no point-source coupling η, so the bare 0.8145 no longer
applies to it, and neither does η times Strehl. Output j of the chip receives
`Σ_k |(U S)_jk|²` of one physical input's share. U is the deployed chip stack,
per wavelength bin and per PBS arm. S is the `injection.mode_splitter`
isometry.

- `background.space_zodi` (Issue 2). (a) Local zodi is routed through the chip
  unitary instead of being scaled by the planet throughput at the planet's
  separation. Exozodi now follows a profile: 22 mag/arcsec² (V) per zodi at
  1 AU, falling as r⁻² outside the dust sublimation radius (Stark et al.
  2014). Its effective brightness is that profile averaged with the
  position-angle-averaged transmission T(r) of the 1.12 m pupil modes as the
  weight, never the 8 m primary. (b) The étendue is η_d λ² per input, not
  (λ/D_primary)² divided over the dark ports. (c) The exozodi brightness uses
  the host's absolute magnitude, `M = m − 5 log10(d / 10 pc)`. (d) The second
  lookup of the SMF coupling is gone. `routing = "scene_planet_throughput"`
  and `"dark_fraction"` still load but log a deprecation warning. Both route
  like the new default `"chip_unitary"`. `aperture_area_lam_over_d2` is
  removed. The stage now needs `scene.star_distance_pc` whenever the exozodi
  level is nonzero.
- `background.sky_surface_brightness` and `background.nayra_sky` (Issue 11).
  The uniform `n_in / n_out` splitter share is replaced by the routed share.
  The dark-port keys carry the mean over the chip's dark rows. The spectro-port
  keys carry the mean over all outputs, because the characterization stages
  retune the spectro rows downstream. New keys: `sky_routing_fraction_per_port`,
  `sky_rate_hz_per_output_port`, and `sky_rate_hz_per_input_mode`. The default
  `n_subapertures` now counts the physical inputs before the splitter.
- `n_polarizations` defaults to 1 (Issue 12). The old default of 2 doubled the
  sky in `kilo_keck.toml`, `kilo_maunakea.toml`, and
  `nayra_eelt_baseline.toml`.
- The étendue spaxel is evaluated at each bin's wavelength (row 40).
  `spaxel_arcsec2` stays the band-center value. `spaxel_arcsec2_per_bin` is new.
- `injection.single_mode_fiber` marks its coupling `point_source_coupling_only`,
  as `injection.ideal` does, so it no longer debits diffuse light (row 42).

Measured at the background stage, first sweep cell (old → new, Hz per dark
port):

- `hwo_space` (contrast 1e-10 cell, same 5.95 zodi draw): local zodi
  1.19e-40 → 0.0196 and exozodi 9.8e-39 → 2.07. The exozodi surface
  brightness at 1 AU goes from 19.75 to 21.26 mag/arcsec². Its T(r)-weighted
  profile factor is 5.1, because the r⁻² dust inside 1 AU falls within the
  1.12 m mode beam. HWO becomes zodi limited.
- `kilo_maunakea` (J = 5): sky 31.6 → 22.1. That is 0.5 (n_pol) × 0.919 (η_d)
  × 1.54 (the SMF Strehl × η coupling no longer debits the sky).
- `kilo_keck` (J = 4): sky 8.97 → 3.75. The étendue gives 0.46 and the routed
  dark-port mean gives 0.456 against the old uniform 0.5. Per-port routing
  runs from 0.23 to 0.87.

`background.space_zodi` also publishes the unrouted, band-integrated surface
brightnesses `local_zodi_rate_hz_per_m2_per_arcsec2` and
`exozodi_rate_hz_per_m2_per_arcsec2_at_scene_separation`
(photons/s/m²/arcsec² at the telescope entrance).
