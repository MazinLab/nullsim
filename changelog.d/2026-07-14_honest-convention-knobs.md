### Added

- Three honest-convention knobs (2026-07-14 adjudication with NullSimTD;
  all default to the historical optimistic conventions):
  - `injection.ideal` `chromatic_coupling_rolloff` (+
    `coupling_design_wavelength_um`): flat SMF coupling peaks at the
    design wavelength and rolls off toward the band edges as the
    pupil-plane mode radius (proportional to wavelength for a fixed
    fiber MFD) mismatches the aperture, instead of being pinned to the
    geometric limit at every wavelength.
  - `ao.pyramid_wfs` `subaperture_fill_fraction`: charges the Keck
    inter-segment gap loss inside each packed subaperture circle
    (true-gap fill 0.9964 for the 3.14 m KILO ring) instead of booking
    the full circular area.
  - `ao.pyramid_wfs` `wfs_delivered_flux_fraction`: attenuates the WFS
    photon budget for light the residual aberration scatters off the
    detector acceptance.
- `examples/kilo_keck_6_honest.toml`: kilo_keck_6 with all three honest
  conventions enabled, for budget-target comparison against the pinned
  baseline.
