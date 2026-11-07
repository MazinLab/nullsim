### Fixed

- `sensitivity.detection_curve` and the chip-field plot helpers
  (`null_transmission_map`, `imaging_reconstruction`) no longer size the
  input modes from the oldest geometry component. Per-mode vectors win;
  otherwise the scalar `mode_area_m2` / `mode_diameter_m` (and the
  central obstruction ratio) come from the newest component. In
  hwo_space the 8 m primary from `telescope.fixed_aperture` was taken as
  the mode size instead of the 1.12 m `telescope.pupil_modes` modes the
  optimizer uses. Physics review 2026-11-07, Issue 1. hwo_space contrast
  1e-10 cell, cache off: sensitivity mode diameter 8.0 -> 1.120 m (now
  equal to the optimizer's), planet throughput 0.0191 -> 0.261 at 33 mas,
  0.0031 -> 0.761 at 50 mas, 7.8e-9 -> 0.564 at 100 mas, 8.8e-30 -> 0.119
  at 200 mas, and 2.1e-177 -> 7.9e-6 at 500 mas.
- When the closed-loop amplitude servo residual is capped at the
  open-loop scintillation, sensitivity injects the open-loop bandwidth
  `0.5 / tau_scint` with it instead of the servo bandwidth, and labels
  the source `closed_loop_capped_at_open_loop`. This lowers the
  amplitude-leg bandwidth in the systematic-noise `f_eff` for kilo_keck_6
  at J >= 6. Issue 5.
- Sensitivity's stellar normalizations (the published effective leakage
  rate, the static-chip leakage, and the setpoint-trim stellar rate) use
  the detector's uncapped `uncapped_port_rates`, not the saturation-capped
  `port_rates`. Issue 9 (sensitivity part).
- Sensitivity takes the newest `fringe_tracking.closed_loop_residual`,
  matching its common/IID split lookup. Issue 25 (sensitivity part).
- `transport.smf28` / `pm980` / `pm780` publish the per-mode GVD phase
  they apply, and `polarization.pm_fiber` the per-mode, per-arm Jones J00,
  as a field component `static_field_factor` of shape
  `(n_modes, n_wl, n_pol)`. Sensitivity multiplies its rebuilt stellar
  and planet fields by their product (mapped through
  `injection.mode_splitter`; relative to the design-bin phase when the
  chip trims each arm's inputs), so the realized null, its fluctuation,
  the planet throughput, and the disk floor see the fiber dispersion
  leakage and the PM-fiber amplitude imbalance the deployed chip sees.
  Issue 10, row 47. hwo_space realized null moves 2.96e-31 -> 3.09e-31;
  the KILO and NAYRA shifts were not measured here.
