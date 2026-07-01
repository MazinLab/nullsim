---
date: 2026-05-20
---
### Dual-polarization optical stages

Added a `polarization` stage family with `polarization.hwp`,
`polarization.pbs_split`, and `polarization.pm_fiber`. These stages preserve the
existing `(n_modes, n_wavelength_bins, n_pol)` field contract while allowing
scalar unpolarized fields to split into two PBS arms and PM-fiber Jones
matrices to apply same-axis coupling with cross-polarized leakage treated as
loss.

The stages are registered through autoload and covered by focused stage tests.
