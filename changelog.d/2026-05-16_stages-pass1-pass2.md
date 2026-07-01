---
date: 2026-05-16
---
## §15.3 Pass 1 — atmosphere + AO stages (commit a4ff99f)

- `nullsim/stages/atmosphere/kolmogorov_screens.py` — `KolmogorovScreens` stage:
  Kolmogorov phase PSD → r0, Strehl per wavelength, atmospheric piston OPD budget entry.
- `nullsim/stages/ao/pyramid_wfs.py` — `PyramidWFS` stage:
  Pyramid WFS noise model → residual fitting + temporal + WFS-noise phase variances;
  supersedes the open-loop atmospheric OPD component.
- `nullsim/pipeline/state.py` additions: `OPDComponent.superseded` flag;
  `Wavefront.r0_at_500nm_m` typed channel; `Scene.altitude_deg` and `Scene.azimuth_deg`
  typed fields.

## §15.3 Pass 2 — injection + transport + delay lines (commit b1f19fb)

- `nullsim/stages/injection/single_mode_fiber.py` — `SingleModeFiber` stage:
  Ruilier–Cassaing coupling efficiency vs Strehl, applies to `Field` amplitudes and
  `Throughput`.
- `nullsim/stages/transport/smf28.py` — `SMF28Transport` stage:
  SMF-28e attenuation + chromatic dispersion OPD; uses tabulated
  SMF-28e loss curve from `nullsim/data/`.
- `nullsim/stages/delay_lines/geometric.py` — `GeometricDelayLines` stage:
  KILO-style geometric delay line model with per-baseline fiber-length computation,
  loss budget, and max-delay validation.
- `nullsim/stages/_utils.py` — `unique_component_name()` shared helper: collision-safe
  naming for sub-budget contributions when the same stage type is instantiated twice.
- `Throughput.components` canonical shape adopted:
  `{wavelength_grid_um, transmission_fraction, +provenance}`.
