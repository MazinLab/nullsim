---
date: 2026-05-21
---
### Config

- Update `examples/hwo_space.toml` to model a single 8 m unobstructed HWO
  primary with a photon-noise-limited first-pass baseline.
- Materialize `[telescope.aperture].obscuration_fraction` into
  `injection.single_mode_fiber` so fiber coupling uses the same telescope
  source of truth as the aperture and ideal-injection stages.
- Add `telescope.pupil_modes` and wire HWO through `sensitivity.detection_curve`
  so detection-space plots use separation-dependent planet throughput instead
  of the flat performance fallback.
- Allow detection-space planet overlays to opt into all-sky observability for
  space telescopes instead of inheriting the Mauna Kea altitude cut.
- Give the HWO config an explicit 1-500 mas sensitivity grid and request all
  registered diagnostic plots for debugging.
- Move the first-pass HWO pupil surrogate from 2 modes to 12 modes and let
  `subaperture_layout` read `telescope.pupil_modes` when no AO stage exists.
- Use the torch chip optimizer for the 12-mode HWO exploratory run because
  scipy finite-difference gradients scale poorly at 144 chip parameters.
- Set the HWO default grid to 0.8 µm with 10% bandwidth and use a
  `2 bright + 10 dark` port split for the 12-mode surrogate.
- Model the HWO front end as a KILO/NAYRA-style dual-polarization path:
  PBS split, 2 m PM780 transport, PM-fiber Jones/PER loss, and two physical
  per-polarization chip deployments.
- Prevent duplicate Ruilier-Cassaing injection coupling when an
  `injection.ideal` seed stage is followed by `injection.single_mode_fiber`.
