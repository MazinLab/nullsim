---
date: 2026-05-16
---

### Chip-optimizer broadband_refine fast-path (commits 70b5068, bba89f8)

- `ChipOptimizationParams` gains `broadband_refine` (bool, default `True` for
  back-compat). When `False`, the torch backend skips the broadband restart loop
  and uses the single-wavelength bootstrap warm-start as the final chip params.
  The broadband loop was minimizing per-wavelength rescaled-unitary leakages that
  the monochromatic kernel never deploys, empirically drifting from the
  bootstrap's deep basin (chip_ideal floor 3–9e-4 vs bootstrap-reachable ~1e-5).
  Raises `ValueError` if the bootstrap was not run.
- `examples/kilo_keck.toml` sets `broadband_refine=false` and
  `bootstrap_n_restarts=30`. End-to-end chip_ideal drops from ~3–9e-4 to
  1.7–5e-6, below KILO's reference 7.3e-5.

### scene.point_source star_angular_diameter_mas wiring (commit a91c6c2)

- `scene.point_source.Params` gains `star_angular_diameter_mas` (`float | None`,
  default `None`). The field was previously parsed by the config schema but
  silently dropped in `stage_materialize.py`, leaving
  `state.scene.star_angular_diameter_mas` unset and forcing
  `sensitivity.detection_curve` to compute `disk_floor=0` regardless of TOML.
- `stage_materialize.py` now maps `scene.star.angular_diameter_mas` →
  `scene.point_source.star_angular_diameter_mas`.

### pyramid_wfs natural-guide-star R-mag (commit c181979)

- `ao.pyramid_wfs.Params` gains `derive_r_mag_from_scene_jmag` (bool, default
  `False`) and `r_minus_j_offset` (float, default `1.0`). When the flag is set,
  the WFS noise term derives `r_mag = scene.star.jmag + r_minus_j_offset`
  instead of using the static `guide_star_R_mag`. Enables natural-guide-star
  configs (where the host star is also the WFS reference) to have AO performance
  vary correctly across a J-magnitude sweep.
- `PyramidWFS.consumes` now includes `"scene"`. The effective R-mag used is
  recorded in the wavefront component payload.
- `examples/kilo_keck.toml` sets `derive_r_mag_from_scene_jmag=true`,
  `r_minus_j_offset=1.0`.
