---
date: 2026-06-12
---
### Paper bundle: per-planet achievable R and characterization survey table

- `char_max_R` in the paper bundle exoplanet survey is now computed per
  planet: the largest resolving power at which every Nyquist sampled
  spectral bin keeps SNR >= 3 (physics `max_spectral_resolution`, capped
  at the R = 1e4 AWG limit), evaluated through both the
  curve-interpolated and scalar-retune characterization paths.
  Previously the column copied the scene-level `max_spectral_resolution`
  into every row.
- New `characterization_table.tex` bundle product: the ten highest
  characterization SNR planets per class with SNR and achievable R,
  plus a `top_by_category_char` ranking in
  `exoplanet_table_results.json`. The generated `exoplanet_table.tex`
  and `pdi_exoplanet_table.tex` captions now carry the final paper
  caption text (with labels) instead of placeholders.
- `detection_space` plot cleanup: in-figure design-context line and the
  bottom margin caption are removed (the paper figure caption carries
  that information); y-axis floor raised 1e-11 -> 1e-10.
