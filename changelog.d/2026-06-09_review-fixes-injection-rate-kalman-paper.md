---
date: 2026-06-09
---
### Characterization stellar-rate parity: injection coupling now folded into rate reconstruction

`injection.ideal` and `injection.single_mode_fiber` carry their fiber-coupling
loss in the field amplitudes (`E = √(η) · …`), recording the coupling spectrum
as a throughput component for provenance. Those components are now flagged
`field_amplitude_applied = True`. Stages that reconstruct an absolute stellar
rate from the throughput budget (`characterization.spectro`) fold the coupling
back in; the detector path is unaffected (it reads the top-level
`transmission_fraction`, which excludes the coupling, and gets it from
`|field|²`).

Previously the characterization stellar rate omitted injection coupling,
inflating `detected_stellar_rate_hz` — and therefore characterization SNR — by
`√(1/η_inj)`. Because `η_inj` tracks AO Strehl, the bias grew toward faint
guide stars (≈1.6× rate error at J=4 rising to ≈24× at J=12 for the KILO Keck
N=6 quadrature config), making faint-end characterization curves optimistic.

### Temporal-Kalman piston: unobserved modes carry full process variance

`temporal_kalman_piston_covariance` no longer skips modes with a zero
measurement eigenvalue. An unobserved differential-piston mode is uncorrected,
so its one-step prediction now carries the full process variance
`prior_variance · (1 − ρ²)` instead of zero. The common-piston mode (zero
baseline) is still excluded. Dormant for full-rank trackers; corrects an
optimistic piston floor for any rank-deficient (sparse-bright-port) tracker.

### Paper bundle fixes

* The PDI survey table now sorts by the SNR metric it displays
  (`snr_pdi_continuous_4h`), with non-finite/missing values sunk to the bottom,
  so the top-of-table LaTeX/JSON selections are consistent and NaN rows no
  longer displace finite ones. The full CSV/JSON exports still retain every
  planet.
* `null_transmission_map` `.npz` archives now also save display-ready
  (East-left) `design_map_display` / `band_averaged_map_display` arrays
  consistent with `display_x_mas` and the plotted PDF; the raw maps remain in
  native (RA-increasing) orientation paired with `l_coords_mas`. A naive
  re-plot from the archive no longer mirrors the figure.
* Removed the dead `detection_space_cache.json` sidecar request from the paper
  `detection_space` product (the plot never wrote it; it was recorded as
  skipped in `paper_manifest.json`).
