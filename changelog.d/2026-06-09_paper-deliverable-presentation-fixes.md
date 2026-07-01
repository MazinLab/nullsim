---
date: 2026-06-09
---
### Paper-deliverable presentation fixes

Four follow-on fixes to how characterization and fringe-tracking results are
presented in the paper bundle:

* **Fringe-observability figure.** The summary panel now distinguishes the
  single-frame CRLB observability ("CRLB RMS (1 frame)" / "Trunc. CRLB
  (1 frame)") from the realized piston floor that actually enters the contrast
  curve ("Realized floor (temporal)"). The realized value is sourced from the
  `sensitivity.detection_curve` `ao_piston_residual` (covariance-resolved, then
  temporal-Kalman selected, then scalar-resolved). Previously the figure
  headlined the ~0.17 nm CRLB while the operative floor was the ~3 nm
  temporal-lag value, with nothing linking them.

* **Characterization throughput columns.** `char_spectro_throughput`
  (performance_summary) and `spectro_throughput` (characterization_summary) are
  renamed to `*_design_bin` to make explicit that they are the chip efficiency
  at the single design wavelength, and a `*_band_avg` column (mean of the
  per-wavelength planet throughput) is added — the bandwidth-consistent figure
  alongside the band-integrated characterization SNR.

* **Skipped characterization-curve cells.** When a run produces
  separation-resolved characterization curves for some cells but skips others
  (`skip_jmags`), the paper exoplanet survey no longer falls back to the
  single-scene scalar retune SNR for the skipped cells (which is not valid at
  arbitrary planet separations). Those planets keep their detection-mode rates
  but report NaN characterization SNR.

* **Host disk column.** `host_disk_mas` is renamed to `assumed_star_disk_mas`.
  The bundled catalog snapshot carries no per-host stellar radius, so the floor
  is computed for the single configured scene star and applied to every overlay
  planet; the column is the assumed reference disk, not each host's true
  angular diameter.
