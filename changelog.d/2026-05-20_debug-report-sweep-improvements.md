---
date: 2026-05-20
---
### Debug-report sweep improvements and null-depth card fixes

**Sweep PDF readability:**
- Compact coord labels: `_coord_dedupe` / `_shorten_coord_key` / `_short_coord_value`
  fold zipped axes and truncate long module-path keys to bare parameter names.
- Partition-equivalence deduplication: `_partition_signature` groups sweep cells
  that vary only on the same axis, collapsing redundant legend entries.
- `_sweep_trends_page` rewritten to plot one curve per unique N (sub-aperture
  count), with `_group_label_from_coord` generating concise group labels.
- `_sweep_contrast_page` gains a y-axis clip so outlier cells do not compress
  the interesting contrast range.

**Overview and detector-page null-depth card:**
- Renamed metric rows: "Null depth (raw)" → "Detector raw null",
  "Null depth (cal)" → "Final science null".
- `_first_finite_float` used throughout to resolve the effective null depth
  across calibration, performance, and sensitivity components without
  preferring a zero over a real value.
- `final_null` probes `perf["null_depth_effective"]`, `state.results.null_depth`,
  and `sens["effective_null_with_disk"]` in priority order.

**Other plot fixes:**
- `detection_space` and `contrast_curves_multiband` mask non-finite contrast
  values before plotting to avoid matplotlib NaN artifacts.
- `loss_budget` tolerates missing photon-rate components without raising.
