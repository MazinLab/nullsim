---
date: 2026-05-17
---
### Debug-report sweep rework: per-cell PDFs + KILO parity overlay

`nullsim/outputs/plots/debug_report.py` reworked to support sweep runs.

**Per-cell output:** one PDF written under the cell's `run_dir`; a
back-compat mirror is written at the sweep root so single-cell callers
see no change.

**New per-cell pages:**
- Chip optimizer convergence + per-port leakage bars (log scale,
  bright/spectro/dark colored).
- Fringe-tracking σ² OPD breakdown bar chart (nm units, sourced from
  `results['fringe_tracking.closed_loop']`).
- Null-depth waterfall: chip_ideal → realized → disk → effective →
  calibrated, with a KILO badge per cell.

**New sweep PDF (`debug_report_sweep.pdf`):**
- Cover page with PASS/FAIL KILO parity badge (ASCII text).
- Cross-cell summary table (8 columns, fontsize 8, fits 7.5 in width).
- Headline trend panels: log-scale null curves + stacked SNR/σ_OPD/Strehl.
- Contrast page overlaying nullsim curves on the bundled KILO paperv3
  reference per matching J magnitude.

**Layout fixes:**
- Log tick labels render as "10^-4" plain text via `_log_formatter` +
  `LogLocator` (avoids mathtext parse errors when `text.parse_math=False`).
- Strehl panel switches to log scale when Strehl < 5%.
- Seen-set deduplication in `debug_report()` prevents duplicate PDFs
  when the primary cell's `run_dir` and sweep root coincide.

**Tests:**
- `test_debug_report_writes_per_cell_pdfs_and_sweep` — 3-cell
  multiplexing + sweep PDF size threshold.
- Existing `test_debug_report_writes_multipage_pdf` still pins the
  single-cell back-compat path.
