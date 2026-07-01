---
date: 2026-06-12
---
### Paper bundle: survey table class labels match paper prose

- The LaTeX exoplanet survey and characterization survey tables now
  render planet classes with the paper's prose names: "Super Earth"
  (no hyphen) and "Ice Giant" (instead of "Neptune-like"). The mapping
  is display-only in `outputs/paper.py`; the versioned catalog
  classifier names, row ordering, and the survey results JSON keys are
  unchanged.
- The `detection_space` plot legend uses the same display names via
  `TYPE_DISPLAY_LABELS` in the planet-overlay module; marker colors and
  the KILO-1:1 internal classification are unchanged.
