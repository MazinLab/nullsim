---
date: 2026-05-20
---
### Detection-mode disk null constraints

Added finite stellar-disk null constraints to the detection-mode chip
optimizers. `chip_optimization.scipy` and `chip_optimization.torch` now accept
`disk_n_rings`, `disk_n_azimuths`, and `disk_weight` and add L2-equalized
real/imag disk samples to the same dark+spectro null objective used by
`kernel_order`.

The NAYRA E-ELT example now pins `kernel_order = 2` and enables the same
three-ring, four-azimuth finite-disk sampling used by characterization mode.
