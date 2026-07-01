---
date: 2026-05-20
---
### AWG spectrograph path

Added AWG spectroscopy primitives for Nyquist bin counts, wavelength grids, and
per-bin photon-rate/SNR spectra with throughput and free-spectral-range losses.
The `characterization.spectro` stage now reports a nested rate-binned AWG
spectrum plus headline AWG scalars without changing its existing broadband SNR
or maximum-resolving-power outputs.

The KILO and NAYRA example configs now declare explicit high-resolution AWG
settings for their spectro characterization stages.
