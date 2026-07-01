---
date: 2026-06-11
---
### HWO config modernization; Guyon 2006 ideal-coronagraph overlay

- examples/hwo_space.toml: adopted the `[instrument]` shared port topology
  (36 modes, 10 bright / 26 dark, no spectro) in place of port counts
  repeated across five stages, dropped the vestigial
  `calibration.floor_lookup` stage (the photon-noise baseline imposes no
  floor; performance.standard records the absent floor source), and removed
  the detector `integration_time_s` override now materialized from
  `[observation]`.
- New physics module `nullsim/physics/ideal_coronagraph.py`: the Guyon,
  Pluzhnik, Kuchner, Collins & Ridgway (2006, ApJS 167, 81) theoretically
  optimal coronagraph bound. Planet throughput is the complement of the
  off-axis source's projection onto the removed stellar subspace — the flat
  pupil mode at order 2 (`1 - |2 J1(x)/x|^2`, 50% point near 0.5 lambda/D),
  plus the tip/tilt modes at order 4 (`- |4 J2(x)/x|^2`, 50% point near
  1 lambda/D). Finite-star leakage is the same function area-averaged over
  the uniform disk (small-disk limits x^2/8 and x^4/192), and the 5-sigma
  contrast bound is photon noise on leaked starlight plus backgrounds.
- detection_space plot: opt-in `[output.detection_space]
  ideal_coronagraph_orders = [2, 4]` overlays the bound for the same star,
  aperture, zodi background, integration time, and SNR target as the run,
  but through perfect optics (full clear primary, mirror train, detector
  QE — none of the photonic chain's fiber/chip/polarization losses).
  Single-aperture configs only. hwo_space.toml enables orders 2 and 4; the
  order-2 bound is stellar-size walled on the 1.86 mas solar disk at 5 pc,
  which is the Guyon 2006 argument for the fourth-order null the HWO
  kernel projector enforces.
