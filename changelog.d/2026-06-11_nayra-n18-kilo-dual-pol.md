---
date: 2026-06-11
---
### NAYRA N=18 quadrature modernization; dual-pol diagnostics; KILO dual-pol chain

- examples/nayra_eelt.toml: overhauled from the legacy N=48 classic-mesh
  config to N=18 on the modern architecture — duplicated-input quadrature
  mesh (18 physical modes -> 36 Clements inputs, 18 tracker + 17 dark +
  1 spectro), dual-pol PBS front end with `deployment_mode="per_pol_arm"`,
  photometric amplitude pickoff, photon-Kalman fringe tracking (no
  inter-telescope/vibration terms: single bench), `[instrument]`
  materialization replacing the ~25 zipped port-count sweep axes, full
  20-bin compiled characterization objectives, and the modern
  evidence-based values (predictive gain x3, amp servo gain 3.0, chip
  insertion loss -2 dB, auto setpoint trim). The unused
  `calibration.floor_lookup` stage is dropped.
- Single-aperture piston routing: the photon-Kalman FT charges tracker
  measurement noise to a rank-1 inter-telescope common mode, so
  `sensitivity.detection_curve` with `ao_piston_residual_source =
  "opd_budget"` raises on n_apertures = 1. NAYRA sources its realized
  piston covariance from the bright-port CRLB instead
  (`ao_piston_residual_source="fringe_observability"` with the
  temporal-Kalman model; diagnostics stage moved before sensitivity).
  Any single-aperture config must follow the same pattern.
- diagnostics.fringe_observability: dual-pol support. The chip-input
  recovery uses the dominant-power PBS arm's coherent response (the arms
  share the design unitary with small per-arm trims, so fractional piston
  responses agree to first order); detector rate rows are arm-summed and
  per-arm port labels collapsed to physical chip ports, driven by the
  detector-published `n_pol_arms`. Count budgets therefore include both
  arms. Tests pin dual-pol == arm-summed single-pol CRLB.
- examples/kilo_keck_6.toml: the canonical KILO config now carries the
  physical dual-polarization chain (HWP + PBS + PM980 Jones propagation,
  chip deployed per PBS arm) with the hardware values from KILO
  PBSConfig/HWPConfig/PMFiberConfig. The original KILO code's
  `PolarizationMode` defaulted to SCALAR_LEGACY, so the paperv3 products
  and the parity parent `kilo_keck.toml` remain scalar; kilo_keck_6
  results now include PBS/HWP losses and polarization leakage terms and
  will not match the scalar parity targets exactly.
- fringe_tracking.closed_loop: the tracker photon budget pools the
  per-arm detector channels of each physical bright port (both arms see
  the same piston fringe) instead of treating each channel as its own
  estimator — per-channel division doubled the assumed shot variance in
  dual-pol pipelines. The per-port SNSPD capacity scales with the arm
  count (one tracker detector set per arm chip). The detector
  `n_pol_arms` lookup is now the shared `stages._utils.detector_n_pol_arms`,
  reused by the fringe-observability diagnostic.
- background.sky_surface_brightness in dual-pol configs: per-port sky
  etendue set to `n_polarizations = 1` (each PBS-arm waveguide accepts a
  single polarization). Downstream consumers multiply the per-port rate
  by the per-arm detector channel count, so the scalar default of 2
  double-counted the sky background in dual-pol pipelines (both
  kilo_keck_6 and nayra_eelt; the legacy NAYRA N=48 dual-pol runs carried
  the same 2x sky overcount).
- Tracker detector capacity: `_SNSPD_MAX_RATE_HZ` corrected from 2.5e7 to
  the 1e8 cps device capability (matching the detector.snspd stage
  default); SNSPDs gang per port via `tracker_n_detectors_per_port`.
  NAYRA budgets 16 ganged SNSPDs per port per arm (3.2e9 cps/port) in
  both the FT budget and the fringe-observability channel cap — the old
  1e8 channel cap clipped 16/36 arm channels at J=3, and with the N=18
  piston floor coming from the tracker CRLB the bright cells were
  capacity-limited, not photon-limited (J=3 scored below J=5).
  kilo_keck_6's diagnostic channel cap is raised to 8e8 to match its
  4-SNSPD-per-arm FT budget. Dark/spectro ports remain ganged MKIDs
  (1e5 cps/pixel), splittable with photonic power dividers; their pools
  were not near saturation in either config.
