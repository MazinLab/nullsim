---
date: 2026-05-17
---
### Phase 6 — bandpass photon normalization

Per the KILO parity audit (2026-05-17), nullsim's wavelength grid
generator used `np.linspace(start, stop, n_bins)` which places samples
AT the band edges. The detector stages then extrapolated half-bin
edges on top, adding one extra step to the total integrated bandwidth.
For 20 bins over 0.15 µm this inflated the photon-rate integral to
~0.158 µm — a 5% bias on every detection-space photon rate.

KILO detection-space integrates against the configured bandwidth
directly (`kilo/studies/natural_design.py` L1972).

- `nullsim/pipeline/state.py` grid generator now constructs bin
  EDGES via `np.linspace(start, stop, n_bins + 1)` and reports bin
  CENTERS as the midpoints. With the bin-center convention the
  detector's half-bin extrapolation recovers the configured total
  bandwidth exactly.
- Regression test in `tests/pipeline/test_state.py` pins both the
  bin-center locations and the end-to-end total integrated bandwidth
  for the canonical Keck J-band grid (1.25 µm, 0.15 µm, 20 bins).

`photon_rate_jband` xfail in `test_kilo_keck_paperv3_parity_metric`
left in place pending the end-to-end re-run; Phase 6 changes the
absolute rate by ~5% which propagates to every downstream metric.
