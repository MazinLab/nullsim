---
date: 2026-06-09
---
### Sensitivity publishes effective leakage rates alongside static-chip rates

`sensitivity.detection_curve` now records three new payload keys when an
upstream detector stage is present: `leakage_rate_hz_effective`
(`effective_null_with_disk` times the bright + dark + spectro stellar rate
total, the same normalization `performance.standard` uses for SNR),
`leakage_rate_hz_static_chip` (the detector dark-port total, i.e. the static
chip null), and `effective_to_static_leakage_ratio`. When no detector
component is present (standalone smoke tests) all three are `None`. The
`debug_report` detector/performance page now prints both leakage truths on one
line, labeling the realized value as the one SNR uses. This resolves the
two-truths artifact where the detector-stage dark rate and the realized
sensitivity leakage looked like a discrepancy.
