---
date: 2026-05-21
---
### Changed

- Tightened the HWO first-pass physics path: the 24-mode pupil surrogate can
  preserve the full 8 m collecting area while normalizing its maximum baseline
  to the primary diameter, and the HWO config now uses that mode.
- Removed the nonideal PM-fiber Jones/PER degradation from the photon-limited
  HWO baseline while retaining PM780 propagation loss and the dual-PBS-arm chip
  deployment.
- `performance.standard` now bounds sensitivity-stage null floors by the
  detector/calibration residual and uses sensitivity-stage planet throughput at
  the configured planet separation for headline SNR and integration time.
- Added an I-band Vega zero point and opt-in detector-side J-to-band color
  conversion so the 0.8 um HWO config no longer uses the Y/J photometry
  placeholder; KILO/NAYRA detector configs retain the legacy direct-J behavior.
