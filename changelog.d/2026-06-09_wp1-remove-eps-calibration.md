---
date: 2026-06-09
---
### Performance and characterization: remove hand-tuned eps calibration

Removed the `eps_calibration_residual` parameter from performance,
characterization, tests, and examples. Systematic eps is now the quadrature of
the physical null-instability term and the residual speckle-stability factor.
