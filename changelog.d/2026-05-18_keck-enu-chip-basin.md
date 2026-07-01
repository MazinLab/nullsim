---
date: 2026-05-18
---
### KILO Keck ENU geometry for chip optimizer parity

Updated `examples/kilo_keck.toml` to use KILO's geodetic-to-local-ENU Keck I/II
coordinates (`[-26.825864, -34.002224]`, `[26.825864, 34.002224]`) instead of
the previous east-west `[[0, 0], [86.6, 0]]` surrogate.

The kernel-null chip optimizer builds derivative rows directly from the
sub-aperture x/y coordinates, so the one-axis surrogate changed the objective
basis even though the optimizer code matched KILO. A bounded M=24 comparison
with identical field scale, seed, backend, and 250 BFGS iterations reached
mean null `3.41e-3` with KILO geometry vs `8.62e-3` with the east-west
surrogate.
