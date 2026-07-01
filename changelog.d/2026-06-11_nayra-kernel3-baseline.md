---
date: 2026-06-11
---
### NAYRA detection chip adopts the KILO kernel-null recipe

- examples/nayra_eelt.toml chip_opt: `kernel_order` 0 -> 3, `disk_n_rings`
  1 -> 3, `disk_weight` 0.01 -> 1.0, matching the kilo_keck.toml production
  recipe. The N=18 J=3 A/B test (examples/test/nayra_eelt_kernel3_ab.toml,
  run 20260611-094326) showed the kernel_order=0 baseline was stellar-disk
  limited at the bright end: the higher-order kernel cut the disk floor
  1.21e-5 -> 4.9e-7 and the effective null 1.74e-5 -> 1.86e-6 for a 2x
  planet-throughput cost at 10 mas, a net 4.7x detection SNR gain.
