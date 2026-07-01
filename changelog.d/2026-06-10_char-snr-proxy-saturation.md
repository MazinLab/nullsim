---
date: 2026-06-10
---
### Characterization SNR proxy saturates at the null floor

The characterization optimizer's in-loop SNR proxy rewarded static stellar
leak as `sqrt(null_floor / leak)` with the clamp at `null_floor * 1e-12`,
so driving leak three decades below the floor multiplied the reward ~30x
even though the deployed leak is pinned at the realized piston floor
(~3e-6). The optimizer traded planet throughput for un-cashable null
depth: wide characterization-curve anchors landed at 0.04-0.19 spectro
throughput with ~1e-9 static leak. The proxy now clamps at `null_floor`
(matching the restart-selection score), so below-floor leak earns nothing
and the remaining gradient goes to throughput. kilo_keck_6 J=4 anchors:
17.7 mas throughput 0.042 -> 0.409, 31.2 mas 0.072 -> 0.701; contrast
improves 4-7x over 15-100 mas, putting char mode decisively below
detection mode where it should be.

`examples/kilo_keck_6.toml`: the `characterization` spectro stage now uses
the full 20-bin broadband objective (`broadband_n_wavelengths = 20`). The
old single-design-bin economy produced chips with ~5e-4 mean broadband
leak (restart selection evaluated the full grid but only ever saw
chromatically optimized candidates); headline `char_snr` at J=4 rises
12.9 -> 134.6 and `char_leakage_rate_hz` falls 345 kHz -> 3.6 kHz.
