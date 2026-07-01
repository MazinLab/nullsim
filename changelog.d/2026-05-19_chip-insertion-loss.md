---
date: 2026-05-19
---
### Photonic-chip insertion loss

Both built-in chip stages (`chip.kernel_mzi_mesh` and `chip.identity`) gain
a new `insertion_loss_db: float` parameter (default `-2.0`, validated
`≤ 0`) modeling wavelength-flat fabrication insertion loss (waveguide
propagation, fiber-to-chip coupling, MMI/MZI internal losses).

The loss is applied as a scalar `t = 10**(insertion_loss_db / 10)` on
field amplitudes — `new_amplitudes = (U·field) × sqrt(t)` — **after**
the unitary. Cumulative `state.throughput.transmission_fraction` is
*not* multiplied by `t`: per the detector convention at
`stages/detector/mkid.py:_port_rate_hz_per_wavelength`, chip losses are
modeled in field amplitudes and the detector forms photon rates from
`|field|² × transmission_fraction`, so multiplying both channels would
produce a `t²` double-count. The chip-stage throughput component still
records the `t` factor and the literal `insertion_loss_db` so the
loss-budget plot and debug report attribute the loss correctly,
replacing the misleading "loss=1.0 by construction" payload.

Because `chip_optimization.*` runs strictly before the chip stage and
consumes only pre-chip `field` / `throughput`, the optimizer always
operates on lossless inputs and is unaffected by this parameter. A
scalar post-unitary loss also commutes with the optimizer's ratio-based
objectives (broadband null, throughput penalty), so the optimal unitary
is mathematically independent of `insertion_loss_db`.

**Example-config strategy.** `kilo_keck.toml` ships with the realistic
`-2.0` dB value. `kilo_maunakea.toml` and `nayra_eelt.toml` inherit the
same package default — adding any field to those configs would shift
the resolved-config hash and re-seed the upstream stochastic stages
(atmosphere / `transport.smf28` phase noise), moving the chip-
optimizer's local optimum off the cached references, so the cleanest
way to keep their bit-for-bit upstream determinism is to *not* add the
pin. The `-2 dB` scalar is ratio-preserving on the chip's null depth
(amplitudes scale uniformly), and the absolute-throughput / SNR /
integration-time shifts (≈ 0.79× on amplitude, ≈ 0.63× on power, ≈
1.59× on integration time) sit inside the loose tolerances of those
reference fixtures.
