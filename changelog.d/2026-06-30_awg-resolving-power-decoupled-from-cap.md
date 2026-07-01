---
date: 2026-06-30
---
### characterization.spectro: AWG channel count decoupled from the max-R search cap

- `_awg_spectrograph_payload` no longer falls back to
  `resolving_power_cap` when `awg_resolving_power` is unset; it now uses a
  fixed `10_000.0` hardware default. `resolving_power_cap` is only the upper
  bound of the `max_spectral_resolution` search, so a deliberately uncapped
  search (e.g. `resolving_power_cap = 1e9`) previously inflated the AWG bin
  grid to ~10^8 bins and allocated tens of GB, OOM-killing CI runners. All
  shipped configs already set `awg_resolving_power` explicitly, so their
  behavior is unchanged.
- Added `test_awg_resolving_power_defaults_independently_of_resolving_power_cap`
  to pin the decoupling.
