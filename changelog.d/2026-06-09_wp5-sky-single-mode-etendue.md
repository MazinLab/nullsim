---
date: 2026-06-09
---
### Sky background: single-mode etendue default; drop unconsumed exozodi

`background.sky_surface_brightness` now defaults the sky spaxel to the
single-mode receiver etendue `Omega = n_polarizations * lambda^2 / A_per_mode`
(new `n_polarizations: int = 2` param) instead of the fixed 0.005 arcsec^2
table, which under-counted the sky by ~4-9x for Keck-scale subapertures. The
per-mode collecting area is the value the stage already derives for the rate
computation, and the band-center wavelength sets `lambda`. An explicit
`spaxel_arcsec2` override still wins; the legacy `DEFAULT_SPAXEL_ARCSEC2` table
remains only as a documented fallback for when the per-mode area is
unavailable. The payload records `spaxel_source` and `n_polarizations` for
audit. `background.nayra_sky` is unchanged.

The unconsumed `[scene.exozodi]` block was removed from
`examples/kilo_keck.toml` (and thus from `kilo_keck_6.toml`, which extends it):
exozodi is only read by `background.space_zodi` on the HWO space path.
