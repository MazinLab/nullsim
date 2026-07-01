---
date: 2026-05-20
---
### Site, sky, and band-factory layer

Added `background.sky_surface_brightness`, a generic site-profile sky
background stage that shares the downstream payload contract of
`background.nayra_sky` while supporting Cerro Armazones and Maunakea defaults.

Added built-in Cerro Armazones/Maunakea site presets, E-ELT/Keck telescope
presets, direct `grid.*` sweep support, and config-factory helpers for
Y/J/H/K band grids. The NAYRA example now declares its site/telescope presets
and uses the generic sky-background stage.
