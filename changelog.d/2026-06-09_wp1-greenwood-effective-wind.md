---
date: 2026-06-09
---
### AO and fringe tracking: effective wind from tau0

Added the Greenwood/Fried `tau0 = 0.314 r0 / v_bar` helper and made
`ao.pyramid_wfs` derive its temporal-lag wind speed from `tau0_500nm_s` by
default. The AO payload now publishes `effective_wind_speed_m_s`, which
`fringe_tracking.closed_loop` uses for its Conan piston frequency when present.
