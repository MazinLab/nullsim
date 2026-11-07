### Paper survey tables and detection-space overlay fixes

Fixes from the 2026-11-07 physics review (Issues 14, 15, 16, the
detection-space part of 25, and rows 56 to 59).

Paper bundle (`nullsim/outputs/paper.py`):

- **Borrowed characterization curves (Issue 14).** A host whose nearest J
  cell skipped its characterization curve still borrows the nearest
  curve-bearing cell's chip throughput and static leak, but the realized
  spectro floor is now scaled by the ratio of the two cells'
  `characterization.spectro` `realized_spectro_floor` (direct retuned
  measurements, setpoint-trim leakage included); `servo_bw_hz` and
  `with_fringe_tracking` come from the host cell's spectro payload. In
  the archived July kilo_keck_6 run these floor ratios are 3.5 (J6/J4),
  5.9 (J10/J8), and 61 (J12/J8). Eps stays a CV against each anchor's own
  leakage: each anchor's dynamic leak RMS is scaled by the cells'
  `spectro_leak_rms_fraction` ratio, the speckle CV is kept, and eps is
  recomposed over the new leakage, so unchanged budgets leave
  eps x leakage unchanged. A zero donor floor no longer divides by zero:
  both zero leaves the curve as is, and a nonzero host floor is applied at
  every anchor and flagged (`char_floor_from_host_scene`). New survey
  columns `char_floor_scaled_from_jmag` (donor J, NaN when not borrowed)
  and `host_jmag_outside_sweep` (edge cell used without extrapolation); the
  characterization table caption says so when shown rows use either.
- **Characterization eps and max R.** A host with its own curve takes
  eps from `eps_systematic_per_anchor` interpolated at the planet's
  separation (scalar median as fallback). The recomputed characterization
  max R repeats `characterization.spectro`'s own call (row 31): the AWG
  optical throughput in AWG mode, sky summed over the spectro ports, the
  stage's detector pixels per port, pixels per resolution element (1 for
  MKID), `snr_min_per_bin`, and `resolving_power_cap`.
- **Dual polarization factor (Issue 15).** The 0.87 scalar to dual
  polarization factor now applies only to pipelines without
  `polarization.*` stages, and the characterization fallback no longer
  receives it twice. kilo_keck_6 PDI SNRs rise about 7%: tau Boo b
  7.909 to 8.480 before the background change below.
- **PDI background (row 58).** The detection PDI now charges the survey
  row's sky and detector dark per dark port (new `dark_rate_per_dark_port`
  survey column; both carried into the PDI tables). tau Boo b 8.480 to
  8.463, GJ 3512 b 0.03224 to 0.03199 (from the archived July survey rates).
  The detection background is now the rate `performance.standard`
  charged (`sky_rate_hz_per_port`, which prefers the background stages'
  `background_rate_hz_per_dark_port`: sky plus thermal and exozodi), else
  that key before plain sky, newest component first.
  The characterization PDI now charges the spectro ports' sky
  (`sky_rate_hz_per_spectro_port`, arm total) and dark with the spectro
  port count; new survey columns `char_n_spectro_ports`,
  `char_sky_rate_per_spectro_port`, `char_dark_rate_per_spectro_port`.
- **Component selection.** `_find_component` and `_component_name` pick
  the newest component whose name starts with the prefix, with no
  preference for an older exact-name match.
- **Survey interpolation (row 59).** Failed anchors and infinite contrasts
  are gaps that are never bridged, a single surviving anchor holds only at
  its own separation, and planets outside the evaluated grid get NaN
  instead of edge-clamped throughput, null, and rates.

Detection-space figure (`nullsim/outputs/plots/detection_space.py`):

- **Site latitude (Issue 16).** The observability cut reads the resolved
  site preset (latitude and name), with `[site.observability]` overrides
  still winning. nayra_eelt now cuts at Cerro Armazones: 606 planets
  instead of 539 (217 no longer wrongly shown, 284 no longer hidden, AU Mic
  b and c among them).
- **Declared catalogs (row 57).** With `[catalog.exoplanets]`, the overlay
  loads through `catalogs.load_catalog` with the declared classification
  rule, supplied `separation_mas`/`contrast_estimate` columns, and column
  aliases; reflected-light contrast is computed only where no contrast is
  given. hwo_space: 176 of 899 planets change class to the v1 rule.
  Without a catalog block the bundled KILO overlay is unchanged.
- **Ideal-coronagraph background (row 56).** Local plus exozodi surface
  brightness at the scene separation over one (lambda/D)^2 element of the
  primary, times mirror train and QE, from the new `background.space_zodi`
  keys. Without those keys it warns and falls back to the routed zodi.
- **Every characterization curve is drawn (Issue 25).** Several
  `characterization.detection_curve` components on one record each get a
  line, a color, and a legend label naming the component.
