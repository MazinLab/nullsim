### Paper survey tables and detection-space overlay fixes

Fixes from the 2026-11-07 physics review (Issues 14, 15, 16, the
detection-space part of 25, and rows 56 to 59).

Paper bundle (`nullsim/outputs/paper.py`):

- **Borrowed characterization curves (Issue 14).** A host whose nearest J
  cell skipped its characterization curve still borrows the nearest
  curve-bearing cell's chip throughput and static leak, but the realized
  spectro floor is now scaled by the ratio of the two cells' realized-null
  excess (`realized_null_depth - ideal_chip_null_depth`), and eps and the
  systematic bandwidth come from the host's own cell. New survey columns
  `char_floor_scaled_from_jmag` (donor J, NaN when not borrowed) and
  `host_jmag_outside_sweep` (edge cell used without extrapolation); the
  characterization table caption says so when shown rows use either.
- **Dual polarization factor (Issue 15).** The 0.87 scalar to dual
  polarization factor now applies only to pipelines without
  `polarization.*` stages, and the characterization fallback no longer
  receives it twice. kilo_keck_6 PDI SNRs rise about 7%: tau Boo b
  7.909 to 8.480 before the background change below.
- **PDI background (row 58).** The detection PDI now charges the survey
  row's sky and detector dark per dark port (new `dark_rate_per_dark_port`
  survey column; both carried into the PDI tables). tau Boo b 8.480 to
  8.463, GJ 3512 b 0.03224 to 0.03199 (from the archived July survey rates).
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
