---
date: 2026-06-10
---
### Deep-review sweep fixes (kilo_keck_6 scope)

Fixes for the 2026-06-10 deep review (`reviews/kilo_keck_6-sweep-2026-06-10.md`).
Correctness:

- chip_optimization: reported null depth counted delay-state multiplicity
  twice — every reported/accepted null in kilo_keck_6 was 6x optimistic
  (Issue 1). Denominator now counts rows per state; regression test added.
- sensitivity/performance: the f_eff systematic-bandwidth blend's piston leg
  now reads the FT-published closed-loop `f_3db_hz` instead of the static
  100 Hz `auto_f_piston_hz` default (Issue 2).
- characterization: per-bin SNR and SNR-limited max resolving power now use
  the same NSC-reduced combined eps as the headline SNR (Issue 3); the
  optimizer's true broadband objective is published alongside the deployed
  design metric; the spectro stage emits a full-band constraint margin.
- background: sky étendue is scaled by n_physical/n_chip modes after the
  mode splitter (2x overcount, Issue 4), and stellar Strehl / obscuration
  coupling no longer debits the diffuse sky — only Fresnel does (Issue 5).
- config: sweep-cell re-materialization preserves explicit stage-level
  overrides and raises on explicit values shadowing a swept source
  (Issue 6); `[observation].integration_time_s` materializes into detector
  stages and unbound observation sweeps are rejected (Issue 15); typo'd
  typeless stage overrides raise a named error listing inherited ids.
- pipeline: stage RNG identity is now the stage's own spec digest, not the
  full config hash — cosmetic config edits no longer change stochastic
  draws or bust the downstream cache (Issue 7); `polarization.pm_fiber`
  declares `uses_rng`; `NULLSIM_TORCH_COMPILE` resolves into validated
  params so the executed path enters the cache key (Issue 25).
- chip_optimization.scipy: optimization runs on the smooth objective with
  post-hoc DAC quantization — `dac_bits` previously zeroed every
  finite-difference gradient (silent no-op optimization, Issue 13).
- injection.mode_splitter: per-branch area/diameter vectors share the
  amplitude rows' ordering on the default full-split path (Issue 12).
- cli: worker-cell exceptions no longer masquerade as "process pool
  unavailable"; only pool-infrastructure failures fall back to serial
  (Issue 14).
- performance: calibration-branch leakage converts the KILO null convention
  before the rate product (~n_dark underestimate, Issue 27a); NaN
  `effective_null_with_disk` now raises instead of becoming a perfect null.
- physics (Issue 20, deliberate KILO parity break): off-zenith projection
  now foreshortens only the inter-telescope origin; pupil-mounted
  sub-aperture offsets ride unprojected. Producers publish per-mode origins
  (`mode_origin_positions_m`); the splitter expands them.
- atmosphere: science-path transmission is Beer-Lambert airmass-scaled
  (T**airmass) like the WFS leg (Issue 28a).
- realized-null MC: one piston/scintillation realization is shared across
  wavelength bins (achromatic OPD), fixing pooled std/median semantics and
  cutting MC RNG work ~n_wl x (Issue 27d).
- sensitivity: the auto setpoint trim iterates to a fixed point and feeds
  the staged sky background into the lock-in shot-noise floor (Issue 27e).
- fringe tracking: photon-Kalman `f_sample_hz` publishes the real loop
  cadence (1/latency); inert fiber common-mode phase is no longer booked
  into the OPD-budget total (Issue 21a); accounting conventions documented.
- outputs: debug report attributes components by stage type as well as id
  (the 48-minute characterization curve was dropped from the report,
  Issue 8) and scans all eight budgets; the chip convergence panel reads
  the key producers actually publish and the torch backend now honors
  `capture_convergence_history` (Issue 9); paper-bundle dirty beam math
  corrected (DC term N, no double squaring, Issue 10); fringe plot's
  "Realized floor" resolves the opd_budget residual source (Issue 11);
  detection_space headline cells are configurable
  (`[output.detection_space].headline_jmags`), logged, and the figure now
  carries its caption and 5-sigma legend title (Issue 23); uv grid uses the
  run's band-center wavelength; loss budget annotates the sweep-median
  collapse; single-record figures are labeled with their sweep cell; KILO
  ratio badges use the KILO-convention scalar; the cumulative transmission
  curve folds in field-amplitude-applied losses (Issue 24).

Config (`examples/kilo_keck.toml`): `central_obscuration_ratio = 0.0` on
injection (packed sub-apertures are unobstructed, Issue 18);
`auto_amp_predictive_gain_variance = 3.0` (consistent with the piston-side
x3 decision, Issue 17); detector integration time materializes from
`[observation]`.

Performance: planet-throughput PA/transport axes vectorized (~20-100x,
verified to machine precision against brute force); torch optimizer uses a
fused (loss, grad) evaluation (one forward+backward per BFGS iterate) and
computes tracker hinge+CRLB penalties from one eigendecomposition; AO
sub-aperture packing runs once (cached) per geometry.

Deferred: monolith decomposition (Issue 26), per-anchor torch.compile
caching and paper-bundle re-render skip (Issue 31c/d), counts-only GN line
search (Issue 30d), wavelength-dependent fiber dispersion constants.
