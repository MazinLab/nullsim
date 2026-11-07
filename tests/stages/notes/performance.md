# Performance Stage Notes

## KILO source surveyed

The terminal stage of the KILO pipeline lives in
`kilo/performance.py` (1430+ lines).
The functions consumed by nullsim's `StandardPerformance` stage are:

- `detected_stellar_rate_at_telescope` (lines 62-161). Single source of
  truth for "stellar rate at chip" used by paper_plots, n_optimization,
  calibration_time, paperv3_plots, and the FT-floor Monte Carlo. Walks
  the photon budget end-to-end:
  `flux_top_atm -> A_eff -> transport (fiber + connector + airmass) ->
  chip coupling -> MKID QE -> detected rate`. Pre-2026-05-05 KILO had
  three or more parallel inline implementations across the trade
  studies, each missing a different piece of the loss chain (one missed
  fiber attenuation, another missed MKID QE, a third hard-coded a
  `0.5*0.85` fudge factor). Headline numbers drifted apart by 1.8-5x;
  KILO commit `47607de` consolidated them into this helper. The nullsim
  pipeline solves the same problem at the architecture level — the
  rate-at-detector value is computed exactly once by the detector
  stage (`detector.mkid` / `detector.snspd`) and the performance stage
  consumes the result without recomputing it.
- `null_depth(stellar_dark_power, stellar_total_power)` (lines 164-181).
  `N = sum(dark) / sum(all)`. Returns `0.0` for non-positive total.
  nullsim's detector stages already write `null_depth_raw` using the
  same partition; the performance stage consumes whichever component
  the upstream calibration stage produced (`null_depth_residual` if
  the calibration stage ran, else `null_depth_raw`). The old
  `null_depth_calibrated` key is only a compatibility alias for the
  residual. Without a sensitivity stage the residual sets only the
  systematic leakage; the Poisson term always uses the detected dark rate
  (review 2026-11-07 row 71). Every rate is normalized to the physical
  (pre-saturation) stellar rate, with dark-port saturation applied as a
  per-port ratio (Issue 9).
- `planet_throughput(planet_dark_power, planet_total_power)` (lines
  184-202). `T = planet_dark / planet_total`. Returns `0.0` for
  non-positive total. The nullsim stage carries planet throughput as a
  Param-driven default until a real angular-throughput stage lands in
  §15.4.
- `signal_to_noise` (lines 204-292). Headline formula:

      SNR = S / sqrt(S + B + D + Sky + systematic_variance)

  where `S = detected_planet_rate * t`, `B = detected_leakage_rate * t`,
  `D = dark_rate * n_dark_ports * t`, `Sky = sky_rate * n_dark_ports
  * t`. The systematic depends on whether fringe-tracking temporal
  averaging is on:

      without FT averaging:   systematic = (eps * B)^2            [t^2]
      with FT averaging:      systematic = eps^2 * B_rate^2 * t / (2 * f_servo)  [t]

  Without averaging, `SNR -> S / (eps * B)` as `t -> inf` — a hard
  ceiling. With averaging, the systematic is linear in `t` like
  Poisson, so adding integration time always helps and the ceiling
  disappears. The nullsim stage exposes both branches via the
  `with_fringe_tracking` Param.
- `integration_time_for_snr` (lines 802-894). Inverts the SNR formula.
  Without instability:
      t = target_snr^2 * (S + B + D + Sky) / S^2
  With FT averaging:
      t = target_snr^2 * (noise_rate + eps^2 * B^2 / (2 * f_servo)) / S^2
  Without FT averaging:
      t = target_snr^2 * noise_rate / (S^2 - target_snr^2 * (eps * B)^2)
      (returns `inf` when the denominator is non-positive — the target
      SNR exceeds the hard ceiling.)
  Returns `inf` when `planet_rate <= 0`.
- `contrast_curve` (lines 896-1008). At each angular separation, solves
  the quadratic
      S^2 - snr^2 · S - snr^2 · N = 0
      S_min = 0.5 * (snr^2 + sqrt(snr^4 + 4 * snr^2 * N))
  and returns `c = S_min / (stellar_rate * planet_throughput(sep) * t)`.
  N includes the same Poisson + systematic terms as
  `signal_to_noise`. The nullsim stage builds the contrast curve by
  scaling the planet rate via a Param-driven angular throughput
  (linear extrapolation flagged as TODO for §15.4).
- `stellar_disk_null_floor` (lines 1130-1228). For a uniform-brightness
  stellar disk of angular diameter `theta_*`, integrates
  `(2 / rho_0^2) * integral_0^rho_0 r * T(r) dr` to get the stellar
  leakage from the resolved disk. Raises `ValueError` when
  `rho_0 = theta_* / 2` exceeds the tabulated `separations_rad` grid
  (silent extrapolation would corrupt the integral). nullsim defers
  the disk-floor enhancement to §15.5 — the first-pass
  `StandardPerformance` stage exposes only the point-source SNR /
  contrast formulas, matching what the §15.5 KILO Maunakea parity
  tests actually need.

## NAYRA divergence

`nayra/performance.py` is
intentionally thinner:

- `signal_to_noise` (NAYRA lines 86-130) and `integration_time_for_snr`
  (NAYRA lines 132-180) are direct re-exports of the KILO formulas.
- `contrast_curve` (NAYRA lines 181-246) drops the `null_instability`
  systematic-noise branch entirely. NAYRA contrast curves are computed
  in the photon-noise-limited regime only, which is acceptable for
  E-ELT baselines <= 30 m where the angular separation grid stays
  inside the resolved-disk floor regime that `stellar_disk_null_floor`
  is built for — except NAYRA also omits `stellar_disk_null_floor`.
- `detected_stellar_rate_at_chip` (NAYRA lines 24-65) is the NAYRA
  analogue of `detected_stellar_rate_at_telescope`; the consolidation
  lesson from KILO commit `47607de` applies symmetrically.
- NAYRA does not ship a calibration_time study and relies on the KILO
  J-band table via the legacy-fallback warning in
  `kilo.studies.calibration_time.lookup_calibration_null` (lines
  142-153).

The nullsim `StandardPerformance` stage covers the KILO superset
(both branches of the systematic, both regimes of the integration-time
solver) and exposes the `with_fringe_tracking` Param so a NAYRA preset
can pin it `False` if the parity test ever needs the photon-noise-only
branch.

## Source lines for pin tests

- KILO `signal_to_noise`: lines 204-292.
- KILO `integration_time_for_snr`: lines 802-894.
- KILO `contrast_curve`: lines 896-1008.
- KILO `null_depth`: lines 164-181.
- KILO `planet_throughput`: lines 184-202.
- KILO `stellar_disk_null_floor`: lines 1130-1228 (deferred consumer,
  not implemented in this pass).
- KILO `detected_stellar_rate_at_telescope`: lines 62-161 (consolidation
  lesson; nullsim achieves the same single-source-of-truth property by
  routing rate through `state.results.components['detector.<x>']
  ['port_rates']`).
- NAYRA `signal_to_noise`: lines 86-130.
- NAYRA `contrast_curve`: lines 181-246 (no instability branch).

## SNR formula edge cases the nullsim stage handles

- `noise_variance <= 0`: KILO returns `0.0` (line 290). nullsim
  matches.
- `planet_rate <= 0`: KILO `integration_time_for_snr` returns `inf`
  (line 858). nullsim matches.
- `with_fringe_tracking=False` and target SNR above the ceiling:
  KILO returns `inf` when `S^2 - snr^2 (eps B)^2 <= 0` (lines 884-888).
  nullsim matches.
- Empty `contrast_curve_separations_mas`: produces empty
  `contrast_curve_value` tuple. nullsim explicit; KILO callers handle
  this at the call site.
- `with_fringe_tracking=True` with `default_servo_bw_hz` falling back
  to the Param default: stage source_model field records the fallback
  so parity tests can detect missed `fringe_tracking.closed_loop`
  upstream.

## Provenance of the systematic term

Reference: Shao & Colavita (1992) PASP for shot-noise piston variance,
Conan et al. (1995) for the servo-lag term, and KILO `performance.py`
docstring on `signal_to_noise` (lines 220-244) for the
photon-vs-systematic crossover. The "eps" in `eps * B` is the
calibration residual coupling between AO/atmosphere and the chip
(`null_instability` in KILO's API). It defaults to `1e-6` in the
nullsim stage as a free parameter — §15.5/§15.6 parity tests will pin
it to the KILO / NAYRA paperv3 values once the preset configs land.
