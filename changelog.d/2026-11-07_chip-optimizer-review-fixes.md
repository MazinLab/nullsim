### Fixed

- Chip optimizer fixes from the 2026-11-07 physics review:
  - Throughput reward (Issue 20): the off-axis planet fields use the
    published physical mode diameters instead of the equal-area circle.
    A mode splitter halves area but keeps diameter, so the old envelope
    was too wide (review estimate: NAYRA reward overstated 4.7% at 10 mas
    and 20% at 20 mas; kilo_keck_6 0.8% at 10 mas). The deployed chips in
    kilo_keck_6 and nayra_eelt will move; not rerun here.
  - Throughput reward (row 29): planet power is summed over polarization
    arms, matching the leakage sum, so a PBS no longer halves the reward.
    `throughput_penalty_alpha` in `examples/kilo_keck_6.toml` and
    `examples/nayra_eelt.toml` is halved (1e-2 to 5e-3), which leaves their
    objectives unchanged by this item. Configs with a PBS and a hand-tuned
    alpha need the same halving.
  - Tracker observability with delay-state replay (Issue 21): each delay
    state is its own pointing. The hinge and CRLB penalties and the
    rank-priority restart key now act on the worst state instead of one
    stacked matrix, which inflated every singular value by sqrt(6) in
    kilo_keck_6 (the 1.3e-2 floor acted as 5.3e-3). The published
    `tracker_observability_*_final` values describe the worst state;
    `tracker_observability_singular_values_per_nm_per_state_final` holds
    all of them.
  - `calibration_time` plot (Issue 23): plots the normalized null
    (`convergence_history_normalized_null`, the convention of
    `optimization_normalized_null_final`) instead of the dimensional
    objective, which goes negative with the throughput reward. The best
    floor line is the published final null.
  - `chip_optimization.scipy` publishes `optimization_normalized_null_final`
    from the leakage term alone; it previously included the negative
    throughput reward. Unchanged when `throughput_penalty_alpha = 0`.

### Added

- Both chip optimizer backends publish `convergence_history_normalized_null`
  beside `convergence_history` when history capture is on, and
  `optimal_chip_params_unquantized` (the pre-DAC optimum, same layout as
  `optimal_chip_params`) for the DAC bit-depth diagnostic.
