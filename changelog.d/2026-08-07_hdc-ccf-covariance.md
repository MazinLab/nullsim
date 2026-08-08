Fixed the HDC cross-correlation noise model, which understated photon noise
whenever continuum filtering was on. The estimator built its signal from
median-filtered data but kept a raw diagonal photon variance; Monte Carlo on
the pinned alternating-spectrum case measured a 3.6x variance undercount and
1.9x SNR optimism. The continuum filter is now a linear high-pass (identity
minus a normalized boxcar smoother) and the covariance is propagated exactly
through it, in `matched_filter_ccf`, `ccf_vs_rv`, and
`ccf_contrast_at_target_snr` alike. Reported CCF SNRs and contrast gains move,
and detection/characterization runs made before this change are not comparable
with ones made after.

The statistic is relabeled a photon-weighted CCF with a fixed template. It is
not the optimal filter against the rank-one systematic covariance; the
Sherman-Morrison optimum is documented as an upper bound in
`tests/physics/notes/hdc.md` and deliberately not implemented, since it assumes
the leak spectrum is known exactly.

`CcfBudget.n_effective_lines` is renamed `n_effective_channels`, with the old
name kept as a deprecated alias on the dataclass and in stage payloads.
