---
date: 2026-11-07
---
### Fixed: characterization epsilon is the retuned spectro port's own leakage CV

`characterization.spectro` and `characterization.detection_curve` charged the
detection chip's `null_instability` to the retuned spectro port. That CV is
taken over the sum of 11 (KILO) or 17 (NAYRA) dark ports, and one port
fluctuates more than that sum. In a 13 mode toy the single port CV was 3.3x the
summed one (review 2026-11-07 Issue 8).

The retuned spectro Monte Carlo now returns the RMS of the port's band leakage
as well as its mean. Each realization's leakage is summed over wavelength with
detected photon weights; one piston draw is shared across bins, so the bins
are paired. The RMS is the within-azimuth spread. Epsilon is that RMS divided by
the port's total leak fraction (static plus dynamic), followed by the same NSC
reduction and speckle quadrature as before. The curve stage computes it per
anchor and publishes `eps_systematic_per_anchor` and
`null_instability_per_anchor`; its scalars are anchor medians. Without the
retuned MC (no `n_realizations_per_azimuth` upstream), both stages fall back to
the detection CV and say so in `null_instability_source`. The spectro payload
also carries `spectro_leak_rms_fraction` and `detection_null_instability`.

The dynamic floor is now photon weighted with the same weights as the static
leak, instead of an unweighted wavelength mean (row 35).

### Fixed: spectrograph SNRs honor `with_fringe_tracking`

The AWG bin SNR, the MKID bin SNR, and the SNR-limited max R always took the
servo-averaged systematic branch. Only the headline SNR honored the flag (Issue
13). Without fringe tracking those calls now get a zero bandwidth, which
`physics.spectroscopy` treats as the quasi-static branch. The published
`servo_bw_hz` is 0 in that case. Max R in AWG mode also applies
`awg_throughput`, as the AWG spectrum already did (row 31).

On `examples/test/hdc_ccf_systematics.toml` the headline SNR stays at 3.107.
`awg_min_snr_per_bin` moves 9.357 → 2.949, so a single bin no longer beats the
band. `max_spectral_resolution` moves 10000 (the cap) → 3304.
`hdc_ccf_smoke.toml` and `mkid_native_spectro.toml` do not move.

### Fixed: spectro sky counts every polarization arm

The spectro stellar and planet rates are totals over both PBS arms, but the
budget charged one arm's sky. The sky per logical spectro port is now the
per-channel rate times the detector's `n_pol_arms`, as `performance.standard`
counts every dark channel (row 28; kilo_keck_6 J = 12 sky 2.0 → 4.0 Hz).
Payloads add `sky_rate_hz_per_spectro_arm` and `n_pol_arms`. Max R also charges
the sky of all spectro ports, to match its summed planet and leakage rates.

### Fixed: the characterization planet signal includes piston

The stellar leakage carried the MC piston, but the planet throughput did not.
Near the axis, piston scatters planet light between ports (row 36,
characterization half). The band planet throughput is now the expectation
`E|U_s(X exp(i phi))|^2`. The phase covariance is `k^2 C + sigma_f^2 I`, where
`C` is the per-mode OPD covariance the realized-floor MC draws from and
`sigma_f` is its fiber phase noise. The expectation is evaluated in closed form
with `G_mn = exp(-(S_mm + S_nn - 2 S_mn) / 2)`. New physics primitives:
`piston_averaged_port_power` and `mode_piston_opd_covariance_m2` in
`nullsim.physics.characterization`. The review estimated the kilo_keck_6 J = 12
effect on the detection curve at about 2x for 3 to 5 mas, +3.9% at 14 mas, and
−0.2% at the 40 mas peak. It was not rerun for the characterization stages.
