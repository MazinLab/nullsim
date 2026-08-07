---
date: 2026-08-07
---
### High dispersion coronagraphy: matched-filter cross-correlation on the spectro port

HDC is an after-the-fact analysis gain. Nothing in the nuller changes — the
null depth, throughput, and photon rates are exactly the ones
`characterization.spectro` publishes. What the cross-correlation buys is
rejection of the leaked starlight: the planet's line forest, Doppler shifted
out of the stellar rest frame, projects onto a template that the smooth leaked
continuum does not, so the null-instability systematic enters multiplied by
the template-projected fraction of the leak rather than the whole band leak.
Framing and the sqrt(N_lines) scaling follow Wang, Mawet, Ruane, Hu & Benneke
2017, AJ 153, 183.

- **New stage `characterization.hdc_ccf`** (`results`, `scene` →
  `results`). Sits downstream of `characterization.spectro`, dresses that
  stage's per-wavelength budgets with real spectral shapes (planet albedo,
  stellar photosphere, optional telluric transmission), channelizes at the
  spectrograph resolving power, and evaluates the matched-filter budget:
  `ccf_snr`, its photon limit, the systematic CCF leakage and rejection
  factor, the effective line count, and diagnostic scans of SNR versus
  resolving power and versus trial radial velocity. Deterministic — it draws
  no random numbers. `resolving_power = None` takes the AWG or MKID resolving
  power from the upstream payload; `telluric_template = None` is the right
  setting for space.
- **New scene parameter `[scene.planet] radial_velocity_kms`** (stage param
  `planet_radial_velocity_kms`, default 30 km/s). Shifts the planet spectrum
  as a whole; without it the cross-correlation has nothing to separate the
  planet from the star.
- **New plot `hdc_ccf`**: channelized planet and leak spectra, per-channel
  template weights with their share of the photon-limited SNR², CCF SNR
  versus resolving power, and the detection statistic versus trial radial
  velocity with the injected value marked.
- **New `characterization_summary` columns**: `hdc_ccf_snr`,
  `hdc_ccf_snr_photon_limited`, `hdc_systematic_rejection`,
  `hdc_n_effective_lines`, `hdc_resolving_power`. Populated only for rows
  whose `characterization.spectro` component converged and matched an HDC
  component; blank otherwise.
- **Packaged template spectra** under `nullsim/data/spectra/`, all on
  constant-R grids at R = 30000 over 0.5–1.8 um with provenance headers:
  Earth and Venus disk-integrated albedos at quadrature from the NASA
  Planetary Spectrum Generator, G2V/K2V/M2V PHOENIX photospheres, and
  Maunakea/Armazones telluric transmission from ESO SkyCalc.
  `scripts/generate_hdc_spectra.py` regenerates all seven, each from its own
  upstream: the albedos from the PSG web API driven by the frozen configs in
  `scripts/psg_config/*.cfg`, the photospheres from the PHOENIX HiRes FITS
  archive, the tellurics from the SkyCalc REST service. The stage declares the
  snapshots it reads as external dependencies, so their bytes enter the run
  hash.
- **New example config `examples/test/hdc_ccf_smoke.toml`**: four-mode M2V
  host with an AWG spectro port at R = 5000, exercising the whole path end to
  end in a few seconds.
- **New example config `examples/test/hdc_ccf_systematics.toml`**: the same
  instrument with a quasi-static 1e-3 null instability on the leaked
  starlight, hand set via `residual_speckle_stability_factor` with
  `with_fringe_tracking = false`. The broadband band SNR falls from 387.4 to
  3.1 while the cross-correlation only falls from 70.5 to 33.1, which is the
  regime HDC is for. In the photon-limited smoke config HDC loses to
  broadband, 70.5 against 387.4 — correctly so, since the matched filter
  discards the continuum photons that the band SNR counts.
