---
date: 2026-08-09
---
### Changed: HDC template spectra regenerated at three samples per FWHM

Every packaged template under `nullsim/data/spectra/` was stored on an
R = 30000 grid carrying an R = 30000 line-spread function — one sample per
FWHM. Measuring the resolving power back out of the files returned 17-18k, so
`characterization.hdc_ccf` runs labelled R = 30000 were analysing content worth
little more than half that. All seven are regenerated on an R = 90000 sampling
grid with the same R = 30000 LSF, and two new files join them.

Files change bytes and row count (38431 → 115286 rows, 1.1 → 3.5 MB each), so
every run hash that reaches a template moves and the affected stage caches go
cold. That is the intended consequence, not a regression.

- **Metadata split.** The single `resolving_power` header key conflated grid
  density with content resolution. Snapshots now carry
  `sampling_resolving_power` (90000) and `lsf_resolving_power` (30000) as
  separate keys, and `characterization.hdc_ccf` reads the LSF key for every
  content decision. Snapshots written with the old single key still load and
  are read as content resolution. A new `physics.spectra.provenance_number`
  does the parsing on a word boundary, because a substring search for
  `resolving_power` also matches `sampling_resolving_power`.
- **Venus is reflected-only.** PSG's `rif` column is *apparent* I/F: reflected
  sunlight plus the planet's own emission. Venus radiates through its 1.7 um
  window at the per cent level, and that part does not scale with a different
  host star. The generator now measures the thermal fraction sample by sample
  with two extra radiance passes and divides it out. Earth is audited instead
  of subtracted and its emission recorded; both files carry a machine-readable
  `thermal_treatment` and `thermal_fraction_peak`.
- **Earth is labelled honestly.** It is one clear atmospheric column
  integrated over the illuminated disk — no clouds, no land/ocean contrast. A
  clear-sky endmember, not a representative Earth. The data is unchanged
  beyond the resampling; the header, the loader docstring and the generator
  say what it is. A cloud-weighted or GCM ensemble remains future work.
- **Telluric provenance corrected.** SkyCalc's `trans` is *total*
  line-of-sight transmission (molecular absorption × ozone × Rayleigh × Mie),
  not molecular absorption alone, and its `3060m` option is the Cerro
  Armazones site model — so Armazones is native and only Maunakea is a proxy.
  The old headers said the opposite of both.

### Added: packaged sky radiance snapshots

`sky_radiance_maunakea.csv` and `sky_radiance_armazones.csv`, from the same
SkyCalc requests at the same airmass and precipitable water vapour as the
transmission products, keeping the `flux` column: airglow and atmospheric
thermal emission in ph/s/m2/um/arcsec2. Reachable as
`packaged_spectrum_path("maunakea_radiance")` / `("armazones_radiance")` under
the new `sky_radiance` spectrum kind. No stage consumes them yet — the ground
sky is still spread as a flat photon-rate density — so this is packaging and
loading only.

### Changed: the spectra generator binds its own inputs

`scripts/generate_hdc_spectra.py` regenerates all nine files and no longer
leaves any part of a request to a live service default.

- The frozen PSG configs pin the whole `GENERATOR` block as literal text. PSG
  does not expand its own defaults when echoing a config back, so anything
  left out was being filled in at request time — including
  `GENERATOR-CONT-STELLAR`, which flipped to `Y` would stamp the host star's
  absorption lines into what the file calls an albedo, moving samples by up to
  82%. `--pin-psg-generator` writes the block into an existing config without
  re-harvesting the atmosphere.
- Stellar snapshots record the SHA-256 of the PHOENIX FITS they were built
  from, and the snapshot tests recompute the PSG config digest from the
  committed file rather than trusting the recorded value.
- The multi-chunk seam validator was median-only, which is blind to sparse
  line corruption: a boundary that drops a handful of lines moves a few per
  mille of samples by order unity and leaves the median flat. It now also
  checks the 99th percentile, the worst single sample, and the disagreement
  after both sides are high-pass filtered onto their line residuals.
- PSG serialises one job per client and answers a second with prose rather
  than an error line. Concurrent targets had a request for Earth answered with
  a Venus spectrum; requests now retry on the busy message, and the object
  column is still looked up by name so a swap fails loudly.

### Added: the content resolving power is measured, not asserted

`tests/physics/test_spectra_snapshots.py` locates isolated lines in each
packaged file, refines each peak with a three-point parabola, measures the
half-depth width in `ln lambda`, and requires the median to sit near the
declared LSF. On the one-sample-per-FWHM files the same measurement returned
about 12000; it is what would catch this class of regression next time.
