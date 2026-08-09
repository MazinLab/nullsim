The HDC continuum filter now runs blockwise over contiguous runs of admitted
channels rather than over the compacted admitted vector. Smoothing across a
telluric gap treated channels that are not neighbours in wavelength as if they
were, inferring a ramp between two flat segments and manufacturing residuals
where the spectrum has no line at all. Masked-configuration CCF numbers move:
the characterization smokes go to `ccf_snr_photon_limited` 70.22 and
`ccf_snr` 31.29, and the detection smoke's `ccf_contrast_gain` to 28.199.

`CcfBudget` gains `admitted_mask` and `noise_weights`, published by
`characterization.hdc_ccf` as `ccf_admitted_mask_per_channel` and
`noise_weights_per_channel`. Without them a consumer cannot reproduce the
photon variance from the payload, because the filter's island structure is not
recoverable from the weights and rates alone.

The contrast solver no longer drops channels with zero background but non-zero
planet template: at non-zero contrast their variance is the planet's own shot
noise, which the quadratic already carries.

The RV scan's "peak at the true radial velocity" is now documented as
conditional on a smooth leak rather than guaranteed. Cauchy-Schwarz equality is
with the observed spectrum, not the planet template, so a leak with structure
the filter cannot remove moves the peak.
