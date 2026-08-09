---
date: 2026-08-09
---
### Fixed: detection-mode HDC normalizes against the physical stellar rate

`performance.hdc` scaled its unit-contrast planet reference by
`performance.standard`'s `stellar_rate_hz_total`, which is the sum of the
detector's port rates *after* per-pixel saturation clipping. Saturation is a
readout limit; a planet at contrast `c` still delivers `c` times the photons
the star actually sends. On the shipped KILO example at J = 4 the bright ports
clip by 3.804x, so every contrast on the CCF curve came out that much too
bright.

The physical rate now travels the whole chain. `detector.mkid` and
`detector.snspd` publish `uncapped_port_rates` (per-port totals before the cap)
and a `detector_saturated` verdict; `performance.standard` sums them into
`stellar_rate_hz_total_uncapped`; `performance.hdc` prefers that total. If a
payload reports saturation and carries no uncapped total, the stage raises and
names the missing key rather than falling back — a wrong contrast curve that
looks right is worse than a run that stops.

Everything else in `performance.standard` still uses the capped rates, which is
correct for a noise budget but means the broadband curve it publishes is on the
old normalization. When the two differ, `performance.hdc` emits a
`RuntimeWarning` and records
`stellar_normalization_mismatch_note`: `ccf_contrast_gain` divides a physically
normalized CCF curve by a saturation-normalized broadband one and carries the
ratio. On `examples/test/hdc_detection_smoke.toml` the gain moves 7.483 →
28.183 (0.163 → 0.618 with the systematics knobs removed); the CCF curve is the
corrected one, the gain is the flagged comparison.

### Fixed: detection-mode HDC charges dark per pixel, not per channel

One spectrograph channel is one resolution element, which spans
`pixels_per_resolution_element` detector pixels, and dark is a per-pixel rate.
`characterization.hdc_ccf` takes that factor from the AWG metadata;
`performance.hdc` was charging a single pixel per channel, a silent factor-of-2
dark discount between the two modes at the Nyquist AWG default. A new
`pixels_per_resolution_element` param inherits from an upstream
`characterization.spectro` AWG payload when there is one, and otherwise takes
the Nyquist value of 2.

### Fixed: the chromatic leak floor is read in the effective null convention

`performance.hdc` reconstructs the per-wavelength null profile and raises it
over the grey floors `performance.standard` applies. It read those floors from
`null_depth_residual` and `calibration_floor`, which are raw KILO
`mean(dark)/mean(bright)` nulls, and compared them against a profile that is an
effective dark/total power fraction — a factor of 2.01 apart on the KILO Keck
layout, in the direction that flattens a genuinely chromatic leak.
`performance.standard` now publishes `calibration_floor_effective_fraction`
alongside the existing `null_depth_residual_effective_fraction`, and
`performance.hdc` uses both. A payload predating those keys still resolves, and
`leak_shape_model` states which convention the floor was read in.

### Fixed: a rejected spectro solution no longer yields a plausible CCF

`characterization.hdc_ccf` built its whole budget from the upstream retune's
throughput, leak, and photon rates without ever asking whether the pipeline
accepted that retune. The `characterization_summary` table and the `hdc_ccf`
plot both masked on non-convergence; the stage payload — what a scripted
consumer reads first — did not.

The acceptance rule now lives once, in
`nullsim.stages.characterization.acceptance`, and all three consumers call it.
On rejection the stage publishes NaN for every metric, empty spectra,
`spectro_solution_accepted = False`, and a machine-readable
`invalidity_reason` (`spectro_optimizer_did_not_converge` or
`spectro_null_constraint_violated`). Accepted payloads carry the same keys with
the opposite verdict.

### Fixed: a refined anchor must actually reach the threshold

`performance.hdc`'s single refinement step accepted any SNR within ±2% of the
target, so an anchor whose matched filter delivered 4.90 was published on a
curve labelled 5 sigma. The acceptance band is now one sided — `[target,
target x 1.02]` — because bounded overshoot only makes the published contrast
conservative while undershoot overstates the instrument. The correction step
aims 1% above the target so its own residual error lands inside the band rather
than straddling the lower edge; an anchor that still falls short is a
non-detection, handled by the existing per-anchor containment.

### Added: `hdc_detection` table

Detection-mode HDC curves existed only as dash-dot lines inside the
`detection_space` figure. The new registered table writes one row per
separation anchor — component, separation, both contrasts, the gain, the target
SNR, the integration time, the resolving power, and the stage's own reason for
any non-detection — so the numbers behind the headline claim can be read back
without re-running.
