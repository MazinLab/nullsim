# Fresnel Propagation Notes

## Source files reviewed

Search command:

```bash
grep -rilE 'fresnel|angular_spectrum|propagat|fft2|ifft2|transfer_function|diffract' kilo nayra --include='*.py'
```

KILO hits read in full:

- `kilo/injection.py`
- `kilo/sky_background.py`
- `kilo/config.py`
- `kilo/studies/uv_coverage.py`
- `kilo/studies/verification.py`
- `kilo/studies/calibration_time.py`
- `kilo/studies/exoplanet_table.py`
- `kilo/studies/exoplanet_characterization.py`
- `kilo/studies/natural_design.py`
- `kilo/studies/delay_line_sensitivity.py`
- `kilo/transport.py`
- `kilo/chip.py`
- `kilo/plotting.py`
- `kilo/fringe_tracking.py`
- `kilo/exoplanets.py`
- `kilo/pwfs.py`
- `kilo/atmosphere.py`

NAYRA hits read in full:

- `nayra/injection.py`
- `nayra/sky_background.py`
- `nayra/config.py`
- `nayra/studies/ao_diagnostics.py`
- `nayra/studies/paper_plots.py`
- `nayra/transport.py`
- `nayra/chip.py`
- `nayra/wfs.py`
- `nayra/polarization.py`
- `nayra/performance.py`

Neither codebase contains a Fresnel propagator, angular-spectrum propagator,
FFT transfer-function propagator, or tests for diffraction propagation. The
`fft2` / `ifft2` hits are dirty-beam Fourier synthesis:

- `kilo/studies/uv_coverage.py:25-67` grids `(u, v)` samples and applies
  `fftshift(abs(ifft2(ifftshift(grid))))` to form a dirty beam.
- `kilo/plotting.py:137-150` uses the same center-origin grid and shifted
  `ifft2` convention for visualization.

Those routines are interferometric PSF construction, not near-field wave
propagation. Since neither reference tree has propagation code, the nullsim
pinning test is derived from Goodman, "Introduction to Fourier Optics" (3rd
ed., 2005), Ch. 4-5, and Voelz, "Computational Fourier Optics" (SPIE 2011),
sections 5.1 and 7.2. The shipped implementation suppresses evanescent
components (`|k_perp| > k`) but does not implement the additional
angular-bandwidth limit of Matsushima & Shimobaba, Opt. Express 17, 19662
(2009); that paper is therefore *not* a reference for the current kernel and
is not cited in `nullsim/physics/fresnel.py`. Adding the Matsushima-Shimobaba
anti-aliasing bandwidth limit is a future extension.

## Method chosen: angular-spectrum with evanescent-wave suppression (preferred over single-FFT impulse-response Fresnel because it avoids the sampling-criterion failure mode for small z)

Use an evanescent-suppressed angular-spectrum propagator as the default Fresnel primitive.
The impulse-response single-FFT Fresnel method is tempting because it is compact,
but its sampling criterion can fail at small propagation distance `z_m`; that is
exactly the regime where nullsim lab-bench and chip-adjacent checks are likely
to run. Angular-spectrum propagation keeps the sampled field on the same
`dx_m` grid after propagation and supports exact round trips for low-NA fields
when evanescent components are excluded consistently.

The expected public primitive is:

```python
propagate_angular_spectrum(
    field: np.ndarray,
    *,
    wavelength_m: float,
    dx_m: float,
    z_m: float,
) -> np.ndarray
```

The input `field` is a scalar 2D complex amplitude sampled on a center-origin,
square, even grid. Pipeline field tensors with shape
`[n_modes, n_wavelength_bins, n_pol]` are a different abstraction; callers that
want 2D wave-optics validation should extract or construct scalar planes before
calling this primitive.

## Forward kernel sign: exp(+i k_z z) with k_z = sqrt(k^2 - kx^2 - ky^2) (real for propagating, imaginary for evanescent - set evanescent components to 0 unless flagged)

Use the forward transfer function

```text
H(kx, ky; z) = exp(+i * kz * z)
kz = sqrt(k^2 - kx^2 - ky^2)
k = 2*pi / wavelength_m
```

For propagating components (`kx^2 + ky^2 <= k^2`), `kz` is real and the transfer
function has unit magnitude. For evanescent components, the default should set
the transfer-function value to zero rather than applying exponentially growing
terms during negative-distance round trips. A future explicit flag can expose
evanescent retention for specialized near-field validation, but the default
should be conservative and stable for astronomical simulation use.

The pinning test uses forward `z_m` followed by backward `-z_m` on a broad
Gaussian field. With only propagating spatial frequencies present at numerical
precision, this should recover the original complex field and conserve total
power.

## Dtype: float64 for spatial/frequency grids; complex128 for field

Grid construction should use `float64` for spatial coordinates and frequency
coordinates. The propagated field should be coerced to `complex128` so phase
accumulation is not silently done in `complex64`. This matches the numerical
analysis side of nullsim, where precision dominates over memory footprint for
validation primitives.

## FFT ordering: numpy.fft.fft2 -> multiply by transfer function in shifted convention -> ifft2; document whether KILO/NAYRA use fftshift or center-origin

KILO's dirty-beam routines use center-origin grids: they place zero spatial
frequency at `n_pixels / 2`, call `ifftshift` before `ifft2`, then `fftshift`
the image-plane result. NAYRA has no `fft2`/`ifft2` optical propagation code.

For nullsim Fresnel propagation, use a center-origin spatial field and a shifted
frequency grid for readability:

```python
spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
propagated = np.fft.fftshift(
    np.fft.ifft2(np.fft.ifftshift(spectrum * transfer_function))
)
```

This is equivalent to the unshifted `fftfreq` convention when implemented
carefully, but the shifted form makes the band-limit mask and evanescent cutoff
auditable.

## Pixel-pitch parameter: dx_m (meters) - units explicit in name

The sample pitch must be named `dx_m` and measured in meters. Do not call it
`dx`, `pix`, or `sampling`; the physical unit is load-bearing because the
Nyquist spatial frequency and phase kernel both depend on it. Rectangular
pixels are not required for the first implementation; if they become necessary,
add `dy_m` deliberately instead of overloading `dx_m`.

## Edge cases: even vs odd N (require even square for symmetric kx,ky grid; raise if not)

The first nullsim Fresnel primitive should raise `ValueError` for:

- non-2D fields,
- non-square fields,
- odd-sized grids,
- empty fields,
- non-positive `wavelength_m`,
- non-positive `dx_m`,
- non-finite `z_m`.

Even square grids give a symmetric center-origin `kx, ky` layout and align with
KILO's dirty-beam indexing convention. Supporting odd grids is possible later,
but it complicates the origin convention and is not needed for the current
validation role.

## Hard-won lessons from KILO/NAYRA comments

- The word "Fresnel" in KILO/NAYRA usually means fiber-face reflection loss
  (`FRESNEL_LOSS = 0.035`), not diffraction propagation.
- KILO's atmosphere code uses the Fresnel radius as a scintillation validity
  scale. The Roddier aperture-averaging law is clipped below the Fresnel radius
  instead of being extrapolated into an unphysical divergence.
- KILO distinguishes the boundary-layer scintillation timescale from the faster
  high-altitude jet-stream timescale used for amplitude fringe-tracker sizing.
- KILO and NAYRA assume scintillation draws are uncorrelated when baselines are
  much larger than the Fresnel radius; this is a stochastic amplitude model, not
  propagated phase screens.
- The pyramid-WFS modules explicitly use closed-form analytic models and do not
  perform Fourier-optics propagation.
- The `propagate` wording in chip, transport, and performance modules means
  matrix propagation through a unitary, fiber attenuation/delay, or batched
  field-vector multiplication. None of those are 2D diffraction propagation.
- The only `ifft2` routines in the reference trees are dirty-beam synthesis
  helpers. They document the reference projects' center-origin FFT convention,
  but they do not supply Fresnel physics to lift.

## Architect-Spec Divergence

The public primitive proposed in the notes above differs from the shipped
implementation in three deliberate ways.

### 1. Rename: `propagate_angular_spectrum` → `angular_spectrum_propagate`

The notes propose `propagate_angular_spectrum(field, *, wavelength_m, dx_m,
z_m)` as the primary public name.  The implementation uses
`angular_spectrum_propagate` as the canonical export, matching the
verb-last naming convention used for the other physics primitives
(`decompose_clements`, `reconstruct_clements`, `jones_to_stokes`).

### 2. Unit switch: meters → micrometers at the public boundary

The notes specify `wavelength_m` (meters) and `dx_m` (meters).  The
implementation accepts `wavelength_um` and `pixel_pitch_um` (micrometers).
This matches the existing package-wide convention: `ruilier_cassaing`,
`photometry`, and `kolmogorov` all work in micrometers at their public
boundaries.  Keeping Fresnel consistent avoids silent unit-mismatch bugs
when callers pass values from those primitives directly.

### 3. Compat wrapper `propagate_angular_spectrum` (removed)

`nullsim/physics/fresnel.py` once carried a thin compat wrapper
`propagate_angular_spectrum` that converted meters to micrometers and
delegated to `angular_spectrum_propagate`.  It existed only to satisfy
`test_fresnel_pinning.py`, which had been written against the notes API in
meters before the implementation landed.  It was never part of the stable
public API — absent from `nullsim/physics/__init__.py` and excluded from
`fresnel.py`'s `__all__`.

Both the wrapper and that pinning test were removed on 2026-08-09.  The
propagator contract is now covered entirely by `tests/physics/test_fresnel.py`,
which calls `angular_spectrum_propagate` directly with µm arguments.
