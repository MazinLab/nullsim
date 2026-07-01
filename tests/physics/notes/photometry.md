# Photometry Notes

## Upstream Search

Search terms: `photomet`, `zero_point`, `vega`, `jmag`, `jband`,
`magnitude`, `flux_density`.

KILO has no standalone `photometry.py` module. Detector-side photometry lives in
`kilo/detectors.py`, with supporting sky
surface-brightness use in `kilo/sky_background.py`, WFS R-band photon-count
calibration in `kilo/pwfs.py`, and canonical detected-rate loss chains in
`kilo/performance.py`.

NAYRA also has no standalone photometry module. `nayra/detectors.py` is a thin
re-export of `kilo.detectors`, so it inherits the KILO detector zero points and
photon-rate helper. NAYRA has site-specific sky background code in
`nayra/sky_background.py`, WFS R-band photon-count calibration in `nayra/wfs.py`,
and multiband channel definitions in `nayra/multi_band.py`.

No dedicated upstream `test_photometry.py` exists. The nearest upstream tests
are `TestStellarPhotonRate` in both KILO and NAYRA `test_performance.py`, which
only check positivity, brighter/fainter ordering, area scaling, and a broad
order-of-magnitude range. Those behaviors are covered here by direct
zero-point, magnitude-scaling, and photon-rate tests rather than lifted as
module parity tests.

## Upstream Constants And Conventions

KILO/NAYRA detector photometry is Vega-system unless otherwise stated. KILO's
`VEGA_ZEROPOINTS_JY` table uses:

- `V = 3640 Jy`
- `I = 2416 Jy` is added in nullsim as the HWO 0.8 um optical proxy
- `Y = 2026 Jy`
- `J = 1594 Jy`
- `H = 1024 Jy`
- `K = 666.7 Jy`

The J/H/K entries come from Cohen, Wheaton & Megeath 2003 for the 2MASS system.
KILO converts Jy to SI `F_nu`, uses `delta_nu = c * delta_lambda / lambda^2`,
then divides by `h c / lambda` to get photons. KILO can force the zero-point
band independently of the wavelength, which is useful when catalog magnitudes
are J-band but the simulation wavelength is H-band.

KILO and NAYRA WFS code use an R-band photon zero point,
`3.01e10 photons s^-1 m^-2 um^-1`, not the detector Jy table.

## HWO I-Band Reference

The HWO first-pass config is centered at 0.8 um, outside the KILO/NAYRA
near-IR science bands. `nullsim.physics.photometry` therefore includes a
Cousins I Vega proxy from Bessell, Castelli & Plez 1998:

- `lambda_eff ~= 0.806 um`
- `F_nu ~= 2416 Jy`
- `F_lambda ~= 1.12e-8 W m^-2 um^-1`

Detector stages keep the legacy KILO/NAYRA convention by default: the supplied
`scene.star.jmag` is interpreted directly in the configured detector band. HWO
opts into `derive_band_mag_from_scene_jmag=true`, which estimates the detector
band magnitude from `scene.star.jmag` and `scene.star.spectral_type` before
applying the I-band zero point.

## Adopted J-Band Reference

This WP adopts the 2MASS J-band isophotal wavelength `lambda = 1.235 um` and
bandwidth `0.162 um` from Cohen, Wheaton & Megeath 2003 / the 2MASS
Explanatory Supplement. Bessell, Castelli & Plez 1998 is noted as the alternate
broad-band calibration reference, but is not used for the pinned constant.

The 2MASS table gives the J-band Vega zero-magnitude flux density as:

- `F_lambda = 3.129e-13 W cm^-2 um^-1`
- `F_nu = 1594 Jy`

`nullsim.physics.photometry` returns SI flux density, so the adopted public
constant is:

`3.129e-13 W cm^-2 um^-1 * 1e4 cm^2 m^-2 = 3.129e-9 W m^-2 um^-1`.

## Photon-Rate Conversion

For a Vega-system magnitude `m`, spectral flux density is:

`F_lambda(m) = F_lambda(0) * 10^(-0.4 m)`.

For a rectangular spectral bin of width `delta_lambda_um`, the photon rate per
square meter per bin is:

`R = F_lambda(m) * delta_lambda_um / (h c / lambda_m)`.

The current function treats the zero-point spectral flux density as constant
across the requested rectangular bin. This is the correct narrow-bin interface
for a future replacement of `IdealCounter.Params.zero_point_rate_hz_per_m2_per_bin`,
which is currently multiplied by the magnitude factor and by the per-bin
throughput spectrum without applying a bin width.

## Ideal Counter Sanity Value

For the current vertical-slice H-band grid used in examples,
`lambda = 1.65 um` and `delta_lambda = 0.30 um / 32 = 0.009375 um`.
With the SI-converted Cohen J-band zero point:

- `J = 8`: `1.53739e5 photons s^-1 m^-2 bin^-1`
- Integrated over the full `0.30 um` band: `4.91965e6 photons s^-1 m^-2`
- The current placeholder path gives
  `1.0e10 * 10^(-0.4 * 8) = 6.30957e6 photons s^-1 m^-2`

The full-band physical estimate is within a factor of about `1.28` of the
placeholder after magnitude scaling. The per-bin value is lower by `32x`, which
is expected because `ideal_counter.py` currently stores a per-bin placeholder
constant but its numerical value is closer to a full-band count-rate scale.
