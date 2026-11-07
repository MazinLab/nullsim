# Kolmogorov Primitives Notes

## Upstream Search

Search terms: `kolmog`, `r0`, `fried`, `phase_variance`, `phase_structure`,
`opd_rms`, `marechal`, `strehl`.

Neither KILO nor NAYRA expose a standalone `kolmogorov.py` module. The
relevant primitives live in `kilo/atmosphere.py`. NAYRA's `nayra/atmosphere.py`
is a thin re-export of `kilo.atmosphere`; the WFS-noise wavelength scaling
that diverged between the two projects was unified upstream in KILO commit
`5d7119e` (2026-05-13), so for atmosphere physics there is one source of
truth (KILO) and one downstream consumer (NAYRA).

Files inspected:

- `kilo/atmosphere.py` (905 lines).
- `nayra/atmosphere.py`
  (re-exports KILO).

## Upstream Constants And Conventions

### Wavelength scaling

KILO `scale_r0(r0_500nm, wavelength_m) = r0_500nm * (wavelength_m / 500e-9)**1.2`.
The same `1.2 = 6/5` exponent appears in `scale_tau0` and `scale_theta0`.
nullsim adopts this convention in `r0_scaled` with 500 nm as the default
reference wavelength.

### Phase structure function coefficient

KILO does not name `D_phi(r) = C * (r / r0)^(5/3)` explicitly, but uses the
same numeric coefficient `C = 6.88` in two places:

1. As `C_TEMP = 6.88` in `ao_residual_variance` — applied to the temporal
   error term `sigma2_temp = (f_G / f_servo)^(5/3)`. The
   `C_TEMP` constant is the Greenwood (1977) coefficient and the
   phase-structure-function coefficient because both stem from the same
   Kolmogorov spectrum normalization.
2. Implicit in the Greenwood frequency formula `f_G = 0.426 * v / r0`,
   where the `0.426` factor comes from the same integral over the
   Kolmogorov phase spectrum (Greenwood 1977).

The canonical Kolmogorov phase structure function is

    D_phi(r) = 6.88 (r / r0)^(5/3)    rad^2,    r << L0   (Fried 1965)

nullsim pins this as `KOLMOGOROV_PHASE_SF_COEFFICIENT = 6.88` in
`nullsim/physics/kolmogorov.py`. **KILO/NAYRA agree** — both use 6.88
(via `C_TEMP`); no discrepancy.

### Full-aperture phase variance coefficient (Noll 1976)

Noll (1976) tabulates the residual variance Delta_J left over a circular
aperture of diameter D after the first J Zernike modes are removed:

    Delta_1 = 1.0299 (D / r0)^(5/3)    rad^2    piston removed
    Delta_2 = 0.582  (D / r0)^(5/3)    rad^2    piston and one tilt removed
    Delta_3 = 0.134  (D / r0)^(5/3)    rad^2    piston, tip, and tilt removed

There is no piston-included value. For pure Kolmogorov turbulence the
k^(-11/3) spectrum is not integrable at k -> 0, so the piston variance
over a finite aperture diverges; only a finite outer scale makes it
finite. Before 2026-11-07 nullsim labeled 1.0299 "piston included" and
0.582 (which is Delta_2) "piston removed", so the atmosphere stage's
piston-removed variance was 43.5% low.

Independent check (`test_noll_coefficients_match_kolmogorov_spectrum_quadrature`):
integrating Phi(f) = 0.022896 r0^(-5/3) f^(-11/3) against
`1 - (2 J1(x)/x)^2` (and minus `16 J2(x)^2/x^2` for tip/tilt), with
x = pi D f, gives 1.0324 and 0.1347. The structure function route,
`(1/2) <6.8839 (|r - r'|/r0)^(5/3)>` over the disk, gives the same 1.0324.
Noll's tabulated values sit 0.24% and 0.5% below; the test tolerance is
0.6%. The 0.24% offset in Delta_1 is not explained here; it is too small
to matter for any nullsim budget.

**Neither KILO nor NAYRA expose this constant directly.** Both packages
jump straight to AO error sub-budgets:

- `C_FIT = 0.28` (Hardy 1998 Table 9.2) — the *fitting error*
  coefficient for a continuous-facesheet DM, applied as
  `sigma2_fit = 0.28 (d_actuator / r0)^(5/3)`.
- The temporal and WFS noise terms are independent budget items, not
  derived from a single total-variance integral.

The 0.28 coefficient is **not** an alternative to 1.0299 — it is the
post-DM-fit residual coefficient with an actuator pitch as the spatial
scale. The two coefficients describe different physical situations
(uncorrected piston-removed variance over a full aperture vs. fitting
error residual after a deformable mirror with actuator pitch d_act).

Because the Noll 1976 values are not duplicated in KILO/NAYRA, **no
KILO/NAYRA-vs-canonical discrepancy exists**. nullsim adopts Delta_1 =
1.0299 as the default and Delta_3 = 0.134 as an opt-in via
`tip_tilt_removed=True`.

### Marechal Strehl

KILO `strehl_ratio(sigma2_rad2) = exp(-sigma2_rad2)` is the standard
Marechal approximation in phase-variance form. nullsim provides the
equivalent OPD-form `strehl_marechal(sigma_opd_m, wavelength_m) =
exp(-(2 pi sigma_OPD / lambda)^2)`. The two are algebraically
identical via `sigma_phi^2 = (2 pi sigma_OPD / lambda)^2`. KILO's
`ao_piston_opd_rms` uses this conversion when reducing phase variance
to a wavelength-independent OPD RMS.

## Architect-Spec Divergence: `phase_variance` Signature

SPEC §15.2 / WP1 lists the signature as

    phase_variance(r0_m, wavelength_m) -> float
    opd_rms_m(r0_m, wavelength_m)      -> float

The aperture diameter D is a load-bearing input — the variance scales
as `(D / r0)^(5/3)` and is unbounded for D -> infinity. The architect
specification lacks an aperture argument.

This implementation adopts

    phase_variance(aperture_diameter_m, r0_m, *, tip_tilt_removed=False)
    opd_rms_m(aperture_diameter_m, r0_m, wavelength_m, *, tip_tilt_removed=False)

Rationale:

- r0 already encodes the wavelength via Fried scaling. Callers who have
  r0 at a different wavelength scale it with `r0_scaled` first.
- `opd_rms_m` keeps `wavelength_m` for the explicit phase-to-OPD
  conversion factor `lambda / (2 pi)` and so the API documents the
  dimensioned conversion at the call site.
- The tip/tilt-removed Noll coefficient is opt-in via keyword. The
  default is the piston-removed Delta_1 (1.0299), the smallest finite
  Kolmogorov aperture variance; Delta_3 (0.134) is exposed without
  proliferating function names.

Flagged for Inspector / Architect to confirm and update SPEC §15.2 to
reflect the corrected signature.

## Outer-Scale Handling

The Kolmogorov primitives in this module assume an **infinite outer
scale** `L0 -> infinity` (pure Kolmogorov spectrum, no von Karman
correction). Both `phase_structure_function` and `phase_variance`
diverge slowly for very large apertures (`D >> L0` in real
turbulence), but the Maunakea/Cerro Armazones / L2-halo regimes that
nullsim targets all satisfy `D < L0` (typical L0 ~ 20-30 m at
Maunakea), so the infinite-L0 approximation is accurate to a few
percent. A finite-L0 von Karman primitive is a clean extension when a
downstream stage needs it; the function names `phase_structure_function`
and `phase_variance` deliberately do not embed "kolmogorov" in their
names so a future von Karman variant can register alongside.

KILO/NAYRA do not apply a finite-L0 correction in `atmosphere.py`
either, so this matches upstream behavior.

## NaN / Zero / Negative Handling

| Input          | Behavior                                                 |
| -------------- | -------------------------------------------------------- |
| `r0_m = NaN`   | Propagate NaN through to output (catalog rows with       |
|                | missing turbulence measurements survive)                 |
| `r0_m = 0`     | Raise `ValueError` (infinite phase variance)             |
| `r0_m < 0`     | Raise `ValueError` (unphysical)                          |
| `D <= 0`       | Raise `ValueError` (aperture must be positive)           |
| `lambda <= 0`  | Raise `ValueError`                                       |
| `r < 0`        | Raise `ValueError` (separation is non-negative)          |
| `sigma_OPD = NaN` | Propagate NaN (Marechal Strehl returns NaN)          |
| `sigma_OPD < 0` | Raise `ValueError` (RMS is non-negative by definition)  |

The NaN-propagates / unphysical-raises split matches the rule adopted
for `photometry.py` in INS-4-006: missing data flows through, physical
nonsense raises.

## Dtype Invariance

All primitives go through a single `_as_floating` coercion that uses
`np.asarray` without forcing a dtype. float32 inputs stay float32 in
the output array; float64 stays float64; Python int/float promotes to
float64. This matches the dtype rule pinned in
`tests/physics/test_ruilier_cassaing.py`.

## References

- Noll, R. J. "Zernike polynomials and atmospheric turbulence." JOSA
  66, 207 (1976).
- Fried, D. L. "Statistics of a geometric representation of wavefront
  distortion." JOSA 55, 1427 (1965).
- Greenwood, D. P. "Bandwidth specification for adaptive optics
  systems." JOSA 67, 390 (1977).
- Roddier, F. "The effects of atmospheric turbulence in optical
  astronomy." Prog. Opt. 19, 281 (1981).
- Hardy, J. W. "Adaptive Optics for Astronomical Telescopes" (1998).
