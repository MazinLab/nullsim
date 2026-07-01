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

The canonical Noll 1976 result for the total piston-included wavefront
phase variance over a circular aperture of diameter D is

    sigma_phi^2 = 1.0299 (D / r0)^(5/3)    rad^2    (Noll Table I)

After the first Zernike (piston) is removed:

    sigma_phi^2 = 0.582 (D / r0)^(5/3)    rad^2    (Noll Table II)

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
(uncorrected total variance over a full aperture vs. fitting error
residual after a deformable mirror with actuator pitch d_act).

Because the canonical Noll 1976 value is not duplicated in KILO/NAYRA,
**no KILO/NAYRA-vs-canonical discrepancy exists for the total-variance
coefficient**. nullsim adopts the Noll 1976 value 1.0299 directly,
citing Noll Table I, and the 0.582 piston-removed value from Noll
Table II as an opt-in via `piston_removed=True`.

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

    phase_variance(aperture_diameter_m, r0_m, *, piston_removed=False)
    opd_rms_m(aperture_diameter_m, r0_m, wavelength_m, *, piston_removed=False)

Rationale:

- r0 already encodes the wavelength via Fried scaling. Callers who have
  r0 at a different wavelength scale it with `r0_scaled` first.
- `opd_rms_m` keeps `wavelength_m` for the explicit phase-to-OPD
  conversion factor `lambda / (2 pi)` and so the API documents the
  dimensioned conversion at the call site.
- The piston-removed Noll coefficient is opt-in via keyword. This keeps
  the default behavior aligned with the textbook Noll Table I value
  (1.0299) while exposing the piston-removed Table II value (0.582)
  without proliferating function names.

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
