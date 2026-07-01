# Ruilier-Cassaing coupling notes

Reference files read in full:

- `kilo/injection.py`
- `nayra/injection.py`

## KILO and NAYRA source behavior

Both codebases use the same reduced Ruilier-Cassaing coupling model for
single-mode fiber injection:

```text
eta = strehl * eta_opt * fresnel_transmission
eta_opt = 0.81 * (1 - epsilon^2)^2
fresnel_transmission = 1 - 0.035
```

where `epsilon` is the linear central-obscuration ratio. With Fresnel disabled,
both return `0.405000000000` for `strehl = 0.5` and
`central_obstruction = 0.0`. With Fresnel enabled, both return
`0.390825000000`. For `epsilon = 0.3`, both reduced models return
`0.670761000000` for the geometric term.

KILO uses this in `compute_coupling_per_subaperture()` per telescope. NAYRA
collapses that to one scalar or wavelength vector Strehl broadcast over all
sub-apertures.

## Analytical geometric limit

The more precise unobstructed circular-aperture / Gaussian-mode overlap from
the Ruilier-Cassaing pupil-plane integral is approximately `0.8145`. KILO and
NAYRA both round this to `0.81`. `nullsim` uses the analytical value for the
pure physics primitive while retaining a sub-percent parity test against the
source value.

Using a Gaussian pupil-plane field `exp(-(rho r / R)^2)`, the overlap is:

```text
eta_geo(rho, epsilon) =
    2 * (exp(-(rho * epsilon)^2) - exp(-rho^2))^2
    / (rho^2 * (1 - epsilon^2))
```

For an unobstructed pupil, the optimum is at `rho ~= 1.1209`, giving
`eta_geo ~= 0.814528755`. This `rho` is the aperture-radius / Gaussian-mode
radius ratio in the pupil-plane expression. Some Ruilier summaries describe
the focal-plane matched mode scale with the reciprocal convention, so the
constant name in nullsim spells out the ratio.

## Strehl and phase-variance convention

KILO computes Strehl from residual phase variance via the Marechal
approximation:

```text
S = exp(-sigma2_rad2)
```

where `sigma2_rad2` is positive residual phase variance in radians squared.
NAYRA re-exports the KILO atmosphere implementation and its diagnostic plot
uses the same `S = exp(-sigma^2)` convention. Neither codebase uses a
Born-Wolf alternative in the injection module.

## Central obstruction handling

KILO and NAYRA both expose a central-obstruction parameter in the injection
coupling function, but their normal sub-aperture paths pass `0.0` because the
sub-apertures are placed in clear pupil. The reduced source formula is
`0.81 * (1 - epsilon^2)^2`; the nullsim primitive evaluates the Gaussian
overlap integral for the supplied obstruction ratio and validates
`0 <= epsilon < 1`.

## Wavelength dependence

The source `coupling_efficiency()` functions have no direct wavelength
argument. Wavelength dependence enters upstream through the AO Strehl
calculation and, for off-axis coupling, through the Gaussian mode size in
`angular_coupling_factor()`. KILO and NAYRA use
`OPTIMAL_MODE_FIELD_RATIO = 0.6436` in that angular rolloff helper. The pure
on-axis nullsim primitive keeps Fresnel and fiber-transport wavelength effects
out of this module; those belong in later transport/stage work.
