# Fringe-Tracking Disturbance Notes

This note pins the analytic values used by `tests/physics/test_fringe_tracking.py`
for the inter-telescope piston and vibration residual primitives.

## Von Karman Differential Piston

The uncorrected piston difference between two circular apertures of diameter
`D` whose centers are `B` apart follows from the von Karman phase spectrum
(Conan, Rousset, and Madec 1995):

```text
W(f)    = 0.023 r0^(-5/3) (f^2 + L0^-2)^(-11/6)
sigma^2 = 2 int W(f) [1 - J0(2 pi f B)] [2 J1(pi D f) / (pi D f)]^2 d^2f
        = 4 pi 0.023 r0^(-5/3) int_0^inf f (f^2 + L0^-2)^(-11/6) A(f) [1 - J0] df
```

in rad^2 at 500 nm, converted to OPD with `lambda / 2pi`. Each aperture's
piston is the mean over its full clear disk. Because `r0(lambda) ~
lambda^(6/5)`, the OPD is achromatic and the 500 nm evaluation is exact.

Limits pinned by the tests:

- `B -> inf, D -> 0`: `[1 - J0] -> 1` and the radial integral is
  `(3/5) L0^(5/3)`, so `sigma^2 = (12 pi / 5) 0.023 (L0/r0)^(5/3) =
  0.1734 (L0/r0)^(5/3)` rad^2. At `r0 = 0.15 m, L0 = 25 m` that is
  875.2 rad^2, `2.354 um` RMS.
- `B << L0, D -> 0`: the Kolmogorov structure function `6.88 (B/r0)^(5/3)`.
  At `B/L0 = 1e-9` the outer-scale correction is about `-1.5 (B/L0)^(1/3)
  = -0.15%`, and the rounded 0.023 prefactor sits 0.44% above the 0.02290
  that 6.88 implies, so the code returns about 1.003 times 6.88.
- `r0 = 0.15 m, L0 = 25 m, B = 85 m` (Keck-Keck): `2.354 um` for points,
  `1.688 um` for 10 m primaries, `2.292 um` for 1.4 m subapertures, from an
  independent adaptive-quadrature integration (review 2026-11-07 Issue 3).
  At 85 m the result is within 0.01% of the `B -> inf` value.

The code integrates on a grid that is geometric below `1/max(B, D)` and
linear above it, with 40 samples per Bessel period, and adds the closed-form
point tail above the cutoff.

The earlier implementation used `sigma_OPD = 2.62 (lambda/2pi)
(L0/r0)^(5/6)`, the Kolmogorov structure function evaluated at `B = L0`
(`2.62^2 = 6.88`). That is not the saturated von Karman value: it gave
14.81 um, 39.6 times the point-aperture variance and 77 times the 10 m
aperture variance.

The `fringe_tracking.closed_loop` stage passes the horizontal pair separation
from `geometry.aperture_positions_enu_m` and the full primary diameter
`geometry.diameter_m`, and averages the variance over telescope pairs. With
no aperture positions it uses `B -> inf`, the upper bound.

## Flat-Knee Servo Residual

The simplified atmospheric-piston temporal PSD is flat up to `f_k`, then falls
as `f^(-17/3)`. Let the flat level be `P0`:

```text
S(f) = P0                         0 <= f <= f_k
S(f) = P0 * (f / f_k)^(-17/3)     f > f_k
```

The total variance is

```text
int S(f) df = P0 f_k + P0 f_k int_1^inf x^(-17/3) dx
            = P0 f_k * (1 + 3/14)
            = P0 f_k * 17/14.
```

For a closed-loop rejection `|1 - H|^2 = (f / f_c)^2` below `f_c`, and unity
above `f_c`, in the regime `f_c >> f_k`:

```text
residual = P0 int_0^f_k (f/f_c)^2 df
         + P0 int_f_k^f_c (f/f_c)^2 (f/f_k)^(-17/3) df
         + P0 int_f_c^inf (f/f_k)^(-17/3) df
         = P0 f_k^3/f_c^2 * (1/3 + 3/8 + 3/14)
         = P0 f_k^3/f_c^2 * 13/14.
```

Dropping the small high-frequency leak above `f_c` gives the conservative
low-frequency rejection approximation used by the classical stage model:

```text
residual = P0 f_k^3/f_c^2 * (1/3 + 3/8)
         = P0 f_k^3/f_c^2 * 17/24.
```

Relative to the total variance this is

```text
(17/24) / (17/14) = 7/12,
```

so

```text
sigma_residual^2 = (7/12) * sigma^2 * (f_k / f_c)^2.
```

With the Keck-Keck `sigma = 1.688 um`, `f_k = 0.6656 Hz`, and `f_c = 50 Hz`,
the residual is `sqrt(7/12) * 1688.1 nm * (0.6656/50) = 17.16 nm`.

## Vibration Lines And Broadband Floor

The line-spectrum defaults represent residual-after-feedforward OPD amplitudes.
Keck vibration tones at 29 Hz and 58 Hz are anchored in Colavita et al. (2013)
and the associated feedforward work by Woillez et al. (2012). A 12 Hz
wind-excited term follows the LBTI HOSTS context in Ertel et al. (2020), and
150 Hz / 200 Hz coudé-class lines follow Laugier et al. (2024).

For each line, the classical servo leaves the fraction
`min(1, (f_line / f_c)^2)` of the line variance. For the test set
`(12, 30), (29, 40), (58, 25), (88, 20), (150, 10), (200, 10)` at `f_c=100 Hz`,
the residual line variance is

```text
12.96 + 134.56 + 210.25 + 309.76 + 100 + 100 = 867.53 nm^2,
```

or `29.454 nm` RMS.

The broadband vibration floor is modeled as a flat PSD from 0 to `f_max`. At
`f_c < f_max`, the residual fraction is

```text
(int_0^f_c (f/f_c)^2 df + int_f_c^f_max 1 df) / f_max
= (f_c/3 + f_max - f_c) / f_max.
```

With `f_c=100 Hz`, `f_max=500 Hz`, and `30 nm` RMS broadband floor, the residual
floor variance is `30^2 * 0.86667 = 780 nm^2`; adding the line variance gives
`sqrt(1647.5) = 40.59 nm`.

## Photon-Kalman Structure Function Budget

The frame-free photon-counting model minimizes an estimate-plus-predict budget
over an averaging window `T`:

```text
sigma_res^2(T) = crlb^2 / T + D(T + tau_latency) / 2 + N_line * floor_line^2.
```

For the same flat-to-knee / `f^(-17/3)` PSD above, the small-lag structure
function follows from

```text
D(tau) = 2 int S(f) [1 - cos(2 pi f tau)] df
       -> (2 pi tau)^2 int S(f) f^2 df.
```

The second moment of the atmospheric PSD is

```text
int S(f) f^2 df = P0 f_k^3 (1/3 + 3/8)
                = P0 f_k^3 * 17/24.
```

Using the normalization `sigma^2 = P0 f_k * 17/14`, this becomes

```text
D(tau) -> (7/12) sigma^2 (2 pi f_k tau)^2.
```

This is the same `7/12` coefficient used by the classical servo residual. For
`sigma=1`, `f_k=0.6656 Hz`, and `tau=1e-4 s`, it gives
`0.5833 * (4.182e-4)^2 = 1.020e-7`. At lags long compared with the knee,
`D(tau)` saturates to `2 sigma^2`.

The boxcar estimator plus predictor is intentionally conservative relative to a
full causal Kalman/Wiener smoother by roughly a factor of 1.5-2 in variance.
That conservatism absorbs the usual classical closed-loop noise gain so it is
not counted again. Narrow vibration lines are treated as explicit oscillator
states and contribute `N_line * floor_line^2`, while the broadband vibration
floor remains inside `D(tau)`.

## Citations

- Absil, O. et al. 2006, A&A 448, 787.
- Bohm, M. et al. 2014, MNRAS 442, 2446.
- Choquet, E. et al. 2014, A&A 569, A2.
- Colavita, M. M. et al. 2013, PASP 125, 1226.
- Conan, J.-M., Rousset, G., and Madec, P.-Y. 1995, JOSA A 12, 1559.
- Ertel, S. et al. 2020, AJ 159, 177.
- Hanot, C. et al. 2011, ApJ 729, 110.
- Kellerer, A. and Tokovinin, A. 2007, A&A 461, 775.
- Lacour, S. et al. 2019, A&A 624, A99.
- Laugier, R. et al. 2024, A&A 686, A308.
- Linfield, R. P., Colavita, M. M., and Lane, B. F. 2001, ApJ 554, 505.
- Males, J. R. and Guyon, O. 2018, JATIS 4, 019001.
- Martinache, F. and Ireland, M. J. 2018, A&A 619, A87.
- Menu, J. et al. 2012, A&A 541, A81.
- Mennesson, B. et al. 2011, ApJ 743, 178.
- Nowak, M. et al. 2024, A&A 684, A35.
- Ono, Y. H. et al. 2016, MNRAS 465, 4931.
- Poyneer, L. A. et al. 2016, Appl. Opt. 55, 323.
- Snyder, D. L. 1972, IEEE Trans. Inf. Theory 18, 91.
- Walter, A. B. et al. 2020, Proc. SPIE 11447.
- Woillez, J. et al. 2012, Proc. SPIE 8445.
- Yonezawa, H. et al. 2012, Science 337, 1514.
