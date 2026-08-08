# High Dispersion Coronagraphy CCF Notes

This note derives the matched-filter cross-correlation budget implemented in
`nullsim/physics/hdc.py` and pinned by `tests/physics/test_hdc.py`. The framing
follows Wang, Mawet, Ruane, Hu and Benneke (2017), AJ 153, 183.

## Why HDC Helps

A photonic nuller delivers a dark port containing a small planet rate `P` on top
of a much larger leaked stellar rate `L`. Broadband, the contrast is set by how
well `L` is known. High dispersion coronagraphy exploits a spectral asymmetry
instead: the planet spectrum is a forest of narrow molecular lines, whereas the
leaked starlight is the (smooth) stellar continuum times a smooth instrument
response. Subtracting a running median removes the smooth part almost entirely
and leaves the line forest, which is then correlated against a template.

## Channelization

Rates arrive as high-resolution densities in Hz/um on a constant-R log grid.
`channelize` degrades to the instrument resolving power with a Gaussian LSF and
then bin-averages onto the channel edges, multiplying by the channel widths to
get Hz per channel.

Order matters. `resample_flux_conserving` returns bin *means*, which is already
a boxcar average over each channel. Convolving after resampling would apply the
LSF on top of that boxcar and over-smooth; convolving first, on the log-uniform
grid where a Gaussian LSF has a sample-constant width, is the correct sequence.
Because both operations conserve the band integral, a flat density `f` over a
band of width `W` must channelize to a total rate `f W`. That identity is the
first pinned test.

## Matched Filter

Per channel `i`, with Poisson statistics the variance rate equals the mean rate:

```text
v_i = P_i + L_i + Sky_i + Dark_i     [Hz]
```

Let `m_i` be a telluric keep-mask and let a tilde denote continuum filtering
(subtraction of a running median of odd width >= 3). Treating channels as
independent, the linear statistic `X = sum_i a_i N_i` with maximum SNR has
weights proportional to (template)/(variance), giving

```text
w_i = m_i * Ptilde_i / v_i           [dimensionless]
S   = t * sum_i w_i Ptilde_i         [counts]
V_ph = t * sum_i w_i^2 v_i           [counts^2]
```

With those weights `V_ph = S` identically, so

```text
SNR_photon = S / sqrt(V_ph) = sqrt(t * sum_i Ptilde_i^2 / v_i).
```

Two limits follow directly and are both pinned.

*Flat template.* One channel-independent planet rate `P_i = S_tot/n` against a
flat background `L_i = B_tot/n` gives `w_i = S_tot/(S_tot + B_tot)` for every
channel and

```text
SNR = S_tot t / sqrt((S_tot + B_tot) t),
```

the broadband photometric result. HDC costs nothing when there is no spectral
structure to exploit.

*N identical lines.* `N` lines of equal height on a flat leak `L >> P` give
`w_i = P/L` on the line channels and zero elsewhere, so `SNR^2 = N t P^2 / L`
and `SNR ~ sqrt(N)`. This is the Wang et al. line-count scaling; the test
compares 100 lines against 25 and requires a ratio of 2.

## Systematic Term And Why `B_w` Replaces The Band Leak

The broadband stage `nullsim/stages/performance/standard.py` models null
instability as a fractional fluctuation `eps` on the leaked starlight and splits
into two regimes in `_systematic_variance`:

```text
fringe tracking on:   V_sys = eps^2 B^2 t / (2 f_servo)       (grows like t)
fringe tracking off:  V_sys = (eps B t)^2                     (grows like t^2)
```

with `B` the integrated dark-port leakage rate. The CCF module keeps both
regimes verbatim and changes only what plays the role of `B`.

The reason is that the CCF statistic is not a photon count; it is a projection.
A coherent fractional excursion `eps` multiplies the leak spectrum as
`L_i -> (1 + eps) L_i`, and what that injects into `X = sum_i w_i N_i` is

```text
delta X = eps * t * sum_i w_i Ltilde_i = eps * t * B_w,   B_w = sum_i w_i Ltilde_i [Hz].
```

So the template projection `B_w`, not the raw band counts `B`, sets the
systematic amplitude. This is the whole point of the technique: a null
instability that is spectrally smooth is annihilated twice, once by the
continuum filter (`Ltilde` is small) and again by the projection onto a template
that is orthogonal to whatever smooth residual survives. The broadband treatment
has no way to express either cancellation, which is why it needs an `eps` chosen
by hand as a stand-in for an unmodeled calibration floor, while the CCF version
computes the corresponding quantity.

Two tests pin the projection directly. With planet lines on channels
`10, 30, 50, ...` and leak lines on the disjoint set `3, 23, 43, ...`, the
filtered leak has support only where the weights vanish, so `B_w = 0` to machine
precision. With the leak lines sitting exactly on the planet lines,
`B_w > 0` and `V_sys > 0`. A third test drives the no-fringe-tracking regime to
long integrations: `S ~ t` while `sqrt(V_sys) ~ t`, so the SNR saturates rather
than growing as `sqrt(t)` -- the ratio between `t = 1e6 s` and `t = 1e7 s` is
required to stay under 1.05.

## Telluric Masking

Channels whose atmospheric transmission falls below a threshold are removed by
zeroing their weights. Because `SNR^2 = t sum_i m_i Ptilde_i^2 / v_i` is a sum
of non-negative terms, dropping channels can only reduce the SNR, and raising
the threshold produces a nested sequence of masks. The pinned test therefore
requires both the channel count and the SNR to fall monotonically as the
threshold tightens. Masking is a mask only: the transmission is assumed to be
already folded into the rates.

## Self-Subtraction And `N_eff`

The continuum filter removes some of the planet signal along with the continuum:
broad planet features are indistinguishable from a stellar continuum. The
quantity reported is an **SNR-retention ratio, not a flux ratio** -- the
photon-limited matched-filter SNR of the filtered template divided by that of the
unfiltered one, on the same unmasked channels and the same noise budget:

```text
self_subtraction_factor = sqrt(sum_i Ptilde_i^2 / v_i) / sqrt(sum_i P_i^2 / v_i).
```

The obvious alternative, the weighted flux ratio
`sum_i w_i Ptilde_i / sum_i w_i P_i`, does not work. For an
absorption-dominated template it changes sign, because `Ptilde` is negative in
the lines while `P` is positive, and its denominator can pass through zero. The
SNR ratio is non-negative by construction and answers the question a filter
width is actually traded against.

Note that the matched filter itself is untroubled by absorption: the signal
`S = t sum_i Ptilde_i^2 / v_i` is positive whether the lines are in emission or
absorption, since a negative template correlates with a negative data excess.
Only the *summary diagnostic* needed the quadratic form.

### It is not bounded by 1, and should not be

The name invites the reading "fraction retained", but there is no hard `(0, 1]`
bound and none should be imposed. Take `P = [5, 0.1, 5, 0.1, 5, 0.1, 5]` on a
flat leak of 100 with a width-3 median filter. The running median of an
alternating sequence returns the *opposite* level at every interior sample, so

```text
Ptilde = [0, -4.9, +4.9, -4.9, +4.9, -4.9, 0],
```

and the factor comes out at 1.11147. That is the correct answer. The unfiltered
template is a bright, nearly flat spectrum whose matched-filter SNR is limited
by how little it varies; the filter converts narrow absorption dips riding that
bright continuum into large signed excursions about zero, and a template with
large signed excursions carries more matched-filter SNR than the near-flat one
it came from. Deep absorption on a bright continuum is exactly the regime where
high dispersion helps most, so a diagnostic that reported "1.0, nothing gained"
there would be hiding the effect the technique exists to exploit.

The honest statement is therefore: non-negative, typically at or below 1 for
emission-line-like templates, and above 1 when absorption structure dominates
over a bright continuum. Anything that clamps it is throwing away information.

### Conventions at the edges

- `width <= 0` (filtering bypassed) returns exactly 1.0 *unconditionally*.
  `Ptilde` is `P` by construction, so nothing was lost, and this is stated
  directly rather than computed -- otherwise an all-zero template would take
  the `0/0` path and report 0.0, which would be wrong.
- Filtering active with an all-zero template (`sum_i P_i^2/v_i = 0`) returns
  0.0 by convention. There is no signal whose retention could be measured, and
  0.0 is the value that will not be mistaken for a healthy detection.

Four cases are pinned: the reviewer's alternating counterexample at its exact
value 1.11147, sparse absorption dips on a continuum (where the factor is a
small fraction, ~0.18), the bypass path with an all-zero template, and the
active-filter zero-denominator path.

The effective line count is the participation ratio of the weights,

```text
N_eff = (sum_i |w_i|)^2 / sum_i w_i^2,
```

which equals `n` for `n` equally weighted channels and equals 1 when a single
channel dominates. It answers "how many channels is this detection actually
leaning on", which is the honest count to quote next to a sqrt(N) scaling
argument.

## Radial-Velocity Scan

`ccf_vs_rv` correlates a fixed observed spectrum (planet at its true radial
velocity plus leak, sky, dark) against templates rebuilt at trial velocities.
The statistic is the *template-normalized* matched-filter score

```text
CCF(rv) = sqrt(t) * (sum_i w_i(rv) Dtilde_i) / sqrt(sum_i w_i(rv)^2 v_i),
w_i(rv) = m_i Ttilde_i(rv) / v_i,
```

with `v_i` and `Dtilde` taken from the observation and held fixed across the
scan.

The normalization is load-bearing. The bare numerator is `T^T V^-1 D`, which
scales with the template norm, so its value can be raised simply by giving the
template more power inside the mask. Concretely: put planet lines of strength
`[1, 1, 10]` at channels `[40, 60, 80]`, mask channel 80, and scan integer
shifts. At the true shift the strong line is masked away and the numerator is
`2/V`; at a shift of `-20` the template's strength-10 line lands on a
strength-1 data line and the numerator is `11/V`. The bare numerator picks the
wrong velocity. Dividing by `sqrt(T^T V^-1 T)` turns the statistic into a
Cauchy-Schwarz cosine in the `V^-1` metric, bounded by `sqrt(D^T V^-1 D)` and
attained only when `T ∝ D`, so the true velocity wins: the same example gives
`0.141` at the true shift against `0.108` and `0.0099` at the competitors. Both
halves of this are pinned -- the test asserts that the unnormalized numerator
peaks at `-20` and that the shipped statistic peaks at 0.

Two further requirements are enforced rather than assumed. The fixed data
spectrum is a required input: a scan that shifts the template and rebuilds the
weights and the "data" together would be an autocorrelation and would be flat in
`rv`. And the trial closure must return channels on the data's wavelength grid;
`_require_same_grid` compares edges and centers to `rtol = 1e-9` and raises,
because a same-length grid at different wavelengths would otherwise be
correlated index by index and quietly return a wrong answer.

## Contrast At A Target SNR (Detection Mode)

`ccf_contrast_at_target_snr` inverts the matched filter: rather than scoring a
known planet, it asks how faint a planet the same observation would still detect
at SNR `n`. That inversion is what a CCF contrast curve is made of, so it runs
once per separation anchor and has to be cheap.

Let `P1_i` be the per-channel planet rate at *unit* band-integrated contrast, so
the actual planet rate is `c P1_i`. The planet-free variance rate is
`v0_i = L_i + sky_i + dark_i`. With weights frozen at zero contrast,

```text
w_i = m_i Ptilde1_i / v0_i,
A   = sum_i w_i Ptilde1_i      [Hz per unit contrast]
B2  = sum_i w_i^2 v0_i         [Hz]
C2  = sum_i w_i^2 P1_i         [Hz per unit contrast]
B_w = sum_i w_i Ltilde_i       [Hz]
```

the signal is `c A t`, the photon variance is `t B2 + c t C2` (the planet's own
shot noise is the `C2` term), and `V_sys` is whichever of the two regimes
applies. So

```text
SNR^2(c) = c^2 A^2 t^2 / (t B2 + c t C2 + V_sys).
```

Setting that equal to `n^2` and clearing the denominator gives a quadratic,

```text
(A^2 t^2) c^2 - (n^2 t C2) c - n^2 (t B2 + V_sys) = 0,
```

whose positive root is

```text
c = [n^2 t C2 + sqrt(n^4 t^2 C2^2 + 4 A^2 t^2 n^2 (t B2 + V_sys))] / (2 A^2 t^2).
```

The constant term is `<= 0` and the leading coefficient is `> 0` whenever
`A > 0`, so the discriminant is non-negative and exactly one root is positive.
No iteration is needed, and that is only true because `V_sys` does not depend on
`c`: both regimes scale with `B_w`, which is a property of the leak and the
template, not of the planet.

### Solving it without overflowing

Evaluating that root literally does not survive contact with realistic input
ranges. `A^2 t^2` overflows for `A ~ 1e150`, raises `OverflowError` outright at
`t = 1e160`, and flushes to zero for rates around `1e-200` — all of which are
representable float64 inputs whose exact roots are perfectly ordinary numbers.
The implementation therefore substitutes `c = c0 x` with the photon-limited
root

```text
c0 = n / sqrt(A t),
```

reducing the quadratic to

```text
x^2 - p x - q = 0,   p = c0 C2 / A,   q = B2/A + V_sys/(A t),
```

which removes the `A t` and `A^2` products. It is *not* true that the reduced
coefficients then sit near unity — `p` and `q` each still span the dynamic
range of the inputs, and `q` overflows on its own whenever a large `V_sys`
meets a small `A`. What actually keeps the arithmetic in range is that every
surviving term enters through a square root, which halves its exponent:

- `c0` is computed as `(n / sqrt(A)) / sqrt(t)`, never `n / sqrt(A t)`.
- `q` is never materialized. Only `sqrt(q)` is needed, and it is built as
  `hypot(sqrt(B2/A), sqrt(V_sys) / (sqrt(A) sqrt(t)))`. The systematic term has
  to be grouped that way rather than as `sqrt(V_sys / (A t))`: `V_sys / A`
  alone overflows for `A ~ 1e-150` against `V_sys ~ 1e300`, which is reachable
  with `P1 = 1e10`, `L = 1e170`, `t = 1e140`, `eps = 1` — a case whose true
  root, `5e160`, is perfectly representable.
- The discriminant is `hypot(p, 2 sqrt(q))` rather than `sqrt(p^2 + 4q)`, so
  neither argument is ever squared.

Finally the returned root is checked finite and positive, and raises with the
full coefficient set in the message if it is not.

Four magnitudes are pinned as tests, each against its analytic limit:
`n / sqrt(A t)` in the two photon-limited cases; `n^2 C2 / (A^2 t)` in the
`1e-200` case, where a second of integration collects no photons at all and the
planet's own shot noise sets the root; and `5e160` in the large-`V_sys` case
above.

One step further out, `V_sys` itself stops being representable: it is
`(eps B_w t)^2` without fringe tracking, so `eps B_w t > ~1.3e154` overflows.
Python's `**` raises `OverflowError` there while the fringe-tracking branch's
plain multiplications saturate silently to `inf`. The solver catches both and
re-raises as the `ValueError` it documents, quoting `eps B_w t`, so callers
sweeping anchors have one exception type. `matched_filter_ccf` is deliberately
left alone — changing what it raises would be a behavior change to a function
with its own consumers.

### Two identities worth knowing

`A = B2` analytically -- both equal `sum_i Ptilde1_i^2 / v0_i`, since `w_i` is
`Ptilde1_i / v0_i` on the usable channels. (They are computed by different
summations, `sum w Ptilde1` and `sum w^2 v0`, so they agree to rounding rather
than bit for bit; do not assert equality on them.) The photon-limited limit
(`C2 -> 0`, `V_sys -> 0`) therefore collapses to

```text
c_photon = n / sqrt(A t),
```

which is the `1/sqrt(t)` scaling a contrast curve should show, with `A` playing
the role of an effective line-weighted rate.

The flat-template limit reproduces the broadband answer. With `P1_i = p` and
`L_i = l` on all `n_ch` channels, no filtering and no mask, the quadratic
reduces term by term (after multiplying through by `l^2 / p^2`) to

```text
c^2 (n_ch p t)^2 - n^2 c (n_ch p t) - n^2 (n_ch l t) = 0,
```

which is `S^2 = n^2 (S + B)` with `S = c P_tot t` and `B = L_tot t`. The pinned
test compares against the closed-form root of that scalar quadratic at
`rel = 1e-10`.

### The frozen-weights caveat

The weights are evaluated at `c = 0`, so `v0` excludes the planet's own
contribution to the *weighting* while `C2` still carries the planet shot noise
into the denominator. Exact weights would depend on the answer, which is
precisely what the closed form avoids.

**Photon-limited, the error is second order.** Expanding the true statistic in
`c P1 / v0` with `V_sys = 0`,

```text
t c^2 sum_i Ptilde1_i^2 / (v0_i + c P1_i) = t c^2 (A - c C2 + O(c^2)),
```

and the frozen version,

```text
c^2 A^2 t / (B2 + c C2) = t c^2 A / (1 + c C2 / A) = t c^2 (A - c C2 + O(c^2)),
```

agree through the linear term. The residual is `O((c C2 / A)^2)`, i.e. roughly
the square of the planet/leak ratio. That is why the roundtrip test asserts only
`rel = 2e-2` at a leak of `1e3` per channel and tightens to `rel = 1e-10` at
`1e7`: raising the leak by `1e4` raises `c` by roughly `1e2` (since
`c ~ sqrt(L)` in the photon limit) and so drops `c P1 / v0` by roughly `1e2` --
measured, `8.3e-4 -> 1.0e-5` -- which drops the squared error by about four
orders of magnitude.

**With a nonzero systematic, the error is generically first order.** It
vanishes when the structural coefficient below is zero, and the frozen
statistic is exact to all orders when `c P1_i / v0_i` is channel-independent.
Outside those cases the cancellation that holds at `V_sys = 0` is a property of
that denominator specifically -- it is not a statement about systematics being
"small enough". Writing
`G(c) = sum_i Ptilde1_i^2 / v_i` and `H(c) = sum_i Ptilde1_i Ltilde_i / v_i`,
the exact statistic is

```text
SNR^2(c) = c^2 t^2 G(c)^2 / (t G(c) + K H(c)^2),
```

with `K = eps^2 t / (2 f)` or `eps^2 t^2`. (The matched filter is invariant
under an overall rescaling of the weights, which is why the same `c` appears
in the solver's per-unit-contrast form and in `matched_filter_ccf`'s
absolute one.) With `G = A - c C2 + O(c^2)` and `H = H0 - c D + O(c^2)`,
`D = sum_i Ptilde1_i Ltilde_i P1_i / v0_i^2`, expanding at general `K` gives

```text
SNR^2_true / SNR^2_frozen = 1 - 2 c eta_sys (C2/A - D/H0) + O(c^2),
eta_sys = K H0^2 / (t A + K H0^2) = V_sys / (t B2 + V_sys),
```

using `A = B2`. So `eta_sys` is simply the systematic share of the
zero-contrast variance. It tends to 1 in the systematics-dominated limit, which
recovers `1 - 2c(C2/A - D/H0)`, and falls smoothly to 0 as the budget becomes
photon-limited. The first-order term therefore switches on gradually with the
systematic fraction rather than at a threshold — there is no regime with a
nonzero `V_sys` in which the error is second order.

The statistic itself is `SNR`, not `SNR^2`, so the observed fractional error is
half the coefficient above:

```text
1 - SNR_true/SNR_frozen  ~=  c eta_sys (C2/A - D/H0).
```

`C2/A - D/H0` is the second gate, and it vanishes identically for a flat
template against a flat leak.

There is a stronger statement available, but it needs a stronger hypothesis:
**the frozen statistic is exact to all orders when `x_i = c P1_i / v0_i` is
channel-independent.** That is what makes the substitution below legal, since
it requires factoring a single common `(1+x)` out of each sum:

```text
G = A/(1+x),   H = H0/(1+x),   B2 + c C2 = A(1+x),

exact = c^2 t^2 (A/(1+x))^2 / (t A/(1+x) + K H0^2/(1+x)^2)
      = c^2 t^2 A^2 / (t A (1+x) + K H0^2) = frozen.
```

Flat `P1` and flat `L` do *not* suffice on their own: `v0 = L + sky + dark`, so
structured sky or dark makes `v0` vary, `x_i` vary with it, and the factoring
fails. Measured on the same flat probe (`eta_sys = 0.9996`,
`c P1 / v0 = 0.025`, a *larger* planet/leak ratio than the structured probe's
0.017 and more systematic-dominated):

| sky | `x` range | roundtrip error |
|---|---|---|
| none (`v0` uniform) | `[0.02501, 0.02501]` | `4.4e-16` |
| alternating 0/80 Hz | `[0.02084, 0.02501]` | `1.0e-9` |

The first row is machine epsilon — exact. The second shows the first-order
cancellation surviving (`1e-9` against the structured probe's `3.6e-2`) while
exactness does not: the residual re-enters at second order through the
structural coefficient.

So the bias needs *both* a nonzero systematic and spectral structure. It is
driven by structure, not by magnitude, and the honest statement is "generically
first order", not "first order whenever `V_sys > 0`".

Measured on an 80-channel structured probe (`P1` a modulated line forest, `L`
a modulated continuum, width-11 filter, no fringe tracking, `t = 3600 s`),
where `C2/A - D/H0 = 0.02247` analytically:

| `eps` | `eta_sys` | `c` | roundtrip SNR / target | error / (`c` `eta_sys`) |
|---|---|---|---|---|
| 0 | 0 | 0.176 | 1.000000 | — |
| 0.005 | 0.189 | 0.195 | 0.999172 | 0.02245 |
| 0.010 | 0.483 | 0.244 | 0.997359 | 0.02241 |
| 0.020 | 0.789 | 0.382 | 0.993282 | 0.02230 |
| 0.050 | 0.959 | 0.866 | 0.981731 | 0.02200 |
| 0.100 | 0.989 | 1.705 | 0.963674 | 0.02154 |

The last column is the measured first-order coefficient. It is constant to 4%
across a five-fold range in `eta_sys` and a nine-fold range in `c`, and it
agrees with the analytic `C2/A - D/H0 = 0.02247` to better than 1% at small
`c`. Note the `eta_sys = 0.189` row: systematics are a *minority* of the
variance there and the error is still first order, just small.

Two tests pin this. One asserts the systematics-dominated roundtrip lands in
`[0.90, 1.02]`, is *not* within `rel = 2e-2` of the target, and doubles when
`eps` doubles. The other checks the `eta_sys` scaling directly, requiring
`error / (c eta_sys)` to agree within 5% between a weak-systematic case
(`eta_sys ~ 0.19`) and a dominated one (`eta_sys ~ 0.99`), and requiring the
weak case to have a strictly nonzero error.

The practical statement: the returned contrast is reliable wherever the planet
is faint against the leak *and* the budget is not deep in the
systematics-dominated corner. Where it is, it is optimistic by a few percent,
and `matched_filter_ccf` is the cross-check.

### Failure convention

Every unusable input raises `ValueError`, never `OverflowError` and never a
sentinel value. That matters because the caller sweeps separation anchors and
wants to record a non-detection for a bad anchor without aborting the curve, so
it needs exactly one exception type to catch.

`A <= 0` raises `ValueError` rather than returning `inf`. `A` is
`sum_i Ptilde1_i^2 / v0_i` over usable channels, so it is zero only when the
filtered template vanishes everywhere it is allowed to count: a lineless planet
spectrum, or one whose lines all land in telluric-masked channels. There is no
contrast at which such a template reaches the target, and an `inf` propagating
into a contrast curve would be read as "no constraint" rather than "the input
was unusable". The other guards (`t > 0`, `n > 0` finite, finite non-negative
rates of equal length, and the shared `eps`/`servo_bw` checks) are the same ones
`matched_filter_ccf` applies, via the extracted `_validate_noise_params`.

## Citations

- Wang, J., Mawet, D., Ruane, G., Hu, R., and Benneke, B. 2017, AJ 153, 183.
- Snellen, I. A. G. et al. 2015, A&A 576, A59.
- Mawet, D. et al. 2017, ApJ 838, 92.
- Sparks, W. B. and Ford, H. C. 2002, ApJ 578, 543.
