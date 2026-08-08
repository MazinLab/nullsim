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

## The Continuum Filter Is A Linear Projection

The high-pass is

```text
H = I - S,      S = diag(1/c) K,     c = K 1,
```

with `K` the zero-padded boxcar of odd width `m >= 3`:
`K_ij = 1/m` for `|i - j| <= m//2`, truncated to the array. Two properties do
all the work.

*`K` is symmetric.* Restricting a symmetric Toeplitz matrix to a principal
submatrix leaves it symmetric, so `K = K^T` exactly, and therefore

```text
S   x = K x / c,
S^T y = K (y / c),
```

one smoother application each. `H^T = I - S^T` follows. No dense `n x n` matrix
is ever built; `continuum_filter` and `continuum_filter_adjoint` are the two
calls, and a test pins the adjoint against the dense transpose at `1e-14`.

*`S` preserves constants exactly.* `S 1 = diag(1/c) K 1 = diag(1/c) c = 1`, at
the array edges as well as the interior, because `c` is precisely the in-array
fraction of each window. So `H 1 = 0` everywhere and a flat stellar continuum
is annihilated with no edge residual. (A ramp is killed only in the interior;
the edge residual is second order in the slope.)

### Why not a running median

A median rejects the lines themselves, so it estimates the continuum with less
self-subtraction. That is why it was the first choice here, and it was wrong.
The estimator is a weighted sum of *filtered* counts, and a median has no
closed-form covariance: filtered channels become differences of raw counts and
neighbours correlate. Pairing that signal model with a raw diagonal variance
understated the noise badly. Monte Carlo on the alternating-spectrum case
(200k Poisson realizations through the actual median):

```text
claimed Var (diagonal-raw) = 1176.9   -> SNR 34.31
measured Var               = 4275.7   -> SNR 18.00
                             3.63x variance undercount, 1.91x SNR optimism
```

A linear `H` costs more self-subtraction -- the lines do drag a mean, and the
filter leaves negative wings `m` channels wide around every line -- but it
propagates exactly, and the cost is then *reported* by
`self_subtraction_factor` instead of hiding in an unmodelled covariance. That
is the whole trade. Boxcar rather than Gaussian because its support is exactly
`m` channels, so the banded operator and its adjoint are exact with no kernel
truncation to reason about.

Worth noting that the median was not even winning on its own terms: measured
against its own true variance it retained ~0.58 of the unfiltered SNR on the
alternating case, against 0.637 for the boxcar. The apparent 1.111 it used to
report was an artifact of the mismatched variance.

## The Statistic, And What It Is Not

Per channel `i`, with Poisson statistics the variance rate equals the mean rate:

```text
v_i = P_i + L_i + Sky_i + Dark_i     [Hz]
```

Let `m_i` be a telluric keep-mask and `Ptilde = H P`. The statistic is the
**photon-weighted CCF against a fixed template**,

```text
w_i = m_i * Ptilde_i / v_i           [dimensionless]
X   = w . (H N)                      [counts]
S   = t * sum_i w_i Ptilde_i         [counts]
```

with weights set by the template and the photon variance alone.

### Covariance propagation

`Cov(N) = diag(v t)` for independent Poisson counts, so

```text
Var(X) = w^T H diag(v t) H^T w = t * sum_i u_i^2 v_i,     u = H^T w.
```

This is the correction the review forced. The raw diagonal `t sum_i w_i^2 v_i`
is the variance of `w . N`, not of `w . (H N)`, and the two differ by however
much the filter correlates the channels -- 23% on the width-3 line comb the MC
test pins, and it grows as the filter narrows. Note that `u` is generally
nonzero on telluric-masked channels even though `w` is zero there: masking a
channel out of the *sum* does not stop the filter pulling that channel's
photons into its neighbours, and the propagated variance says so.

Only when filtering is bypassed does `u = w`, and then `V_ph = S` identically:

```text
SNR_photon = S / sqrt(V_ph) = sqrt(t * sum_i Ptilde_i^2 / v_i).
```

### This is not the optimal filter

Under the full covariance `C = diag(v t) + K Ltilde Ltilde^T` -- photon noise
plus the rank-one systematic -- the maximum-SNR weights are `C^-1 Ptilde`, which
Sherman-Morrison gives in closed form. That filter actively projects out the
leak direction and beats the fixed-template statistic by roughly an order of
magnitude once systematics dominate.

It is documented here as an **upper bound, not implemented**, because it assumes
the leak spectrum `Ltilde` is known exactly. A filter that subtracts a leak it
knows perfectly will always look good on paper; what an instrument can actually
deliver is a leak estimate with its own errors, and the gain collapses with the
knowledge. Adopting it is a design decision about which assumption to make, not
a bug fix, so the module keeps the honest fixed-template statistic and labels
it as such. Everything published -- module docstring, stage model strings --
says "photon-weighted CCF", never "matched filter" unqualified.

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

The continuum filter costs the detection twice: it removes some of the planet
signal along with the continuum (broad planet features are indistinguishable
from a stellar continuum), and it correlates the photon noise. The reported
quantity is an **SNR-retention ratio, not a flux ratio** -- the photon-limited
SNR of the filtered statistic over the unfiltered one, on the same unmasked
channels and the same noise budget:

```text
                          (t sum_i w_i Ptilde_i) / sqrt(t sum_i u_i^2 v_i)
self_subtraction_factor = --------------------------------------------------,
                                  sqrt(t sum_i P_i^2 / v_i)
```

with `u = H^T w`. Since the numerator now carries the propagated variance, the
ratio prices both costs rather than only the signal loss. At `width <= 0` the
numerator is `sqrt(t sum_i P_i^2/v_i)` and the ratio is exactly 1.

The obvious alternative, the weighted flux ratio
`sum_i w_i Ptilde_i / sum_i w_i P_i`, does not work. For an
absorption-dominated template it changes sign, because `Ptilde` is negative in
the lines while `P` is positive, and its denominator can pass through zero. The
SNR ratio is non-negative by construction and answers the question a filter
width is actually traded against.

Note that the statistic itself is untroubled by absorption: the signal
`S = t sum_i w_i Ptilde_i = t sum_i Ptilde_i^2/v_i` is positive whether the
lines are in emission or absorption, since a negative template correlates with
a negative data excess. Only the *summary diagnostic* needed the quadratic form.

### On whether it can exceed 1

Under the running median this diagnostic reported 1.11147 for
`P = [5, 0.1, 5, 0.1, 5, 0.1, 5]` on a flat leak of 100 at width 3, and an
earlier version of this note argued at length that the above-1 value was real
and must not be clamped.

It was not real. That number divided a *filtered* signal by an *unfiltered
diagonal* variance; the apparent gain was the covariance defect showing up in a
diagnostic. With the linear filter and the propagated variance the same case
gives 0.637088, and a scan of 3000 random templates and widths found nothing
above 0.85.

What survives from the earlier argument is the caution, not the claim: **no
upper bound is proven**. Do not clamp the value, range-validate it into
`[0, 1]`, or render it as a percentage capped at 100%. If a configuration ever
reports above 1 that is a result to look at, not a value to squash. What is now
retracted is the assertion that above-1 is expected for absorption templates.

### `N_eff`

The effective channel count is the participation ratio of the weights,

```text
n_effective_channels = (sum_i |w_i|)^2 / sum_i w_i^2,
```

which equals `n` for `n` equally weighted channels and 1 when a single channel
dominates. It answers "how many channels is this detection actually leaning
on", which is the honest count to quote next to a sqrt(N) scaling argument.

Two naming notes. The absolute value matters: the design spec writes
`(sum_i w_i)^2 / sum_i w_i^2`, which is wrong for any template with both
emission and absorption structure, where positive and negative weights cancel
in the numerator and the ratio collapses toward zero even though every channel
is contributing. `(sum |w_i|)^2` is what the code computes and what the note
specifies. And the quantity counts *channels*, not lines: at any oversampling
one spectral line occupies several channels, so the field is
`n_effective_channels`, with `n_effective_lines` kept as a deprecated alias on
`CcfBudget` and in the stage payloads.

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

`A = B2` **only when filtering is bypassed.** With `H = I` both equal
`sum_i Ptilde1_i^2 / v0_i`, since `w_i = Ptilde1_i / v0_i` on the usable
channels; even then they are computed by different summations
(`sum w Ptilde1` and `sum u^2 v0`) so they agree to rounding, not bit for bit,
and should not be asserted equal. With a filter active `u = H^T w != w` and the
identity fails outright: `B2 = sum_i u_i^2 v0_i` is the propagated photon
variance rate, a different quantity from the signal coefficient. The
photon-limited root is `n sqrt(B2)/(A t)` in general, collapsing to

```text
c_photon = n / sqrt(A t),
```

in the bypassed case. Either way it is the `1/sqrt(t)` scaling a contrast curve
should show, with `A` playing the role of an effective line-weighted rate.
(The overflow-safe substitution still uses `c0 = n / sqrt(A t)` as its scale;
it no longer coincides with the photon-limited root when a filter is active,
but it remains the right order of magnitude, which is all the substitution
needs. `q = B2/A + V_sys/(A t)` stays a ratio of like-dimensioned sums.)

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

**Photon-limited, the error is second order under a diagonal covariance.**
Expanding the true statistic in `c P1 / v0` with `V_sys = 0`,

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

**With a filter active, a small first-order term appears even photon-limited.**
The `H^T` in the variance breaks the cancellation above: perturbing the weights
perturbs `u = H^T w`, and that feeds back into `Var` in a way the frozen `C2`
does not capture. Carrying it through, with
`G(c) = sum_i Ptilde1_i^2/v_i`, `Phi(c) = sum_i (H^T(Ptilde1/v))_i^2 v_i`, and
`J(c) = sum_i Ptilde1_i Ltilde_i / v_i`, the exact statistic is

```text
SNR^2_true(c) = c^2 t^2 G(c)^2 / (t Phi(c) + K J(c)^2)
```

(the statistic is invariant under an overall rescaling of the weights, which is
why the same `c` appears in the solver's per-unit-contrast form and in
`matched_filter_ccf`'s absolute one). Expanding each of `G`, `Phi`, `J` to first
order about `c = 0` and dividing by the frozen version gives

```text
SNR^2_true / SNR^2_frozen = 1 - 2 c Delta + O(c^2),

Delta    = C2'/A - eta_ph E/B2 - eta_sys D/J0,
eta_sys  = K J0^2 / (t A + K J0^2) = V_sys / (t B2 + V_sys),
eta_ph   = 1 - eta_sys,
C2'      = sum_i Ptilde1_i^2 P1_i / v0_i^2          (from dG/dc)
E        = sum_i u0_i (H^T g)_i v0_i,  g = Ptilde1 P1 / v0^2   (from dPhi/dc)
D        = sum_i Ptilde1_i Ltilde_i P1_i / v0_i^2   (from dJ/dc)
J0       = B_w.
```

Sanity check: without a filter `u0 = w0`, so `C2' = E = C2` and `B2 = A`, and
`Delta` collapses to `eta_sys (C2/A - D/J0)` -- the earlier diagonal-covariance
result, whose photon-limited end (`eta_sys = 0`) is identically zero. With a
filter the photon-limited end is `C2'/A - E/B2`, which is *not* identically
zero. On the structured probe it is `-8.27e-05`, some 330x below the
systematics-driven end, and negative: the solver is very slightly pessimistic
there.

The observed fractional error is half the `SNR^2` coefficient:

```text
1 - SNR_true/SNR_frozen  ~=  c Delta(eta_sys).
```

Measured on an 80-channel structured probe (`P1` a modulated line forest, `L` a
modulated continuum, width-11 filter, no fringe tracking, `t = 3600 s`):

| `eps` | `eta_sys` | `c` | error | `(err/c) / Delta(eta_sys)` |
|---|---|---|---|---|
| 0 | 0 | 0.180 | `-1.50e-05` | 1.009 |
| 0.005 | 0.554 | 0.270 | `4.05e-03` | 0.996 |
| 0.010 | 0.832 | 0.440 | `9.89e-03` | 0.988 |
| 0.020 | 0.952 | 0.823 | `2.09e-02` | 0.975 |
| 0.050 | 0.992 | 2.017 | `5.13e-02` | 0.939 |

The prediction holds to ~1% while `c` stays small and degrades to 6% at
`c = 2`, where the `O(c^2)` terms bite. Note the `eta_sys = 0` row: the error is
`-1.5e-05`, first order in `c` (verified by holding the probe fixed and varying
`t`: `err/c` stays at `-8.3e-05` across a 10x range in `c` while `err/c^2` does
not), but worth about a part in `1e5` and of no practical consequence.

Two tests pin this. One asserts the systematics-dominated roundtrip lands in
`[0.90, 1.02]`, is *not* within `rel = 2e-2` of the target, and doubles when
`eps` doubles. The other computes `Delta(eta_sys)` independently from the probe
arrays and requires `error / c` to match it at `rel = 0.03` at both
`eta_sys ~ 0.17` and `eta_sys ~ 0.89`, and pins the photon-limited end at
`-8.268e-05`.

The practical statement is unchanged: the returned contrast is reliable
wherever the planet is faint against the leak and the systematic share is
small. Where it is not, it is optimistic by a few percent, and
`matched_filter_ccf` is the cross-check.

## Monte Carlo Validation

`test_monte_carlo_photon_variance_matches_the_propagated_prediction` (marked
`stats`) draws 100k Poisson realizations of the channel counts, runs each
through the real `continuum_filter`, forms `X = w . (H N)`, and compares the
empirical variance against `V_ph`. It also asserts the empirical mean matches
`t sum_i w_i (H v)_i`, that the empirical SNR matches `ccf_snr`, and -- the
regression guard -- that the old diagonal expression is more than 15% below the
measured variance on this case.

This is the test the module was missing. Every earlier test checked internal
consistency: given the model, is the arithmetic right? None of them drew a
photon, so none could see that the model itself mismatched its own estimator.
`sqrt(2/N)` is a 0.45% standard error on a variance at `N = 100k`, so the 2%
band is a comfortable three sigma and still catches the 23% error the diagonal
formula makes here by a factor of ten.

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
