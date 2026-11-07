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

Two numerical details in that chain, both found by the 2026-08-07 review.

*The resample uses the analytic antiderivative.* The source is piecewise
linear between samples, so its integral is piecewise *quadratic*; interpolating
the cumulative linearly is exact only when a bin edge lands on a source sample.
On a tent `[0, 1, 0]` over `x = [0, 1, 2]` the bin `[0.5, 1.5]` holds 0.75 of
flux and linear interpolation returns 0.5, a 33% deficit. Within cell
`[x_k, x_k+1]` of width `h`, with `s = (p - x_k)/h`, the exact partial integral
is `h (y_k s + (y_k+1 - y_k) s^2 / 2)`, which is what the code now evaluates.
At the oversampling the packaged templates use the old error was under 1%, but
it scaled with how far bin edges fell from source samples, which is not a thing
a caller controls.

*The LSF pads with zeros, not by edge replication.* Scipy's `nearest` mode
fills the pad with copies of the edge *sample*, so a line sitting on the first
channel is smeared out of a pad full of copies of itself and returns 2.23x its
flux. The convolution now pads explicitly with zeros -- outside-band model:
no flux beyond the supplied grid -- convolves, and crops. An edge delta then
keeps `0.5 + g(0)/2` of its flux, the right half of the kernel plus half its
own centre sample, and nothing is ever created. Reflection would conserve
in-band flux but only by fabricating a mirror image of the spectrum, a stronger
claim about the unobserved band than "nothing is there". The cost is that the
caller must leave margin: values within `~8 sigma` of either end are
attenuated, so channel edges must sit at least that far inside the grid.

That model is right for a *rate density* and wrong for a dimensionless
*shape*. A transmission or albedo does not vanish past the supplied grid, so
zero padding there reads as absorption: a flat T = 0.9 came back near 0.77 in
the edge channels at R = 100 and fell under the 0.8 mask (review 2026-11-07
row 50). `outside="edge"` continues the curve at its edge value, which leaves
any constant exactly unchanged; the telluric mask and every template shape use
it, and the rate densities keep zero padding.

*Mixed content resolution.* The templates are each degraded to the lowest
content resolving power among them (and the analysis grid) before they are
multiplied, and only the product is broadened further, from that R to the
instrument R, when the instrument R is lower. Broadening each template
separately to a different R and multiplying left the sharper templates
unbroadened whenever the instrument R reached the coarsest one (row 51).

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
H = I - S,      S = diag(1/c) M,     c = M 1,
```

with `M` the zero-padded boxcar of odd width `m >= 3`:
`M_ij = 1/m` for `|i - j| <= m//2`, truncated to the array. (`M`, not `K`: `K`
is the systematic-amplitude scalar in the covariance below, and reusing the
letter for both was confusing.) Two properties do all the work.

*`M` is symmetric.* Restricting a symmetric Toeplitz matrix to a principal
submatrix leaves it symmetric, so `M = M^T` exactly, and therefore

```text
S   x = M x / c,
S^T y = M (y / c),
```

one smoother application each. `H^T = I - S^T` follows. No dense `n x n` matrix
is ever built; `continuum_filter` and `continuum_filter_adjoint` are the two
calls, and a test pins the adjoint against the dense transpose at `1e-14`.

*`S` preserves constants exactly.* `S 1 = diag(1/c) M 1 = diag(1/c) c = 1`, at
the array edges as well as the interior, because `c` is precisely the in-array
fraction of each window. So `H 1 = 0` everywhere and a flat stellar continuum
is annihilated with no edge residual. (A ramp is killed only in the interior;
the edge residual is second order in the slope.)

### Blockwise over islands

`H` is not built on the full channel grid. It is built on each **island** -- a
maximal run of consecutive *admitted* raw channels -- and is therefore block
diagonal, with `H 1 = 0` holding per island rather than globally.

That is forced by wavelength adjacency. Two admitted segments either side of a
telluric band are not neighbours, and a filter that smooths across the gap
invents continuum that is not there. Compacting the admitted channels into one
vector, as an earlier round did, took two flat segments `[10,10,10]` and
`[100,100,100]` separated by six masked channels and had the width-3 boxcar
infer a continuum of `[10,10,40,70,100,100]`, manufacturing residuals of `+-30`
and a detection SNR of 2.22 in a spectrum containing no line at all. Filtered
per island each flat segment annihilates exactly, and both statistics report
zero. This is reachable in practice: the Maunakea stage fixture masks 88 of 550
channels, including an interior 11-channel gap.

An island narrower than the filter width just gets its own renormalized boxcar,
and a one-channel island gets `H = 0` -- correct, since one sample cannot
separate a line from a continuum.

Consequences worth holding onto: `null(H)` is spanned by the per-island
indicators, not by the single global constant, so the RV optimum subtracts a
per-island weighted mean; and masking changes the island structure, which is
part of why masking is not monotonic in SNR.

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
test pins, and it grows as the filter narrows.

All of this lives on the admitted sub-vector. `w`, `u` and the sums run over
admitted channels only, and `u` is zero off that set by construction, because
`H` is built blockwise on the islands and never reaches a masked channel. An
earlier version of this note said the opposite -- that `u` is nonzero on masked
channels because the filter pulls their photons into their neighbours. That was
a true description of the code at the time and a statement that the masking
semantics were wrong: a telluric-masked channel carries no usable data, so the
statistic must not read it. See the masking section.

Only when filtering is bypassed does `u = w`, and then `V_ph = S` identically:

```text
SNR_photon = S / sqrt(V_ph) = sqrt(t * sum_i Ptilde_i^2 / v_i).
```

### This is not the optimal filter

Under the full covariance of the *filtered* data,

```text
C_y = t H diag(v) H^T + K Ltilde Ltilde^T,
```

photon noise propagated through `H` plus the rank-one systematic, the
maximum-SNR weights are `C_y^+ Ptilde`, whose rank-one part Sherman-Morrison
inverts in closed form. That filter *downweights* the leak direction -- by
`1/(1 + K Ltilde^T C_ph^+ Ltilde)`, so "projects it out" is exact only as
`K -> inf`; at finite systematic amplitude the suppression is partial -- and
beats the fixed-template statistic by roughly an order of magnitude once
systematics dominate. `C_y` is singular because `H 1 = 0`, so the inverse is a
pseudoinverse on `range(H)`, where both `Ptilde` and the data live.

The filtered photon term is not a detail. Writing that term as the unfiltered
`diag(v t)` -- as an earlier version of this note did -- gives
`sqrt(t A)` with `A = sum_i Ptilde_i^2/v_i`, against an implemented statistic of
`sqrt(t) A / sqrt(B2)`. Their ratio is `sqrt(A/B2)`, so the "bound" drops
*below* the thing it is supposed to bound whenever `A > B2`. That is reachable:
on a 17-channel width-5 probe with `A/B2 = 1.075`, the unfiltered form gives
6.390e-03 against an implemented 6.624e-03, while the correct filtered optimum
is 8.933e-03. A bound stated in the wrong space is not a bound.

It is documented here as an **upper bound, not implemented**, because it assumes
the leak spectrum `Ltilde` is known exactly. A filter that subtracts a leak it
knows perfectly will always look good on paper; what an instrument can actually
deliver is a leak estimate with its own errors, and the gain collapses with the
knowledge. Adopting it is a design decision about which assumption to make, not
a bug fix, so the module keeps the honest fixed-template statistic and labels
it as such. Everything published -- module docstring, stage model strings --
says "photon-weighted CCF" for the *detection* statistic. The RV scan is a
different object and keeps the plain term: its weights really are optimal for
`C_y`, so "whitened matched filter" is accurate there, and the asymmetry in
naming is the point rather than an oversight.

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

**Semantics: raw-channel exclusion.** A channel whose atmospheric transmission
falls below the threshold carries no usable data, so no statistic may depend on
its photons. The analysis compacts to the admitted channels and builds the
continuum filter on *that* vector; the statistic is `a . N` with `a` supported
on the admitted set, full stop. Channels with `v_i <= 0` are dropped on the
same footing -- they carry no information, and admitting them would make
`H diag(v) H^T` rank-deficient relative to `H`, breaking the
`range(C_y) = range(H)` identity the RV optimum needs.

The alternative reading -- filter the full vector, then zero the
filtered-output weights -- looks equivalent and is not, because `H^T` spreads
support: `a = H^T w` is generally nonzero where `w` is zero, so masked photons
leak back in. `matched_filter_ccf` did exactly that until the 2026-08-09 review,
which is why its SNR disagreed with the RV scan under a mask (0.1936 against
0.0 on a five-channel probe where the two admitted channels have equal planet
rate and a zero-sum filter on them can see nothing).

**Masking is not monotonic in SNR.** With a diagonal covariance it would be:
`SNR^2 = t sum_i m_i Ptilde_i^2 / v_i` is a sum of non-negative terms, so
dropping a channel could only lose. That argument dies with the filter. The
real quantity is

```text
SNR^2 = t (sum_i w_i Ptilde_i)^2 / sum_i u_i^2 v_i,    u = H^T w,
```

over the admitted sub-vector, and dropping a channel changes `H` itself -- the
smoother's windows, its edge normalization, and which neighbours each surviving
channel is filtered against. A noisy channel adjacent to a line can cost more
in propagated variance than it returns in signal, so removing it *raises* the
SNR. Measured on a six-channel probe with a width-3 filter: masking one channel
took the SNR from 1.871 to 4.266. Nested masks therefore do not produce
monotonically falling SNRs, and nothing should assert that they do.

Masking is a mask only: the transmission is assumed to be already folded into
the rates.

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

### Per island, because the global number is a different quantity

`H` is block diagonal over islands, so the signal and the propagated variance
are both island sums and each island's retention is a well-defined number on its
own. `island_channel_counts` and `island_self_subtraction_factors` publish them.
Measured on a 40-channel probe (a 30-channel comb against a flat 100 Hz leak, a
3-channel island, a 1-channel island, telluric gaps between them, width 15):

| island | channels | retention |
|---|---|---|
| comb | 30 | 0.890 |
| narrow | 3 | 0.632 |
| singleton | 1 | 0.000 |
| **global** | 34 | **0.801** |

A one-channel island retains nothing, ever: its boxcar window holds one sample,
so the estimated continuum *is* the datum and `H = 0`. Beyond that the spread is
not monotone in island length, and it would be wrong to read it as "short is
bad". What sets retention is how much of the template's power sits above the
filter's passband on that island, and island length changes the passband: on an
island shorter than the width the renormalized boxcar spans the whole block, so
the filter removes only the island mean and a line survives nearly intact, while
on a long island the running mean tracks local structure and takes more of the
line with it. Both directions are real. In the probe above the 30-channel comb
(all its power at the line scale, perfectly matched to a width-15 high-pass)
beats the 3-channel island, whose mean absorbs a third of a `[1, 5, 1]` line —
and the same 3 channels retain 0.823 for `[0, 5, 0]` and 0.379 for a ramp
`[1, 2, 3]` the filter reads as continuum. In the shipped
`hdc_ccf_systematics.toml` the ordering is the other way round: islands of
382 / 196 / 17 channels retain 0.066 / 0.167 / 0.556 against a global 0.0784,
because that planet template is smooth on the scale of 15 channels and only the
short island is spared.

The point of publishing per island is not a rule of thumb; it is that a single
number covering islands with a factor-of-8 spread in retention is not a
description of anything, and a config whose signal lives on the island that
retains 0.066 is in a different situation from one whose signal lives on the
island that retains 0.556.

**And the global factor is not an average of the island factors.** Writing
`S_k`, `V_k`, `U_k` for the island sums of filtered signal, propagated variance
and unfiltered `sum P^2/v`, the island factor is `S_k / sqrt(V_k U_k)` while the
global one is

```text
self_subtraction_factor = sum_k S_k / sqrt( (sum_k V_k) (sum_k U_k) ),
```

one ratio of three sums. Cauchy-Schwarz gives
`sqrt(sum V sum U) >= sum sqrt(V_k U_k)`, so the global value is at most the
`sqrt(V_k U_k)`-weighted mean of the island factors, with equality only when
`V_k / U_k` is constant across islands. When the islands are unbalanced -- one
carrying the noise, another carrying the unfiltered signal power -- the global
value falls **below every island factor**: a two-island probe with factors 0.594
and 0.480 reports a global 0.344. Earlier text here (and in the dataclass
docstring) called the global number an average of the island ones; it is not,
and the direction of the error is not bounded by the island range.

### A filtered template that is only roundoff

`H` annihilates a constant analytically and to a few ulp in floating point: the
worst case measured over flat arrays (lengths 1 to 4096, widths 3 to 1001) is
`max|H x| = 89 eps max|x|`. Since every quantity downstream is scale-free in the
template — the weights are `Ptilde/v`, and the SNR is a ratio — that residue does
not stay small. Three flat channels at 1.03 Hz against a 100 Hz leak reported
`ccf_snr = 13.099` at `t = 3600 s`, and 100.1 Hz against 1000 Hz reported 6.222
at `t = 1 s`: ordinary rates, ordinary exposures, a detection manufactured from
rounding.

Both entry points therefore zero any island whose filtered template is below
`1024 eps` times that island's own raw template magnitude. The threshold is
~11x the measured worst-case residue and ~3 orders below the filtered amplitude
of a `1e-10` fractional line (`4e5 eps`), so nothing a spectrum could physically
carry is at risk. Per island and relative, because a global absolute cut is
wrong in both directions: it misses roundoff on a bright island and erases real
structure on a faint one.

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

Four cases are pinned: the alternating case at 0.637088 (it read 1.11147 under
the running median and the mismatched variance -- see above), sparse absorption
dips on a continuum, the bypass path with an all-zero template, and the
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

`ccf_vs_rv` correlates a fixed observed spectrum against templates rebuilt at
trial velocities. The observation it correlates against is `planet + leak` on
the admitted channels -- the function subtracts nothing, and a caller that wants
a leak-subtracted residual (as `characterization.hdc_ccf` does, folding the leak
mean into the variance budget on the strength of bright-port monitoring) passes
it in that way. Its job is to *say something about* the planet's velocity, which
takes a different filter from the detection statistic; what it is not is a
statistic whose maximum is provably at the truth, and the difference is worth
understanding.

The data is `y = H N` with covariance `C_y = t H diag(v) H^T`. The generalized
matched filter under that covariance,

```text
CCF(rv) = <Ttilde(rv), y>_{C_y^+} / sqrt(<Ttilde(rv), Ttilde(rv)>_{C_y^+}),
```

is a cosine under a single inner product, so Cauchy-Schwarz bounds it by
`sqrt(<y, y>_{C_y^+})`, with equality when `Ttilde(rv)` is parallel to `y`.

**The peak is not therefore at the true radial velocity.** Equality is with
`y`, the *observed* spectrum -- planet plus leak -- not with the planet
template, so the scan finds the template best matching the observation. That is
the true velocity only when the filtered leak projection is negligible, which
needs the leak smooth enough for `H` to annihilate it. Structured leak breaks
it outright: on a twelve-channel probe the true velocity scores `-5.99` against
`9.80` for a four-channel shift. Earlier rounds of this note asserted the
guarantee unconditionally; it is conditional on a smooth leak, and that
condition is a property of the instrument, not of the algebra.

### The pseudoinverse is closed form here

`C_y` is singular, but only just. `H` is **block diagonal over the admitted
islands**, and each block is `I - S_k` with `S_k` row-stochastic (rows sum to 1
by the window-fraction normalization) and irreducible (the band connects every
channel of the block), so Perron-Frobenius makes each block's unit eigenvalue
simple. Hence

```text
null(H) = span{ 1_k : k an island },
```

the span of the island *indicator* vectors -- not `span{1}`. Earlier versions of
this note asserted `null(H) = span{1}` and one global Lagrange multiplier, which
was right only while the filter ran on a single compacted vector; the
blockwise filter has one constraint per island, and the implementation has
always done that. Maximizing `(a . P)^2 / (t a^T diag(v) a)` over
`range(H^T) = null(H)-perp` is then one Lagrange problem per island:

```text
a_i = m_i (T_i - lambda_k) / v_i     for channel i in island k,
lambda_k = sum_{i in k} m_i T_i/v_i / sum_{i in k} m_i/v_i,
CCF(rv) = sqrt(t) * (a . R_obs) / sqrt(a^T diag(v) a),
```

the raw template inverse-variance weighted with a variance-weighted mean removed
island by island. `sum_{i in k} a_i = 0` on each island by construction of
`lambda_k`. No eigendecomposition, no dense matrix, `O(n)` per trial. A test
pins it against the literal `np.linalg.pinv` form at `rtol 1e-8`.

### Why the implementation uses pair differences

Written as `T_i - lambda_k` the formula has two numerical failures, both of them
reachable and both fixed by the algebraically identical

```text
a_i = u_i (T_i - T_ref) - (u_i / s_k) sum_{j in k} u_j (T_j - T_ref),
u = 1/v,   s_k = sum_{j in k} u_j,   ref = the largest-u channel of island k.
```

*Flat islands.* Subtracting `lambda_k` from a template constant on the island
leaves one ulp, not zero, and every downstream quantity is scale-free in `a`, so
normalizing by that residue amplifies it into an O(1) score. Round 3 guarded it
with an absolute `1e-12` on the globally normalized template residual, which is
wrong in the other direction: a `1e12 Hz` island beside a `1 Hz` one pushes the
faint island's *genuine* residual to `8e-13`, and the guard zeroed the whole
trial (the correct score there is `sqrt(0.4) = 0.63246`, the faint island's own
statistic). Pair differences need no guard at all: `T_i - T_ref` is exactly zero
on a flat island whatever its magnitude, and exactly the real structure on a
faint one.

*Variance spans wider than float64.* `1/v` overflows on its own for a subnormal
`v`, and `a` is built from products of a huge `u` with a tiny one, so no single
normalization of `u` -- linear or logarithmic -- holds both ends. Each factor is
therefore carried as a `(mantissa, exponent2)` pair through `math.frexp`, with
the powers of two added as integers and materialized once against a global
reference. For `v = [5e-324, 1e308]` the optimum is antisymmetric, `a = [-1, 1]`
up to scale, and the score is the rate difference over the noisy channel's
sigma, `1e154`; the linear form returned NaN.

Base-two exponents rather than logarithms because the split is exact.
`exp(sum(log x))` carries an absolute error in the log that becomes a relative
error of `|log x| * eps` in the result -- `1.6e-13` at these magnitudes, ten
times the tolerance the extreme-variance test pins at.

### What was wrong before

The scan used to take the fixed-template weights `w = diag(v)^-1 Ttilde` and
divide by the covariance norm `sqrt(w^T C_y w)`. Numerator and denominator
belong to different inner products, so the ratio is not a cosine and the
Cauchy-Schwarz argument does not apply. It is not a small effect: on a
ten-channel probe with a flat leak of 100 and

```text
P = [0.532967, 1.263326, 1.007460, 1.088272, 3.241559,
     2.635570, 1.414478, 9.568137, 0.156691, 66.788444],
```

a template slid eight channels with zeros filling in behind scored 3.591
against 3.383 for the truth -- the wrong template winning by 6.2%. Both halves
are individually defensible: before the covariance fix the denominator was
`sqrt(w^T diag(v) w)`, which *is* the right norm for those weights and made a
valid cosine under `diag(v)^-1`. What broke it was swapping in the covariance
denominator without also moving the weights.

### What whitening costs

Whitening by `C_y` necessarily undoes whatever `H` suppressed. The filter
attenuated signal and noise together, and dividing by the true noise restores
the ratio; the optimum is blind only to the per-island constant modes, the only
directions `H` genuinely destroys. So this scan, unlike `matched_filter_ccf`, is
*not* protected against non-constant leak structure.

That is a deliberate split, not an inconsistency. The detection statistic keeps
its fixed-template weights and with them the boxcar's leak suppression, at the
cost of not being optimal; the localization statistic takes the optimum for the
covariance the data actually has, at the cost of the suppression. What it does
not buy is a guarantee that the peak sits at the true velocity -- see above, and
`-5.99` against `9.80`. What the scan needs beside it is a statement about
whether *this* observation's peak can be trusted, which is the next section.

### Grading the argmax: a margin, over the whole grid

`ccf_rv_scan` takes an optional `residual_leak_rate_hz_per_channel` -- the part
of the leak a pipeline's own subtraction cannot remove -- and returns, beside
the scores, that residual's contribution to *every* trial's score,

```text
shift(rv) = sqrt(t) (a(rv) . L_res) / sqrt(a(rv)^T diag(v) a(rv)),
```

in the same sigma units. `RvScan.localization(truth)` then compares two
quantities of the same kind:

```text
margin            = score(truth) - max_{rv != truth} score(rv),
worst_case_margin = min over sign in {+1, -1} of
                      [ (s_truth + sign d_truth) - max_j (s_j + sign d_j) ],
```

and grades `reliable` when `worst_case_margin > 0`, `marginal` when the
unperturbed scan peaks at the truth but the residual could overturn it,
`unreliable` when `margin <= 0`. The truth must be *on* the grid: a scan whose
trials straddle the true velocity has not evaluated it, and grading the nearest
trial would be a verdict about a different velocity, so that returns
`not evaluated` -- as does a scan built with no residual model at all, whose
all-zero shifts would otherwise grade `reliable` on a perturbation nobody
supplied. `characterization.hdc_ccf` inserts the scene velocity into its scan
grid so an even `rv_scan_points` cannot straddle it. No dimensionless cut
appears anywhere: the
criterion is one sigma quantity against another, so there is nothing to
calibrate.

**Why it has to be grid-wide.** A single projection at the true velocity is not
a statement about an argmax. The regression builds a positive leak by
projecting the shift-`+4` filter off the truth's filter, which *targets*
orthogonality at the truth but does not achieve it exactly: changing the sky to
carry that leak changes the variance, and with it both filters, so the
construction is iterated and lands on a leak that is merely much better aligned
four channels away than at the truth. That is enough. The residual shifts the
truth's score by `0.061` sigma and the `+4` trial's by `1.343`, against a margin
of `0.432`, and the argmax moves. The truth-only ratio there is `31.4` raw and
`0.031` after the `eps` scaling -- comfortably inside the cut the round-4
verdict used, which duly graded the case `reliable`.

The coherent worst-case margin is `-0.161` sigma, not the `-0.972` a
term-by-term bound would report: the residual is one signed vector, so the two
realizations to check are `+d` and `-d`, and within each every trial moves
together. Combining `|d_truth|` with `max_j |d_j|` assumes a sign pattern that
cannot occur and rejects scans nothing can overturn -- scores `[10, 7]` with
shifts `[5, 3]` has coherent margins of 5 and 1, both safe, where the bound
reports `-5`. The bound was conservative, never optimistic, so correcting it
removes false rejections only.

**What it does not cover.** Only the residual actually passed in. In
`characterization.hdc_ccf` that is `eps_systematic * L`, static, with no
averaging credit -- the same amplitude the stage charges as `V_sys`. An
independent static subtraction error, a template mismatch, or a detector
systematic moves the peak with `eps_systematic = 0` and grades clean. The
payload keys are named `rv_scan_localization_valid_modeled_leak` and
`..._grade_modeled_leak` so that scope is impossible to miss, and the
`hdc_ccf` plot annotates the RV panel with the margin and the shift whenever
the grade is not `reliable`. `rv_leak_contamination`, the truth-trial ratio
`|a . L| / |a . P|`, is still published as a descriptive scalar and is
deliberately not the verdict input.

The two statistics are **not comparable in either direction**, and an earlier
version of this note claimed both that the RV score sits above
`ccf_snr_photon_limited` and that Cauchy-Schwarz put it there. Neither holds.
The optimum dominating any fixed filter in its subspace would give `>=` only if
both correlated against the same vector, and they do not: the scan correlates
against the observed planet-plus-leak spectrum, while the detection signal is
the planet alone. A leak that cancels the planet's structure inverts the
ordering -- `P = [1, 2, 1]` under `L = [9, 8, 9]` is flat in total, so the scan
reports `1e-15` for a featureless observation while detection reports `0.258` on
the strength of knowing the leak. Do not read the RV score as a detection SNR,
and do not assume either bounds the other.

Two further requirements are enforced rather than assumed. The fixed data
spectrum is a required input: a scan that shifts the template and rebuilds the
weights and the "data" together would be an autocorrelation and would be flat in
`rv`. And the trial closure must return channels on the data's wavelength grid;
`_require_same_grid` compares edges and centers to `rtol = 1e-9` and raises,
because a same-length grid at different wavelengths would otherwise be
correlated index by index and quietly return a wrong answer.

## Contrast At A Target SNR (Detection Mode)

`ccf_contrast_at_target_snr` inverts the photon-weighted CCF: rather than scoring a
known planet, it asks how faint a planet the same observation would still detect
at SNR `n`. That inversion is what a CCF contrast curve is made of, so it runs
once per separation anchor and has to be cheap.

Let `P1_i` be the per-channel planet rate at *unit* band-integrated contrast, so
the actual planet rate is `c P1_i`. The planet-free variance rate is
`v0_i = L_i + sky_i + dark_i`. With weights frozen at zero contrast,

```text
w_i  = m_i Ptilde1_i / v0_i,
u0_i = (H^T w)_i                (the propagated noise weights)
A    = sum_i w_i Ptilde1_i     [Hz per unit contrast]
B2   = sum_i u0_i^2 v0_i       [Hz]
C2   = sum_i u0_i^2 P1_i       [Hz per unit contrast]
B_w  = sum_i w_i Ltilde_i      [Hz]
```

`B2` and `C2` carry `u0`, not `w`: they are the two halves of the H-propagated
photon variance rate, split by their dependence on `c`. Only `A` and `B_w`,
which are signal projections rather than variances, keep the plain `w`.

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
ranges. `A^2 t^2` overflows for `A ~ 1e150`, raises `OverflowError` at
`t = 1e160`, and flushes to zero for rates around `1e-200` — all representable
float64 inputs whose exact roots are perfectly ordinary numbers. Worse, the
coefficients `A`, `B2` and `C2` over/underflow while they are being
*accumulated*, before any root formula sees them.

Four devices, each answering a specific failure:

- **Weights with the exponent kept beside the mantissa.** `w = ptilde/v0` is
  formed by dividing the `math.frexp` mantissas and subtracting the exponents,
  then materializing against the largest exponent, so `|w| <= 1` no matter how
  far `v0` spans. Linear scaling cannot do this: `5e-324` against `1e308` is a
  span of `1e631`, and a geometric mean only halves the *exponent* span,
  leaving `1e315` — still outside float64. Negligible channels underflow to
  zero, which is their correct weight. Base two rather than natural logs
  because the split is exact: accumulating in logs costs `|log x| * eps`,
  which is `1e-13` at these magnitudes.
- **Scales carried analytically.** With `alpha = max|P1|`,
  `beta = max v0` and `kappa = alpha 2^G`, `G` the integer exponent above,
  `A = kappa alpha A''`, `B2 = kappa^2 beta B2''`, `C2 = kappa^2 alpha C2''`,
  `B_w = kappa beta B_w''`. `kappa` cancels out of the root entirely.
- **The gain stays factored.** Both `P` and `sqrt(Q)` carry
  `g = n/(alpha A'')`, and `hypot` is homogeneous, so `g` factors out of the
  whole root. It is never formed on its own: it overflows for
  `alpha ~ 1e-308` and underflows for a subnormal target, on roots that are
  representable either way. No fixed multiplication order rescues both ends:
  multiplying `n` into the bracket first saves a subnormal target from
  `n/A''`, but `n = 1e-163` against a `3.2e-172` bracket underflows that
  product to zero on a root of `3.2e-57`. The closing
  `n * bracket / (A'' alpha)` is therefore assembled with the same exponent
  bookkeeping as the weights and materialized once.
- **The systematic enters as `sqrt(V_sys)/t`, never as `V_sys`.** The `t`
  cancels analytically -- `eps |B_w|` without fringe tracking,
  `eps |B_w| / sqrt(2 f t)` with -- so squaring `eps B_w t` only to take its
  square root again was a pure overflow hazard. It rejected `eps = 1`,
  `B_w ~ 1`, `t = 1e200` on a root that is *t-independent*.
- **The reduced coefficients themselves are `(mantissa, exponent)` pairs.**
  `P` and `sqrt(Q)` are formed, added and hypot-ed without being materialized,
  because either can leave float64 while the root stays ordinary: two bypassed
  channels at `P1 = L = 1e308` with `eps = 1` put the systematic half of
  `sqrt(Q)` at `beta eps |B_w''| = 2e308`, and the answer is
  `c = n eps B_w / A = 5` exactly (4.999999999999999 at 8e307, where the sums
  still fit). `A` and `V_sys` are reported as `inf` there, correctly -- they
  genuinely do not fit -- and that no longer blocks the root.
- **`V_sys` is squared in physical units, not normalized ones.** Squaring
  `eps B_w'' t` first and applying `(kappa beta)^2` afterwards overflows on a
  budget whose answer is tiny: `P1 = 1e-300` against a unit leak at `t = 1e200`
  has `(eps B_w'' t)^2 = 1e400` normalized, while the physical
  `V_sys = (1e-300 * 1e200)^2 = 1e-200`. The amplitude is assembled in
  exponent form and squared there.

Plus two shape rules on the root itself: it is
`0.5 P + hypot(P/2, sqrt(Q))`, algebraically the same as
`0.5 (P + sqrt(P^2 + 4Q))` but never forming `2 sqrt(Q)`, which overflows at
`sqrt(Q) ~ 9.2e307` on a finite root; and `n` never appears squared inside a
single product, since a `1e-200` target would flush `n^2` to zero and return a
contrast of 0.

Reported diagnostics (`A`, `B2`, `C2`, `V_sys`) are a separate matter, and none
of them feeds the root — a diagnostic that saturates must never abort a root
that is representable. Each is assembled by `_scaled_product`, which folds every
factor's power of two into an integer and materializes once, so the weight scale
`kappa` is never itself formed. It has to be that way: `P1 = 1e-300` against a
`5e-324` leak puts `kappa` at `2^1074`, and multiplying it out turned the weight,
`A`, `B2` and `C2` into `inf` and `V_sys` into `NaN` on a problem whose reported
values are `2.024e23`, `2.024e-277`, `2.024e-277` and `4.097e-254`. Where a
value genuinely is not representable the closing `ldexp` saturates to `inf` (or
underflows to zero, as `A = 1e-497` does in the subnormal-target case) rather
than raising.

Pinned values, each independently verified: `25` exactly for the `1e-200`-leak
case, `8.373669637337405e96` and `2.754554230312972e-99` for the two extreme
magnitudes, `5e160` across `t` from `1e140` to `1e200` in the
systematics-dominated regime, `9.225286e307` for the doubling probe, and a
subnormal root near `1e-323` for a subnormal target.

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
plain multiplications saturate silently to `inf`. **This no longer raises**, and
an earlier version of this note said it did. Since the root takes the systematic
as `sqrt(V_sys)/t` computed per regime, an unrepresentable `V_sys` costs the
root nothing: `P1 = 1e10, L = 1e170, eps = 1` at `t = 1e200` returns
`c = 5e160` and reports `systematic_var_counts = inf`, which is the honest
label for a diagnostic that does not fit in a float64. The `OverflowError` from
`**` is caught and mapped to `inf` so the two regimes report the same way.
`matched_filter_ccf` is deliberately left alone — changing what it raises would
be a behavior change to a function with its own consumers.

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
eta_sys  = K J0^2 / (t B2 + K J0^2) = V_sys / (t B2 + V_sys),
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
spectrum, one whose lines all land in telluric-masked channels, or one that is
constant on every island, where `H P1` is rounding residue and the same
per-island roundoff guard the detection statistic uses zeroes it. There is no
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
