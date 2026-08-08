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

The error is second order, not first. Expanding the true photon-limited
statistic in `c P1 / v0`,

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

The practical statement: the approximation is good wherever the planet is faint
against the leak, which is the detection regime and the only regime in which a
contrast curve means anything. Near `c P1 ~ v0` the returned contrast is
slightly optimistic, and `matched_filter_ccf` is the cross-check.

### Failure convention

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
