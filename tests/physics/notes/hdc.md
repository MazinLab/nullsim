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
retained fraction, weighted the same way the statistic is,

```text
self_subtraction_factor = sum_i w_i Ptilde_i / sum_i w_i P_i   (<= 1),
```

is 1 exactly when filtering is disabled (`width <= 0`), which the flat-template
test checks, and falls toward 0 as the filter width shrinks toward the line
width.

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
velocity plus leak, sky, dark) against templates rebuilt at trial velocities:

```text
CCF(rv) = t * sum_i m_i * Ttilde_i(rv) / v_i * Dtilde_i,
```

with `v_i` and `Dtilde` taken from the observation and held fixed across the
scan. It is the signal term of the matched filter with the template shifted, so
it peaks where the template lines land on the data lines. The pinned test uses
integer channel shifts of a line comb on a smooth leak and requires the argmax
at the true shift. Note that the fixed data spectrum is a required input: a scan
that shifts the template and recomputes the weights and the "data" together
would be an autocorrelation and would be flat in `rv`.

## Citations

- Wang, J., Mawet, D., Ruane, G., Hu, R., and Benneke, B. 2017, AJ 153, 183.
- Snellen, I. A. G. et al. 2015, A&A 576, A59.
- Mawet, D. et al. 2017, ApJ 838, 92.
- Sparks, W. B. and Ford, H. C. 2002, ApJ 578, 543.
