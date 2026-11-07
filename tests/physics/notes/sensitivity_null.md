# Sensitivity Null Budget Notes

This note covers the derivations behind `physics/realized_null.py`,
`physics/stellar_disk.py`, `physics/planet_throughput.py`, and
`physics/systematic_bandwidth.py`, as `sensitivity.detection_curve` combines
them. It also explains how the tests that pin them are built.

## Band leakage statistics

Photon weights `w_l` are the detected stellar rate in each bin divided by the
band total, the same rates whose sum multiplies the null. One MC draw `(a, r)`
(azimuth, realization) shares its piston across wavelength. Its band leakage is

```text
N_ar = sum_l w_l L_ar,l
mean = <N>,   sigma_MC^2 = mean_a Var_r(N_ar),   CV = sigma_MC / mean.
```

Static differences between azimuths (delay-line imbalance along the track) are
a slow shift of the mean, not instability. On `kilo_maunakea` the six azimuth
means span 3.5e-5 to 7.7e-4, while the RMS within each azimuth is about 5e-5.
Pooling the azimuths gave CV 0.93 and the within-azimuth CV is 0.19. The mean
of per-bin CVs is not the CV of the detected sum. Its error sign depends on the
bin correlation (review 2026-11-07 Issue 18).

`test_band_statistics_two_bin_toy_by_hand` with weights 1:3 gives
`N = [3.25, 0.75 | 1.25, 4.25]`, so the mean is 2.375 and
`sigma_MC^2 = (1.5625 + 2.25) / 2`.

## Disk floor

The MC evaluates the star at r = 0, so its mean already holds T(0). The disk
adds the excess `<T(r) - T(0)>_disk`. It uses 16 Gauss-Legendre nodes in r on
`[0, rho_0]` with weight `2 r / rho_0^2`, and T is evaluated at those nodes,
not interpolated from the separation grid. With pointing jitter the nodes see
the smoothed T, while the subtracted T(0) stays the unsmoothed value the MC
holds.

For the imbalanced Bracewell pair `a1 = 1`, `a2 = 0.9`:

```text
T(r) = (a1^2 + a2^2 - 2 a1 a2 J0(2 pi B r / lambda)) / (2 (a1^2 + a2^2)),
```

so T(0) is not zero. The test integrates `T - T(0)` independently with
`scipy.integrate.quad`.

## Null fluctuation with a static bias

### Model

The setpoint residual and dither are modeled as an unknown static phase offset
`b_k` on each chip input mode `k`. The offsets are independent across inputs,
with `E[b_k b_l] = beta^2 delta_kl`, and the same in every wavelength bin.

The trim loop (`photon_limited_setpoint_trim`) publishes only a leakage budget
over `n_trim_modes` sensed degrees of freedom, not which actuators they are.
Taking them as i.i.d. chip-input phases assumes two things:
- the residuals of separately sensed actuators are uncorrelated;
- they spread over all inputs rather than lining up with a particular
  eigenvector of the chip.

Pol arms with per-arm deployed unitaries are separate chips, so their offsets
are independent too.

### Derivation

To linear order, the bias adds the dark field `i sum_k c_dk b_k`, where
`c_dk = U_dk a_k / sqrt(P_static)` and `a` is the input field. A fluctuation
draw `x` (piston, fiber phase, scintillation) has a static-subtracted dark
field `dE_d(x)`. That is the draw's dark field over the square root of its own
total output power, minus the static dark field over `sqrt(P_static)`.

The leakage term bilinear in `b` and `x` is

```text
sum_l w_l 2 Re sum_d conj(i sum_k c_dk b_k) dE_d(x) = 2 sum_k b_k g_k(x),
g_k(x) = sum_l w_l sum_d Im(conj(c_dk) dE_d(x)).
```

Its variance over `x` at fixed `b` is `4 b^T Cov(g) b`. Averaged over `b`:

```text
cross = 4 beta^2 S,    S = mean_a sum_k Var_r g_k     (within azimuth),
L_b   = beta^2 D,      D = mean_a sum_l w_l sum_dk |c_dk|^2,
cross = 4 L_b S / D.
```

`L_b` is the published setpoint plus dither leakage, and it fixes `beta^2`.
The cross term needs no equal-eigenvalue assumption and has no unstable
denominator. It couples phase and amplitude fluctuations as the output rows
dictate:
- In a real Bracewell row, `c` is real and an amplitude fluctuation gives a
  real `dE`, so it drops out.
- In a complex row (a three-input DFT, or a retuned spectro row fed by
  splitter copies), amplitude fluctuations do beat with a phase bias.

The static field's beat with `b` is constant in `x`, so it only shifts the
mean. Two terms are neglected. One is the trilinear `E_static x b` term, which
is suppressed by `sqrt(L_static * M)` relative to the kept one. The other is
the `x`-dependence of the bias response.

Pol arms combine as
`cross = 4 L_b sum_arm w_arm^2 S_arm / sum_arm w_arm D_arm`. In
`characterization.spectro` the rows are the retuned spectro rows, and `L_b` is
the spectro ports' share of the budget.

### Pinned toys

`test_bias_cross_term_matches_bias_averaged_direct_monte_carlo` uses a 0.02 rad
bias per input, 1.25 um, and 2e4 draws. The reference averages the direct MC
variance over every sign pattern `b in {+-beta}^M`, which reproduces
`E[b b^T] = beta^2 I` with no bias-sampling noise.

| Case | Direct MC RMS | New | Old phase-only form |
|---|---|---|---|
| (a) Bracewell, inputs (1, 0.98), 0.003 rad piston, 1% intensity | 9.539e-5 | 9.542e-5 | 9.540e-5 |
| (b) Bracewell, 0.02 rad static phase error, 0.003 rad piston | 7.352e-5 | 7.355e-5 | 4.13e-4 |
| (c) three-input DFT nulled row, 1% intensity, no piston | 4.806e-5 | 4.808e-5 | 8.37e-6 |

The residual is at most 4e-4 relative, which is the linearization error.

With 2000 random Gaussian bias draws in place of the sign patterns, the
agreement is 0.5-2%. With 200 draws it was 5-9% high, which is bias-draw
sampling noise: a quadratic form in `b` has O(1) relative spread per draw.

The old form `2 (sigma_phase^2 / L_phase,dyn) L_b` fails in two ways:
- In (b), the phase-only variance contains the beat between the random phase
  and the static error, which subtracting the static null from the mean does
  not remove. Its denominator can also go negative or near zero.
- In (c), the phase-only leg is empty.

The disk floor comes from sky points incoherent with the on-axis star and adds
no variance. `null_instability = sigma_L / effective_null`, so consumers that
multiply by the effective null get `sigma_L`.

## Planet throughput under piston

For `phi = k x`, `x ~ N(0, C)`, the Gaussian characteristic function gives

```text
E[e^{i(phi_m - phi_n)}] = exp(-k^2 (C_mm + C_nn - 2 C_mn) / 2) = G_mn,
E sum_d |U_d (X * e^{i phi})|^2 = X^H (Q * G) X,   Q = U_dark^H U_dark.
```

The Monte Carlo check passes each piston draw as a unit-modulus input transport
state, so it does not share the closed-form code path.

## Position-angle count

At separation theta the PA integrand `cos(z cos PA)` with
`z = 2 pi B theta / lambda` has Fourier content up to harmonic `~z`. Uniform
sampling converges once `n_PA > 2 z`, which gives
`n_PA = ceil(4 pi B_max theta / lambda_min)`, capped at 4096. The 85 m pair at
19.07 mas over the 1.175-1.325 um J grid (20 bin centers) has an exact
band-averaged `(1 - J0)/2` of 0.48892. Thirty-six samples give 0.3262 (-33%).

## Effective bandwidth

`1/f_eff = w_phi/f_phi + w_I/f_I`, where `w` is each process's share of the
band leakage variance. Each share comes from an MC rerun with only that
perturbation, using the same seeds. For a Bracewell pair with equal input RMS
sigma (phase in rad, fractional intensity), the leakage variances are
`2 (sigma^2/2)^2` and `2 (sigma^2/8)^2`. Piston therefore carries 16/17 of the
variance, where weighting by input variance would give it 1/2.

The bias cross term is split by leg. Each part is `4 L_b S / D`, with `S`
taken from that leg's own draws (piston-only or amplitude-only), and it is
added to that leg's variance, since it averages at that leg's bandwidth. A
real Bracewell row puts no amplitude part on the amplitude leg; a complex row
does.
