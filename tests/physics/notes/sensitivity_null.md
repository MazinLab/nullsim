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

The setpoint residual and the dither are a static phase offset `b`. For
leakage quadratic in the phase, `L = (x + b)^T Q (x + b)` with random phase
`x ~ N(0, C)` (piston and fiber phase):

```text
Var L = 2 tr(QCQC) + 4 b^T QCQ b.
```

Assume `QC` has `N_eff` equal nonzero eigenvalues `q c` and that `b` lies in
their span. Then:

```text
sigma_phase^2 = 2 tr(QCQC) = 2 N_eff (q c)^2,   L_phase,dyn = tr(QC) = N_eff q c,
4 b^T QCQ b = 4 q c L_b = 2 (sigma_phase^2 / L_phase,dyn) L_b,   L_b = b^T Q b.
```

Both moments come from a phase-only MC leg: piston and fiber phase on,
scintillation off. `L_phase,dyn` is that leg's mean minus the static chip
null. Amplitude (scintillation) errors sit in the orthogonal quadrature and do
not beat with a phase bias. The stage publishes

```text
sigma_L^2 = sigma_MC^2 + 2 (sigma_phase^2 / L_phase,dyn) L_b,
L_b = setpoint + dither leakage.
```

`sigma_MC` comes from the full MC, amplitude included. When the run has no
amplitude noise, the main MC already is the phase-only leg.

The first version used `2 CV_MC^2 L_dyn L_b`, taking the CV of the full MC.
That CV is normalized by the total mean (static leakage included) and its
moments include amplitude noise. On the toys below it was 6.5x low and 4.1x
high in RMS.

`test_bias_cross_term_matches_direct_biased_monte_carlo` uses a two-input
nuller at 1.25 um with a 0.02 rad bias, applied as a deterministic input phase
in a direct MC. Bias leakage is `L_b = a1 a2 (1 - cos b) / (a1^2 + a2^2)`.

| Inputs `a` | Piston (rad) | Intensity RMS | Direct MC | Prediction | Old form |
|---|---|---|---|---|---|
| (1, 0.98) | 0.003 | 0 | 4.288e-5 | 4.298e-5 | 6.62e-6 |
| (1, 1) | 0 | 0.01 | 1.777e-5 | 1.777e-5 | 7.33e-5 |
| (1, 0.98) | 0.003 | 0.01 | 8.514e-5 | 8.528e-5 | 8.24e-5 |

The predictions are within 0.2% of the direct MC (2e5 draws). The residual is
the non-quadratic `1 - cos` leakage plus MC noise, and the test pins it at 1%.
In general the form is an approximation: it is exact only for equal
eigenvalues and misses how `b` aligns with the eigenvectors of `QC`. A
two-input nuller has one dynamic eigenvector, so it does not exercise that
error.

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

The bias cross term is phase driven, so it averages at the piston bandwidth
and is added to the piston leg's variance. On equal inputs with 0.003 rad
piston, 1% intensity RMS, a 0.02 rad bias, and bandwidths of 2000 Hz (piston)
and 11.1 Hz (amplitude), `f_eff` goes from 12.5 Hz without the term to 73.9 Hz
with it. Leaving it out overstated the averaged systematic variance 5.9x.
