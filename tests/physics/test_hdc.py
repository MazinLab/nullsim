"""Tests for nullsim.physics.hdc (matched-filter CCF budget)."""
import numpy as np
import pytest

from nullsim.physics.hdc import (
    ChannelSpectra,
    build_channel_edges,
    ccf_contrast_at_target_snr,
    ccf_vs_rv,
    channelize,
    continuum_filter,
    matched_filter_ccf,
    snr_vs_resolving_power,
)
from nullsim.physics.spectra import constant_r_grid


def _channels(planet, leak, sky=None, dark=None):
    n = planet.size
    edges = build_channel_edges(1.0, 1.4, float(n) / np.log(1.4))
    edges = edges[: n + 1]
    centers = 0.5 * (edges[:-1] + edges[1:])
    zeros = np.zeros(n)
    return ChannelSpectra(
        channel_centers_um=centers,
        channel_edges_um=edges,
        planet_rate_hz=planet,
        leak_rate_hz=leak,
        sky_rate_hz=zeros if sky is None else sky,
        dark_rate_hz=zeros if dark is None else dark,
    )


def test_channelize_conserves_band_rate():
    grid = constant_r_grid(1.0, 1.4, 100_000.0)
    density = np.full(grid.size, 3.0)  # hz per um
    edges = build_channel_edges(1.05, 1.35, 1000.0)
    rates = channelize(grid, density, edges, resolving_power_inst=1000.0)
    total = float(rates.sum())
    assert total == pytest.approx(3.0 * (edges[-1] - edges[0]), rel=1e-3)


def test_matched_filter_flat_template_recovers_broadband_snr():
    # With continuum filtering disabled (huge leak lines absent, filter width
    # spanning the whole band is bypassed via width=0 semantics), a flat
    # planet spectrum over n channels must reproduce S/sqrt(S+B).
    n = 64
    planet = np.full(n, 10.0 / n)
    leak = np.full(n, 1000.0 / n)
    ch = _channels(planet, leak)
    budget = matched_filter_ccf(
        ch,
        integration_time_s=3600.0,
        eps_systematic=0.0,
        servo_bw_hz=100.0,
        with_fringe_tracking=True,
        continuum_filter_width_channels=0,  # 0 disables filtering
        telluric_transmission_per_channel=None,
        telluric_mask_threshold=0.0,
    )
    s, b, t = 10.0, 1000.0, 3600.0
    expected = s * t / np.sqrt((s + b) * t)
    assert budget.ccf_snr == pytest.approx(expected, rel=1e-10)
    assert budget.self_subtraction_factor == pytest.approx(1.0)


def test_orthogonal_leak_lines_give_zero_systematic_projection():
    n = 200
    planet = np.zeros(n)
    planet[10:200:20] = 5.0  # planet lines on even-decade channels
    leak = np.full(n, 50.0)
    leak[3:200:20] += 40.0  # leak lines elsewhere
    ch = _channels(planet, leak)
    budget = matched_filter_ccf(
        ch,
        integration_time_s=1000.0,
        eps_systematic=1e-2,
        servo_bw_hz=100.0,
        with_fringe_tracking=True,
        continuum_filter_width_channels=15,
        telluric_transmission_per_channel=None,
        telluric_mask_threshold=0.0,
    )
    # Filtered leak lines do not overlap template lines: projection ~ 0
    # relative to the unfiltered band leak.
    assert abs(budget.systematic_ccf_leakage) < 1e-6 * leak.sum()


def test_identical_lines_at_zero_shift_give_full_correlation():
    n = 200
    lines = np.zeros(n)
    lines[10:200:20] = 5.0
    ch = _channels(lines, 100.0 * lines + 50.0)
    budget = matched_filter_ccf(
        ch,
        integration_time_s=1000.0,
        eps_systematic=1e-2,
        servo_bw_hz=100.0,
        with_fringe_tracking=True,
        continuum_filter_width_channels=15,
        telluric_transmission_per_channel=None,
        telluric_mask_threshold=0.0,
    )
    assert budget.systematic_ccf_leakage > 0.0
    assert budget.systematic_var_counts > 0.0


def test_snr_scales_as_sqrt_of_line_count():
    # Wang et al. 2017 scaling: N identical lines against flat leak noise.
    def snr_with_lines(n_lines):
        n = 4000
        planet = np.zeros(n)
        step = n // n_lines
        planet[::step][:n_lines] = 1.0
        leak = np.full(n, 100.0)
        ch = _channels(planet, leak)
        return matched_filter_ccf(
            ch,
            integration_time_s=100.0,
            eps_systematic=0.0,
            servo_bw_hz=100.0,
            with_fringe_tracking=True,
            continuum_filter_width_channels=0,
            telluric_transmission_per_channel=None,
            telluric_mask_threshold=0.0,
        ).ccf_snr

    assert snr_with_lines(100) / snr_with_lines(25) == pytest.approx(2.0, rel=0.05)


def test_telluric_mask_drops_channels_and_costs_snr_monotonically():
    n = 400
    rng = np.random.default_rng(3)
    planet = rng.uniform(0.0, 1.0, n)
    leak = np.full(n, 100.0)
    telluric = rng.uniform(0.3, 1.0, n)
    ch = _channels(planet, leak)

    def snr(threshold):
        return matched_filter_ccf(
            ch,
            integration_time_s=100.0,
            eps_systematic=0.0,
            servo_bw_hz=100.0,
            with_fringe_tracking=True,
            continuum_filter_width_channels=0,
            telluric_transmission_per_channel=telluric,
            telluric_mask_threshold=threshold,
        )

    b_none, b_mid, b_hard = snr(0.0), snr(0.6), snr(0.95)
    assert b_none.n_channels_used == n
    assert b_none.ccf_snr >= b_mid.ccf_snr >= b_hard.ccf_snr
    assert b_hard.n_channels_used < b_mid.n_channels_used < n


def test_no_fringe_tracking_systematic_caps_snr_at_long_integration():
    n = 100
    lines = np.zeros(n)
    lines[5:100:10] = 1.0
    ch = _channels(lines, 100.0 * lines + 10.0)

    def snr(t):
        return matched_filter_ccf(
            ch,
            integration_time_s=t,
            eps_systematic=1e-2,
            servo_bw_hz=100.0,
            with_fringe_tracking=False,
            continuum_filter_width_channels=0,
            telluric_transmission_per_channel=None,
            telluric_mask_threshold=0.0,
        ).ccf_snr

    assert snr(1e7) / snr(1e6) < 1.05  # saturating, not sqrt(t)


def test_snr_vs_resolving_power_matches_pointwise_matched_filter():
    def build(r):
        n = 100
        lines = np.zeros(n)
        lines[5:100:10] = r / 1000.0  # more signal at higher R
        return _channels(lines, np.full(n, 100.0))

    kwargs = dict(
        integration_time_s=100.0,
        eps_systematic=0.0,
        servo_bw_hz=100.0,
        with_fringe_tracking=True,
        continuum_filter_width_channels=0,
        telluric_transmission_per_channel=None,
        telluric_mask_threshold=0.0,
    )
    r_values = np.array([1000.0, 4000.0])
    got = snr_vs_resolving_power(build, r_values, **kwargs)
    expected = [matched_filter_ccf(build(float(r)), **kwargs).ccf_snr for r in r_values]
    assert got.shape == (2,)
    assert got == pytest.approx(expected)


def test_ccf_vs_rv_peaks_at_the_true_radial_velocity():
    # Templates are the same line comb shifted by an integer number of
    # channels; the data holds the comb at a known shift on top of a smooth
    # leak, so the correlation must peak there.
    n = 300
    true_shift = 4

    def comb(shift):
        lines = np.zeros(n)
        lines[np.arange(20, 280, 20) + shift] = 3.0
        return lines

    leak = 100.0 + 5.0 * np.linspace(0.0, 1.0, n)
    data = _channels(comb(true_shift), leak)
    shifts = np.arange(-6, 7, dtype=float)
    values = ccf_vs_rv(
        lambda s: _channels(comb(int(s)), leak),
        shifts,
        data_channels=data,
        continuum_filter_width_channels=15,
        integration_time_s=100.0,
    )
    assert values.shape == shifts.shape
    assert shifts[int(np.argmax(values))] == pytest.approx(float(true_shift))


def _unequal_comb(n, positions, strengths, shift):
    lines = np.zeros(n)
    lines[np.asarray(positions) + int(shift)] = strengths
    return lines


def test_ccf_vs_rv_normalization_beats_the_unnormalized_numerator():
    # Unequal line strengths plus a telluric mask over the strongest data line.
    # The raw numerator T^T V^-1 D peaks at shift -20, where the template's
    # strong line lands on a weak data line; the normalized score peaks at the
    # true shift because dividing by the template norm removes that advantage.
    n = 140
    positions = [40, 60, 80]
    strengths = [1.0, 1.0, 10.0]
    true_shift = 0
    leak = np.full(n, 100.0)
    data = _channels(_unequal_comb(n, positions, strengths, true_shift), leak)
    telluric = np.ones(n)
    telluric[80] = 0.0  # mask the strong line
    shifts = np.arange(-20.0, 21.0, 5.0)

    def trial(shift):
        return _channels(_unequal_comb(n, positions, strengths, shift), leak)

    kwargs = dict(
        data_channels=data,
        continuum_filter_width_channels=15,
        telluric_transmission_per_channel=telluric,
        telluric_mask_threshold=0.5,
        integration_time_s=100.0,
    )
    normalized = ccf_vs_rv(trial, shifts, **kwargs)

    # Reconstruct the unnormalized numerator the fix removed.
    variance = data.planet_rate_hz + data.leak_rate_hz
    observed = continuum_filter(data.planet_rate_hz + data.leak_rate_hz, 15)
    keep = telluric >= 0.5
    raw = np.array(
        [
            np.sum(
                keep * continuum_filter(trial(s).planet_rate_hz, 15) / variance * observed
            )
            for s in shifts
        ]
    )
    assert shifts[int(np.argmax(raw))] == pytest.approx(-20.0)  # the trap
    assert shifts[int(np.argmax(normalized))] == pytest.approx(float(true_shift))


def test_ccf_vs_rv_rejects_a_trial_template_on_a_different_grid():
    n = 60
    lines = np.zeros(n)
    lines[10:60:10] = 1.0
    leak = np.full(n, 50.0)
    data = _channels(lines, leak)
    shifted_edges = data.channel_edges_um * 1.01
    off_grid = ChannelSpectra(
        channel_centers_um=0.5 * (shifted_edges[:-1] + shifted_edges[1:]),
        channel_edges_um=shifted_edges,
        planet_rate_hz=lines,
        leak_rate_hz=leak,
        sky_rate_hz=np.zeros(n),
        dark_rate_hz=np.zeros(n),
    )
    with pytest.raises(ValueError, match="different channel_edges_um"):
        ccf_vs_rv(
            lambda _rv: off_grid,
            [0.0],
            data_channels=data,
            continuum_filter_width_channels=0,
        )


def test_ccf_vs_rv_rejects_non_finite_rv_and_integration_time():
    n = 40
    data = _channels(np.ones(n), np.full(n, 10.0))
    with pytest.raises(ValueError, match="rv_grid_kms must be finite"):
        ccf_vs_rv(
            lambda _rv: data,
            [0.0, np.nan],
            data_channels=data,
            continuum_filter_width_channels=0,
        )
    with pytest.raises(ValueError, match="integration_time_s"):
        ccf_vs_rv(
            lambda _rv: data,
            [0.0],
            data_channels=data,
            continuum_filter_width_channels=0,
            integration_time_s=-1.0,
        )


def _factor(planet, leak, width):
    return matched_filter_ccf(
        _channels(planet, leak),
        integration_time_s=100.0,
        eps_systematic=0.0,
        servo_bw_hz=100.0,
        with_fringe_tracking=True,
        continuum_filter_width_channels=width,
        telluric_transmission_per_channel=None,
        telluric_mask_threshold=0.0,
    ).self_subtraction_factor


def test_self_subtraction_factor_for_sparse_absorption_dips():
    # Absorption-dominated template: the filtered template is negative in the
    # lines, which a weighted flux ratio cannot describe. Sparse dips on a
    # bright continuum retain only a small fraction of the broadband SNR.
    n = 200
    planet = np.full(n, 5.0)
    planet[20:200:20] = 1.0
    leak = np.full(n, 100.0)

    assert _factor(planet, leak, 0) == pytest.approx(1.0)
    assert 0.0 < _factor(planet, leak, 15) < 1.0
    assert 0.0 < _factor(planet, leak, 5) < 1.0


def test_self_subtraction_factor_exceeds_one_for_alternating_absorption():
    # NOT a bug and NOT to be clamped. A width-3 running median on an
    # alternating sequence returns the opposite level at every interior
    # sample, so the filter turns a bright, nearly flat spectrum into large
    # signed excursions about zero:
    #     Ptilde = [0, -4.9, +4.9, -4.9, +4.9, -4.9, 0].
    # A template with large signed excursions carries more matched-filter SNR
    # than the near-flat one it came from, so the retention ratio is > 1.
    planet = np.array([5.0, 0.1, 5.0, 0.1, 5.0, 0.1, 5.0])
    leak = np.full(planet.size, 100.0)
    assert _factor(planet, leak, 3) == pytest.approx(1.11147, rel=1e-3)


def test_self_subtraction_factor_edge_conventions():
    n = 40
    zero_planet = np.zeros(n)
    leak = np.full(n, 100.0)
    # Bypassed filtering is exactly 1.0 unconditionally: Ptilde is P, so
    # nothing was lost, even with no signal to lose (must not hit 0/0 -> 0).
    assert _factor(zero_planet, leak, 0) == 1.0
    # Active filtering with no template signal at all: 0.0 by convention.
    assert _factor(zero_planet, leak, 15) == 0.0


def test_matched_filter_rejects_non_finite_servo_bandwidth():
    n = 20
    ch = _channels(np.ones(n), np.full(n, 10.0))
    for bad in (np.nan, np.inf, 0.0, -1.0):
        with pytest.raises(ValueError, match="servo_bw_hz"):
            matched_filter_ccf(
                ch,
                integration_time_s=100.0,
                eps_systematic=1e-2,
                servo_bw_hz=float(bad),
                with_fringe_tracking=True,
                continuum_filter_width_channels=0,
                telluric_transmission_per_channel=None,
                telluric_mask_threshold=0.0,
            )


def test_continuum_filter_removes_smooth_baseline_and_keeps_narrow_lines():
    n = 200
    baseline = 50.0 + 5.0 * np.linspace(0.0, 1.0, n)
    values = baseline.copy()
    values[10:200:20] += 30.0
    filtered = continuum_filter(values, 15)
    line_idx = np.arange(10, 200, 20)
    other = np.setdiff1d(np.arange(n), line_idx)
    assert np.all(filtered[line_idx] > 25.0)
    assert np.max(np.abs(filtered[other])) < 1.0


# --- Contrast-at-target-SNR solver (spec section 6, detection mode) ----------


def _line_comb_unit_contrast(n=200, spacing=20, height=1.0):
    """Planet rate per channel at unit band-integrated contrast."""
    p1 = np.zeros(n)
    p1[spacing // 2 :: spacing] = height
    return p1


def _solver_kwargs(**overrides):
    kwargs = dict(
        integration_time_s=3600.0,
        target_snr=5.0,
        eps_systematic=1e-3,
        servo_bw_hz=100.0,
        with_fringe_tracking=True,
        continuum_filter_width_channels=0,
        telluric_transmission_per_channel=None,
        telluric_mask_threshold=0.0,
    )
    kwargs.update(overrides)
    return kwargs


def _matched_filter_kwargs(solver_kwargs):
    """The matched_filter_ccf kwargs corresponding to a solver call."""
    shared = dict(solver_kwargs)
    shared.pop("target_snr")
    return shared


def test_ccf_contrast_solver_matches_matched_filter_roundtrip():
    # Solve for c at target 5, then evaluate matched_filter_ccf with
    # planet = c * P1 on the same channels/kwargs: SNR must come back to 5.
    # The solver freezes the weights at c=0 while the filter rebuilds them
    # with the planet included, so the agreement is only first-order in the
    # planet/leak ratio.
    n = 200
    p1 = _line_comb_unit_contrast(n)
    zeros = np.zeros(n)

    def roundtrip(leak_level):
        leak = np.full(n, leak_level)
        kwargs = _solver_kwargs()
        solution = ccf_contrast_at_target_snr(
            p1, leak, zeros, zeros, **kwargs
        )
        contrast = solution.contrast_at_target_snr
        budget = matched_filter_ccf(
            _channels(contrast * p1, leak), **_matched_filter_kwargs(kwargs)
        )
        return budget.ccf_snr

    assert roundtrip(1.0e3) == pytest.approx(5.0, rel=2e-2)
    # Planet term negligible against the leak: the frozen-weight error is
    # second order in planet/leak, so a 1e4x deeper leak recovers 5.0 exactly.
    assert roundtrip(1.0e7) == pytest.approx(5.0, rel=1e-10)


def test_ccf_contrast_solver_flat_case_reduces_to_broadband_quadratic():
    # Flat P1, flat leak, no filtering, no mask, eps = 0. The channelized
    # matched filter must collapse onto the scalar broadband condition
    # S^2 = n^2 (S + B) with S = c * P_tot * t and B = L_tot * t.
    n_ch = 64
    p_per_channel = 0.02
    l_per_channel = 40.0
    time_s = 1800.0
    target = 7.0
    p1 = np.full(n_ch, p_per_channel)
    leak = np.full(n_ch, l_per_channel)
    zeros = np.zeros(n_ch)

    solution = ccf_contrast_at_target_snr(
        p1,
        leak,
        zeros,
        zeros,
        **_solver_kwargs(
            integration_time_s=time_s, target_snr=target, eps_systematic=0.0
        ),
    )

    # Positive root of (c P t)^2 - n^2 c P t - n^2 L t = 0 with P, L totals.
    p_tot_t = n_ch * p_per_channel * time_s
    l_tot_t = n_ch * l_per_channel * time_s
    disc = target**4 * p_tot_t**2 + 4.0 * p_tot_t**2 * target**2 * l_tot_t
    expected = (target**2 * p_tot_t + np.sqrt(disc)) / (2.0 * p_tot_t**2)
    assert solution.contrast_at_target_snr == pytest.approx(expected, rel=1e-10)


def test_ccf_contrast_degrades_monotonically_with_eps():
    n = 200
    p1 = _line_comb_unit_contrast(n)
    leak = np.full(n, 1.0e3)
    zeros = np.zeros(n)

    contrasts = [
        ccf_contrast_at_target_snr(
            p1,
            leak,
            zeros,
            zeros,
            **_solver_kwargs(
                eps_systematic=eps, integration_time_s=1.0e5, servo_bw_hz=1.0
            ),
        ).contrast_at_target_snr
        for eps in (0.0, 1e-3, 1e-2)
    ]
    assert contrasts[0] < contrasts[1] < contrasts[2]


def test_ccf_contrast_solver_raises_without_template_signal():
    n = 50
    zeros = np.zeros(n)
    with pytest.raises(ValueError, match="template signal"):
        ccf_contrast_at_target_snr(
            zeros, np.full(n, 100.0), zeros, zeros, **_solver_kwargs()
        )


def test_ccf_contrast_no_ft_regime_still_finite():
    n = 200
    p1 = _line_comb_unit_contrast(n)
    leak = np.full(n, 1.0e3)
    zeros = np.zeros(n)

    def contrast(with_ft):
        return ccf_contrast_at_target_snr(
            p1,
            leak,
            zeros,
            zeros,
            **_solver_kwargs(eps_systematic=1e-2, with_fringe_tracking=with_ft),
        ).contrast_at_target_snr

    without_ft = contrast(False)
    assert np.isfinite(without_ft)
    # t = 3600 s, f_servo = 100 Hz: (eps B_w t)^2 >> eps^2 B_w^2 t / (2 f),
    # so the static-offset regime costs contrast.
    assert without_ft > contrast(True)


@pytest.mark.parametrize(
    ("rate_magnitude", "time_s", "regime"),
    [
        # Old failure -> inf: A^2 overflows when squared directly.
        (1e150, 1.0, "photon"),
        # Old failure -> OverflowError: A^2 t^2 cannot be formed at all.
        (1e150, 1e160, "photon"),
        # Old failure -> ZeroDivisionError: A^2 t^2 flushes to zero.
        (1e-200, 1.0, "self_noise"),
    ],
)
def test_ccf_contrast_solver_survives_extreme_but_valid_magnitudes(
    rate_magnitude, time_s, regime
):
    # The dimensional coefficients A^2 t^2, b^2 and 4ad each blow up or vanish
    # well inside the range of representable rates. The scaled solve must
    # return a finite positive root anyway, matching the analytic limit.
    n = 4
    rates = np.full(n, rate_magnitude)
    zeros = np.zeros(n)
    solution = ccf_contrast_at_target_snr(
        rates,
        rates,
        zeros,
        zeros,
        **_solver_kwargs(
            integration_time_s=time_s, target_snr=5.0, eps_systematic=0.0
        ),
    )
    contrast = solution.contrast_at_target_snr
    assert np.isfinite(contrast)
    assert contrast > 0.0

    a_coeff = solution.signal_coefficient
    if regime == "photon":
        # C2/A = 1 here but c0 C2/A << 2 sqrt(B2/A), so c -> n / sqrt(A t).
        expected = 5.0 / (np.sqrt(a_coeff) * np.sqrt(time_s))
    else:
        # At 1e-200 Hz there are no photons in a second: the planet's own shot
        # noise dominates and the root runs to c -> n^2 C2 / (A^2 t). Grouped
        # so the reference value does not overflow either.
        ratio = 5.0 / a_coeff
        expected = ratio * (ratio * solution.planet_selfnoise_coefficient) / time_s
    assert contrast == pytest.approx(expected, rel=1e-9)


def _structured_probe():
    """Nonuniform planet template and leak, 80 channels."""
    n = 80
    index = np.arange(n)
    planet_unit = 0.2 + 2.0 * np.sin(index * 0.7) ** 2
    planet_unit[::7] += 4.0
    leak = 1.0e3 * (1.0 + 0.6 * np.cos(index * 0.31))
    leak[::5] *= 1.4
    return planet_unit, leak, np.zeros(n)


def test_ccf_contrast_frozen_weights_are_first_order_when_systematics_dominate():
    # The frozen-weight error is second order only in the photon limit. Once
    # V_sys dominates, recomputed weights shift B_w(c) at FIRST order in
    # c*P1/v0, so the solver is measurably optimistic. This test documents that
    # known bias rather than hiding it.
    planet_unit, leak, zeros = _structured_probe()
    time_s, target = 3600.0, 5.0

    def roundtrip(eps):
        kwargs = _solver_kwargs(
            integration_time_s=time_s,
            target_snr=target,
            eps_systematic=eps,
            with_fringe_tracking=False,
            continuum_filter_width_channels=11,
        )
        solution = ccf_contrast_at_target_snr(
            planet_unit, leak, zeros, zeros, **kwargs
        )
        contrast = solution.contrast_at_target_snr
        budget = matched_filter_ccf(
            _channels(contrast * planet_unit, leak),
            **_matched_filter_kwargs(kwargs),
        )
        return solution, budget.ccf_snr / target

    # Photon-limited: the second-order error is invisible.
    photon_solution, photon_ratio = roundtrip(0.0)
    assert photon_solution.systematic_var_counts == 0.0
    assert photon_ratio == pytest.approx(1.0, rel=1e-6)

    # Systematics dominating by ~90x: the solver overshoots by a few percent.
    systematic_solution, ratio = roundtrip(0.1)
    assert systematic_solution.systematic_var_counts > 50.0 * (
        time_s * systematic_solution.photon_var_zero_contrast
    )
    assert 0.90 <= ratio <= 1.02
    with pytest.raises(AssertionError):
        assert ratio == pytest.approx(1.0, rel=2e-2)

    # First order, not second: halving eps halves the error.
    _, half_ratio = roundtrip(0.05)
    assert (1.0 - ratio) / (1.0 - half_ratio) == pytest.approx(2.0, rel=0.05)


def test_ccf_contrast_solver_zeroes_weights_on_masked_and_noiseless_channels():
    n = 12
    planet_unit = np.full(n, 2.0)
    planet_unit[3] = 7.0
    leak = np.full(n, 500.0)
    leak[[4, 9]] = 0.0  # v0 = 0: no noise budget, weight must be zeroed
    sky = np.zeros(n)
    dark = np.zeros(n)
    telluric = np.ones(n)
    telluric[[1, 2]] = 0.1  # masked out
    time_s, target, eps, servo = 600.0, 5.0, 1e-2, 50.0

    solution = ccf_contrast_at_target_snr(
        planet_unit,
        leak,
        sky,
        dark,
        **_solver_kwargs(
            integration_time_s=time_s,
            target_snr=target,
            eps_systematic=eps,
            servo_bw_hz=servo,
            with_fringe_tracking=True,
            telluric_transmission_per_channel=telluric,
            telluric_mask_threshold=0.5,
        ),
    )

    usable = np.ones(n, dtype=bool)
    usable[[1, 2, 4, 9]] = False
    weights = solution.template_weights
    assert weights.shape == (n,)
    assert np.all(weights[~usable] == 0.0)
    assert np.all(weights[usable] > 0.0)
    # Filtering is off in _solver_kwargs, so Ptilde1 = P1 and Ltilde = L.
    expected_weights = np.where(usable, planet_unit / np.where(leak > 0, leak, 1.0), 0.0)
    assert weights == pytest.approx(expected_weights)

    a_coeff = float(np.sum(weights * planet_unit))
    b2_coeff = float(np.sum(weights**2 * leak))
    c2_coeff = float(np.sum(weights**2 * planet_unit))
    b_w = float(np.sum(weights * leak))
    assert solution.signal_coefficient == pytest.approx(a_coeff)
    assert solution.photon_var_zero_contrast == pytest.approx(b2_coeff)
    assert solution.planet_selfnoise_coefficient == pytest.approx(c2_coeff)
    assert solution.systematic_var_counts == pytest.approx(
        eps**2 * b_w**2 * time_s / (2.0 * servo)
    )
    assert solution.target_snr == target

    # The full field set Task 3 consumes, cross-checked against the quadratic.
    quad_a = a_coeff**2 * time_s**2
    quad_b = target**2 * time_s * c2_coeff
    quad_d = target**2 * (time_s * b2_coeff + solution.systematic_var_counts)
    expected_contrast = (
        quad_b + np.sqrt(quad_b**2 + 4.0 * quad_a * quad_d)
    ) / (2.0 * quad_a)
    assert solution.contrast_at_target_snr == pytest.approx(
        expected_contrast, rel=1e-12
    )


def test_ccf_contrast_solver_survives_large_systematic_over_small_signal():
    # q = B2/A + V_sys/(A t) overflows on its own here (V_sys/A = 1e450) even
    # though the true root, 5e160, is representable. sqrt(q) must be built as
    # hypot(sqrt(B2/A), sqrt(V_sys)/(sqrt(A) sqrt(t))) so no term is ever
    # formed at full exponent.
    solution = ccf_contrast_at_target_snr(
        np.array([1e10]),
        np.array([1e170]),
        np.zeros(1),
        np.zeros(1),
        **_solver_kwargs(
            integration_time_s=1e140,
            target_snr=5.0,
            eps_systematic=1.0,
            with_fringe_tracking=False,
        ),
    )
    contrast = solution.contrast_at_target_snr
    assert np.isfinite(contrast)
    # Systematics dominate completely, so c -> n sqrt(V_sys) / (A t), grouped
    # to keep the reference in range as well.
    expected = (
        5.0 * np.sqrt(solution.systematic_var_counts) / solution.signal_coefficient
    ) / 1e140
    assert contrast == pytest.approx(expected, rel=1e-9)
    assert contrast == pytest.approx(5e160, rel=1e-9)


def test_ccf_contrast_frozen_weight_bias_scales_with_the_systematic_share():
    # The first-order coefficient is c * eta_sys * (C2/A - D/H0) with
    # eta_sys = V_sys / (t B2 + V_sys), the systematic share of the
    # zero-contrast variance. It is not a threshold effect: the bias is first
    # order even when systematics are a minority of the budget.
    planet_unit, leak, zeros = _structured_probe()
    time_s, target = 3600.0, 5.0

    def probe(eps):
        kwargs = _solver_kwargs(
            integration_time_s=time_s,
            target_snr=target,
            eps_systematic=eps,
            with_fringe_tracking=False,
            continuum_filter_width_channels=11,
        )
        solution = ccf_contrast_at_target_snr(
            planet_unit, leak, zeros, zeros, **kwargs
        )
        contrast = solution.contrast_at_target_snr
        budget = matched_filter_ccf(
            _channels(contrast * planet_unit, leak),
            **_matched_filter_kwargs(kwargs),
        )
        eta_sys = solution.systematic_var_counts / (
            time_s * solution.photon_var_zero_contrast
            + solution.systematic_var_counts
        )
        error = 1.0 - budget.ccf_snr / target
        return eta_sys, contrast, error

    weak_eta, weak_c, weak_error = probe(0.005)
    strong_eta, strong_c, strong_error = probe(0.1)

    # Weak systematics: a minority of the variance, but a real first-order bias.
    assert 0.1 < weak_eta < 0.3
    assert weak_error > 0.0
    # Dominated.
    assert strong_eta > 0.95

    # The analytic first-order coefficient, computed independently.
    filtered_planet = continuum_filter(planet_unit, 11)
    filtered_leak = continuum_filter(leak, 11)
    weights = filtered_planet / leak
    a_coeff = float(np.sum(weights * filtered_planet))
    c2_coeff = float(np.sum(weights**2 * planet_unit))
    h0 = float(np.sum(weights * filtered_leak))
    d_coeff = float(np.sum(filtered_planet * filtered_leak * planet_unit / leak**2))
    expected_slope = c2_coeff / a_coeff - d_coeff / h0

    assert weak_error / (weak_c * weak_eta) == pytest.approx(
        expected_slope, rel=0.02
    )
    assert strong_error / (strong_c * strong_eta) == pytest.approx(
        expected_slope, rel=0.05
    )


def test_ccf_contrast_solver_raises_value_error_when_systematic_variance_overflows():
    # eps * B_w * t ~ 1e155, whose square is past the float64 ceiling. The
    # no-fringe-tracking branch squares with `**`, which raises OverflowError
    # in matched_filter_ccf; the solver must normalize it to the ValueError it
    # documents, so a per-anchor sweep has one exception type to catch.
    args = (np.array([1e10]), np.array([1e170]), np.zeros(1), np.zeros(1))
    with pytest.raises(ValueError, match="systematic variance overflows"):
        ccf_contrast_at_target_snr(
            *args,
            **_solver_kwargs(
                integration_time_s=1e145,
                target_snr=5.0,
                eps_systematic=1.0,
                with_fringe_tracking=False,
            ),
        )

    # The fringe-tracking branch multiplies rather than squaring, so it
    # saturates to inf silently; it must reach the same ValueError.
    with pytest.raises(ValueError, match="systematic variance overflows"):
        ccf_contrast_at_target_snr(
            *args,
            **_solver_kwargs(
                integration_time_s=1e160,
                target_snr=5.0,
                eps_systematic=1.0,
                servo_bw_hz=1e-130,
                with_fringe_tracking=True,
            ),
        )

    # Just below the threshold the diagnostic semantics are unchanged.
    solution = ccf_contrast_at_target_snr(
        *args,
        **_solver_kwargs(
            integration_time_s=1e140,
            target_snr=5.0,
            eps_systematic=1.0,
            with_fringe_tracking=False,
        ),
    )
    assert solution.systematic_var_counts == pytest.approx(1e300, rel=1e-9)
    assert solution.contrast_at_target_snr == pytest.approx(5e160, rel=1e-9)


def test_flat_spectra_are_exact_despite_large_systematics():
    # The structural coefficient C2/A - D/H0 vanishes identically for a flat
    # template against a flat leak. In fact the frozen and exact statistics
    # coincide to ALL orders there: with G = A/(1+x), H = H0/(1+x) and
    # B2 + c C2 = A(1+x) for x = c P1 / v0, the (1+x) factors cancel between
    # numerator and denominator. This is why the qualification is "generically
    # first order", not "first order for any nonzero systematic".
    n = 64
    planet_unit = np.full(n, 0.05)
    leak = np.full(n, 400.0)
    zeros = np.zeros(n)
    time_s, target = 3600.0, 5.0
    kwargs = _solver_kwargs(
        integration_time_s=time_s,
        target_snr=target,
        eps_systematic=5e-3,
        with_fringe_tracking=False,
        continuum_filter_width_channels=0,
    )
    solution = ccf_contrast_at_target_snr(planet_unit, leak, zeros, zeros, **kwargs)
    eta_sys = solution.systematic_var_counts / (
        time_s * solution.photon_var_zero_contrast + solution.systematic_var_counts
    )
    assert eta_sys > 0.9  # systematics dominate

    contrast = solution.contrast_at_target_snr
    budget = matched_filter_ccf(
        _channels(contrast * planet_unit, leak), **_matched_filter_kwargs(kwargs)
    )
    ratio = budget.ccf_snr / target
    # c P1 / v0 = 0.025 here, larger than the structured probe's 0.017 where
    # the error was 3.6%. Flat, it is exact to machine precision.
    assert contrast * planet_unit[0] / leak[0] > 0.02
    assert ratio == pytest.approx(1.0, rel=1e-12)
