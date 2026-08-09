"""Tests for nullsim.physics.hdc (photon-weighted CCF budget)."""
import math

import numpy as np
import pytest

from nullsim.physics.hdc import (
    ChannelSpectra,
    build_channel_edges,
    continuum_filter_adjoint,
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
    # The linear high-pass has support exactly `width` channels wide, so
    # "orthogonal" means the filtered supports do not overlap: planet and leak
    # combs must be separated by more than the filter width. (The running
    # median this replaced had no such spreading, which is exactly why its
    # noise did not propagate in closed form.)
    n = 400
    planet = np.zeros(n)
    planet[20:400:40] = 5.0
    leak = np.full(n, 50.0)
    leak[40:400:40] += 40.0  # 20 channels away, wider than the width-15 filter
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


def test_telluric_mask_costs_snr_monotonically_when_filtering_is_bypassed():
    # Monotonicity holds ONLY in the bypass case used here (width 0, H = I),
    # where SNR^2 = t sum_i m_i Ptilde_i^2/v_i is a sum of non-negative terms.
    # With a filter active, dropping a channel changes H itself and can raise
    # the SNR -- 1.871 to 4.266 on a six-channel width-3 probe. Do not
    # generalize this test's claim.
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


def test_self_subtraction_factor_on_the_alternating_absorption_case():
    # Re-pinned for the linear filter. The width-3 boxcar on an alternating
    # sequence gives Ptilde = [2.45, -3.267, +3.267, ..., 2.45] rather than the
    # median's [0, -4.9, +4.9, ...], and the retained SNR now also pays for the
    # noise correlation H introduces, so the factor lands at 0.637 instead of
    # the 1.111 this case used to report.
    #
    # That 1.111 was not a real gain: the old number divided a filtered signal
    # by an unfiltered diagonal variance. Measured against its own true
    # variance the median statistic retained ~0.58 here, so the linear filter
    # is in fact the better of the two once both are honestly costed.
    planet = np.array([5.0, 0.1, 5.0, 0.1, 5.0, 0.1, 5.0])
    leak = np.full(planet.size, 100.0)
    assert _factor(planet, leak, 3) == pytest.approx(0.637088, rel=1e-4)


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
    width = 15
    filtered = continuum_filter(values, width)
    line_idx = np.arange(10, 200, 20)
    half = width // 2
    far = np.array(
        [i for i in range(n) if np.min(np.abs(i - line_idx)) > half]
    )

    # Line spacing (20) exceeds the width, so each window holds exactly one
    # line: the boxcar mean is lifted by 30/15 = 2 and the retained peak is
    # 30 - 2 = 28 exactly.
    assert filtered[line_idx] == pytest.approx(28.0)
    # Away from every line the smooth baseline is gone. A constant is removed
    # exactly at all channels (S preserves constants including at the edges);
    # what is left is the ramp's second-order edge residual.
    assert np.max(np.abs(filtered[far])) < 0.1
    # Within a filter width of a line the linear smoother leaves negative
    # wings -- the price of linearity, and the reason self-subtraction is
    # larger than it was with a median.
    wings = np.setdiff1d(np.arange(n), np.union1d(line_idx, far))
    assert np.min(filtered[wings]) < -1.5


def test_continuum_filter_annihilates_a_constant_exactly_everywhere():
    # S 1 = 1 by the window-fraction normalization, so H 1 = 0 at the edges as
    # well as the interior: a flat stellar continuum leaves no edge residual.
    for n, width in ((7, 3), (64, 15), (200, 31)):
        assert continuum_filter(np.full(n, 137.0), width) == pytest.approx(
            np.zeros(n), abs=1e-12
        )


def test_continuum_filter_adjoint_matches_the_dense_transpose():
    # The variance propagation is only correct if H^T really is the transpose.
    for n, width in ((12, 3), (31, 7), (40, 15)):
        basis = np.eye(n)
        forward = np.stack([continuum_filter(basis[i], width) for i in range(n)]).T
        adjoint = np.stack(
            [continuum_filter_adjoint(basis[i], width) for i in range(n)]
        ).T
        assert adjoint == pytest.approx(forward.T, abs=1e-14)


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
    # The frozen-weight error is small but nonzero in the photon limit and
    # grows with the systematic share. This test documents that known bias
    # rather than hiding it.
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

    # Photon-limited. With the linear high-pass this is no longer *exactly*
    # second order -- H^T couples the weight perturbation into the variance --
    # but the residual slope is ~3e2 times smaller than the systematics-driven
    # one, so the roundtrip is still good to a part in 1e4.
    photon_solution, photon_ratio = roundtrip(0.0)
    assert photon_solution.systematic_var_counts == 0.0
    assert photon_ratio == pytest.approx(1.0, rel=1e-4)
    assert photon_ratio != pytest.approx(1.0, rel=1e-6)

    # Systematics dominating by ~124x: the solver overshoots by a few percent.
    systematic_solution, ratio = roundtrip(0.05)
    assert systematic_solution.systematic_var_counts > 50.0 * (
        time_s * systematic_solution.photon_var_zero_contrast
    )
    assert 0.90 <= ratio <= 1.02
    with pytest.raises(AssertionError):
        assert ratio == pytest.approx(1.0, rel=2e-2)

    # First order, not second: halving eps halves the error.
    _, half_ratio = roundtrip(0.025)
    assert (1.0 - ratio) / (1.0 - half_ratio) == pytest.approx(2.0, rel=0.05)


def _frozen_weight_slope(planet_unit, leak, width):
    """Analytic first-order coefficient of the frozen-weight bias.

    ``1 - SNR_true/SNR_frozen ~= c * Delta(eta_sys)`` with

        Delta = C2'/A - eta_ph * E/B2 - eta_sys * D/J0

    where the primed/extra terms are the ones the linear high-pass introduces:
    ``C2' = sum Ptilde1^2 P1 / v0^2`` (from d/dc of the signal), and
    ``E = sum u0 (H^T g) v0`` with ``g = Ptilde1 P1 / v0^2`` (from d/dc of the
    H-propagated variance). Without a filter ``u0 = w0``, ``C2' = E = C2`` and
    the whole thing collapses to ``eta_sys (C2/A - D/J0)``.
    """
    v0 = leak
    filtered_planet = continuum_filter(planet_unit, width)
    filtered_leak = continuum_filter(leak, width)
    w0 = filtered_planet / v0
    u0 = continuum_filter_adjoint(w0, width)
    a_coeff = float(np.sum(w0 * filtered_planet))
    b2 = float(np.sum(u0 * u0 * v0))
    g = filtered_planet * planet_unit / v0**2
    e_coeff = float(np.sum(u0 * continuum_filter_adjoint(g, width) * v0))
    c2_prime = float(np.sum(filtered_planet**2 * planet_unit / v0**2))
    d_coeff = float(
        np.sum(filtered_planet * filtered_leak * planet_unit / v0**2)
    )
    j0 = float(np.sum(w0 * filtered_leak))

    def slope(eta_sys):
        return c2_prime / a_coeff - (1.0 - eta_sys) * e_coeff / b2 - eta_sys * (
            d_coeff / j0
        )

    return slope


def test_ccf_contrast_frozen_weight_bias_scales_with_the_systematic_share():
    # The first-order coefficient interpolates between a photon-limited term
    # and a systematics term, weighted by eta_sys = V_sys / (t B2 + V_sys).
    # It is not a threshold effect: the bias is first order at every eta.
    planet_unit, leak, zeros = _structured_probe()
    time_s, target, width = 3600.0, 5.0, 11
    slope = _frozen_weight_slope(planet_unit, leak, width)

    def probe(eps):
        kwargs = _solver_kwargs(
            integration_time_s=time_s,
            target_snr=target,
            eps_systematic=eps,
            with_fringe_tracking=False,
            continuum_filter_width_channels=width,
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

    weak_eta, weak_c, weak_error = probe(0.002)
    strong_eta, strong_c, strong_error = probe(0.0125)

    # Weak systematics: a minority of the variance, but a real first-order bias.
    assert 0.1 < weak_eta < 0.3
    assert weak_error > 0.0
    assert strong_eta > 0.85

    assert weak_error / weak_c == pytest.approx(slope(weak_eta), rel=0.03)
    assert strong_error / strong_c == pytest.approx(slope(strong_eta), rel=0.03)

    # The photon-limited end of the same formula: small, negative (the solver
    # is very slightly pessimistic there), and no longer identically zero.
    assert slope(0.0) == pytest.approx(-8.268e-05, rel=1e-3)
    assert abs(slope(0.0)) < 0.01 * abs(slope(1.0))


def test_ccf_contrast_solver_zeroes_weights_on_masked_channels():
    # Re-scoped: this used to pin zero-variance channels as excluded too, which
    # is wrong for the solver (see the self-noise test below). Every channel
    # here has a real leak, so only the telluric mask removes any.
    n = 12
    planet_unit = np.full(n, 2.0)
    planet_unit[3] = 7.0
    leak = np.full(n, 500.0)
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
    usable[[1, 2]] = False
    weights = solution.template_weights
    assert weights.shape == (n,)
    assert np.all(weights[~usable] == 0.0)
    assert np.all(weights[usable] > 0.0)
    # Filtering is off in _solver_kwargs, so Ptilde1 = P1 and Ltilde = L.
    expected_weights = np.where(usable, planet_unit / leak, 0.0)
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

    quad_a = a_coeff**2 * time_s**2
    quad_b = target**2 * time_s * c2_coeff
    quad_d = target**2 * (time_s * b2_coeff + solution.systematic_var_counts)
    expected_contrast = (
        quad_b + np.sqrt(quad_b**2 + 4.0 * quad_a * quad_d)
    ) / (2.0 * quad_a)
    assert solution.contrast_at_target_snr == pytest.approx(
        expected_contrast, rel=1e-12
    )


def test_ccf_contrast_solver_keeps_noiseless_channels_that_carry_signal():
    # A channel with v0 = 0 but P1 > 0 is NOT empty for the solver: at contrast
    # c its variance is c*P1, so it is informative through the planet's own
    # shot noise -- the c*C2 term the quadratic already carries. The detection
    # statistic drops it (no information at fixed contrast); the solver must
    # not. Single channel, unit template, no background: SNR = sqrt(c t), so
    # c = n^2/t = 25 exactly.
    solution = ccf_contrast_at_target_snr(
        np.array([1.0]),
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
        **_solver_kwargs(
            integration_time_s=1.0,
            target_snr=5.0,
            eps_systematic=0.0,
            continuum_filter_width_channels=0,
        ),
    )
    assert solution.contrast_at_target_snr == pytest.approx(25.0, rel=1e-12)
    assert solution.photon_var_zero_contrast == 0.0

    # v0 = 0 AND P1 = 0 is genuinely empty and still raises.
    with pytest.raises(ValueError, match="no template signal survives"):
        ccf_contrast_at_target_snr(
            np.zeros(1),
            np.zeros(1),
            np.zeros(1),
            np.zeros(1),
            **_solver_kwargs(
                integration_time_s=1.0,
                target_snr=5.0,
                eps_systematic=0.0,
                continuum_filter_width_channels=0,
            ),
        )


def test_ccf_contrast_solver_root_is_time_independent_when_systematics_dominate():
    # Re-pinned: this used to assert that a large eps*B_w*t RAISED. It should
    # not -- the root needs sqrt(V_sys)/t, in which the t cancels analytically,
    # so squaring eps*B_w*t only to square-root it again was a pure overflow
    # hazard. The systematics-dominated root c -> n eps B_w / A carries no t at
    # all, and now returns the same value across four decades of exposure.
    args = (np.array([1e10]), np.array([1e170]), np.zeros(1), np.zeros(1))
    for time_s in (1e140, 1e145, 1e200):
        solution = ccf_contrast_at_target_snr(
            *args,
            **_solver_kwargs(
                integration_time_s=time_s,
                target_snr=5.0,
                eps_systematic=1.0,
                with_fringe_tracking=False,
            ),
        )
        assert solution.contrast_at_target_snr == pytest.approx(5e160, rel=1e-9)


def test_flat_spectra_are_exact_despite_large_systematics():
    # The structural coefficient C2/A - D/J0 vanishes identically for a flat
    # template against a flat leak. In fact the frozen and exact statistics
    # coincide to ALL orders when x_i = c P1_i / v0_i is channel-independent:
    # only then can a single common (1+x) be factored out of each sum, giving
    # G = A/(1+x), H = H0/(1+x), B2 + c C2 = A(1+x), and cancelling between
    # numerator and denominator. Sky and dark are zero here, so v0 = leak is
    # uniform and this case satisfies that hypothesis by construction -- hence
    # the 1e-12 tolerance. With structured sky the ratio varies, exactness is
    # lost (~1e-9 residual) though the first-order cancellation survives.
    # This is why the qualification is "generically first order", not "first
    # order for any nonzero systematic".
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


@pytest.mark.stats
def test_monte_carlo_photon_variance_matches_the_propagated_prediction():
    """Poisson realizations through the real filter must match V_ph.

    This is the test that would have caught the covariance defect. The
    estimator is X = w . (H N) on independent Poisson counts N, so its variance
    is t sum_i (H^T w)_i^2 v_i. Pairing a filtered signal model with the raw
    diagonal t sum_i w_i^2 v_i -- which is what this module did before the
    linear filter landed -- understates the noise, and no analytic test that
    only checks internal consistency can see it. Only drawing photons can.
    """
    n_channels, width = 200, 3
    planet = np.zeros(n_channels)
    planet[10::20] = 5.0
    leak = np.full(n_channels, 100.0)
    time_s = 500.0
    channels = _channels(planet, leak)

    budget = matched_filter_ccf(
        channels,
        integration_time_s=time_s,
        eps_systematic=0.0,
        servo_bw_hz=100.0,
        with_fringe_tracking=True,
        continuum_filter_width_channels=width,
        telluric_transmission_per_channel=None,
        telluric_mask_threshold=0.0,
    )
    weights = budget.template_weights
    rate = channels.total_rate_hz()

    n_realizations, chunk = 100_000, 10_000
    rng = np.random.default_rng(20260807)
    total = total_sq = 0.0
    for _ in range(n_realizations // chunk):
        counts = rng.poisson(np.broadcast_to(rate * time_s, (chunk, n_channels)))
        # continuum_filter is 1-D by contract; the smoother underneath is not,
        # so filter each realization the way the estimator would.
        filtered = np.stack(
            [continuum_filter(row, width) for row in counts.astype(np.float64)]
        )
        statistic = filtered @ weights
        total += float(statistic.sum())
        total_sq += float((statistic * statistic).sum())
    empirical_mean = total / n_realizations
    empirical_var = total_sq / n_realizations - empirical_mean**2

    # sqrt(2/N) = 0.45% standard error on the variance; 2% is a comfortable
    # band that still catches the 23% error the diagonal formula makes here.
    assert empirical_var == pytest.approx(budget.photon_var_counts, rel=0.02)

    # The mean of the statistic is the filtered total rate projected on w --
    # planet plus whatever continuum survives the filter.
    expected_mean = time_s * float(
        np.sum(weights * continuum_filter(rate, width))
    )
    assert empirical_mean == pytest.approx(expected_mean, rel=0.05)

    # And the defect is real: the old diagonal expression is off by ~23% here,
    # far outside the MC band, so this assertion would have failed before.
    diagonal_var = time_s * float(np.sum(weights * weights * rate))
    assert diagonal_var / empirical_var < 0.85

    # The empirical SNR of the reported signal against the drawn noise.
    empirical_snr = budget.signal_counts / np.sqrt(empirical_var)
    assert empirical_snr == pytest.approx(budget.ccf_snr, rel=0.02)


def _shifted_into(template, shift, n_channels):
    """`template` slid by `shift` channels, zero-filled where it slides in."""
    out = np.zeros(n_channels)
    if shift >= 0:
        out[shift:] = template[: n_channels - shift]
    else:
        out[: n_channels + shift] = template[-shift:]
    return out


def test_ccf_vs_rv_peak_survives_a_zero_filled_shifted_template():
    # Regression for the review counterexample. Pairing diag(v)^-1 Ttilde
    # weights with the C_y denominator is not a cosine under any single inner
    # product, and a template slid eight channels (zero-filled behind it) beat
    # the truth 3.591 to 3.383. The whitened statistic restores the
    # Cauchy-Schwarz guarantee.
    planet = np.array(
        [
            0.532967, 1.263326, 1.007460, 1.088272, 3.241559,
            2.635570, 1.414478, 9.568137, 0.156691, 66.788444,
        ]
    )
    n = planet.size
    leak = np.full(n, 100.0)
    data = _channels(planet, leak)
    shifts = np.arange(-9.0, 10.0)

    scores = ccf_vs_rv(
        lambda rv: _channels(_shifted_into(planet, int(rv), n), leak),
        shifts,
        data_channels=data,
        continuum_filter_width_channels=3,
        integration_time_s=1.0,
    )
    assert shifts[int(np.argmax(scores))] == 0.0
    others = np.delete(scores, np.where(shifts == 0.0))
    assert scores[shifts == 0.0][0] > others.max()


def test_ccf_vs_rv_matches_the_dense_pseudoinverse_statistic():
    # The O(n) closed form must equal the literal C_y^+ statistic. H = I - S
    # with S row-stochastic and irreducible, so null(H) = span{1} exactly and
    # the pseudoinverse optimum over range(H^T) = 1-perp is the inverse-variance
    # weighting with a variance-weighted mean removed.
    rng = np.random.default_rng(4)
    n, width = 24, 5
    planet = rng.uniform(0.0, 10.0, n)
    leak = rng.uniform(5.0, 200.0, n)
    data = _channels(planet, leak)
    template = rng.uniform(0.0, 10.0, n)

    (got,) = ccf_vs_rv(
        lambda _rv: _channels(template, leak),
        [0.0],
        data_channels=data,
        continuum_filter_width_channels=width,
        integration_time_s=1.0,
    )

    basis = np.eye(n)
    h_matrix = np.stack([continuum_filter(basis[i], width) for i in range(n)]).T
    variance = data.total_rate_hz()
    covariance = h_matrix @ np.diag(variance) @ h_matrix.T
    pseudo = np.linalg.pinv(covariance, rcond=1e-12)
    filtered_template = h_matrix @ template
    filtered_data = h_matrix @ (planet + leak)
    expected = (filtered_template @ pseudo @ filtered_data) / np.sqrt(
        filtered_template @ pseudo @ filtered_template
    )
    assert got == pytest.approx(expected, rel=1e-8)


def test_ccf_contrast_solver_survives_large_b2_over_small_signal():
    # Regression: the overflow hardening formed sqrt(B2/A), which overflows now
    # that the covariance fix broke the A = B2 identity (B2 ~ 1e126 against
    # A ~ 1e-191 here). The root itself is perfectly representable.
    solution = ccf_contrast_at_target_snr(
        np.array([1e-100, 0.0, 0.0]),
        np.array([1e-10, 1e-10, 1e308]),
        np.zeros(3),
        np.zeros(3),
        **_solver_kwargs(
            integration_time_s=1.0,
            target_snr=5.0,
            eps_systematic=0.0,
            continuum_filter_width_channels=3,
        ),
    )
    contrast = solution.contrast_at_target_snr
    assert np.isfinite(contrast)
    assert contrast == pytest.approx(1.538462e254, rel=1e-6)
    # Photon-limited with negligible self-noise: c = n sqrt(B2) / (A t).
    expected = (
        5.0
        * math.sqrt(solution.photon_var_zero_contrast)
        / solution.signal_coefficient
    )
    assert contrast == pytest.approx(expected, rel=1e-9)


# --- masking semantics and scale hardening (review round 2) -----------------


def test_mask_semantics_agree_between_detection_and_rv_statistics():
    # Regression: matched_filter_ccf masked the filtered-output weights and then
    # formed a = H^T w, which H^T spreads back onto masked channels, so the
    # detection statistic read photons the mask had excluded (0.1936) while the
    # RV scan, constrained in raw space, reported 0.0. Both now use raw-channel
    # exclusion, so both see the same subspace.
    planet = np.array([4.0, 1.0, 0.5, 2.0, 2.0])
    leak = np.full(5, 10.0) - planet
    telluric = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
    data = _channels(planet, leak)
    mask_kwargs = dict(
        telluric_transmission_per_channel=telluric,
        telluric_mask_threshold=0.5,
    )

    budget = matched_filter_ccf(
        data,
        integration_time_s=1.0,
        eps_systematic=0.0,
        servo_bw_hz=100.0,
        with_fringe_tracking=True,
        continuum_filter_width_channels=3,
        **mask_kwargs,
    )
    (rv_score,) = ccf_vs_rv(
        lambda _rv: _channels(planet, leak),
        [0.0],
        data_channels=data,
        continuum_filter_width_channels=3,
        integration_time_s=1.0,
        **mask_kwargs,
    )

    assert budget.n_channels_used == 2
    assert np.all(budget.template_weights[:3] == 0.0)
    # Two admitted channels with equal planet rate: a zero-sum filter on them
    # sees nothing, and the detection statistic must agree rather than
    # manufacturing signal out of masked photons.
    assert budget.ccf_snr_photon_limited == pytest.approx(0.0)
    assert rv_score == pytest.approx(0.0)
    assert rv_score >= budget.ccf_snr_photon_limited


def test_rv_scan_dominates_the_detection_snr_for_a_smooth_leak_under_a_mask():
    # Conditional, not a theorem. The two statistics correlate against
    # different vectors -- the scan against observed planet-plus-leak, the
    # detection signal against the planet alone -- so the ordering holds only
    # when the filtered leak projection is small, which a smooth leak delivers.
    # What is pinned here is that masking no longer breaks the comparison,
    # which it did when the two used different feasible subspaces.
    rng = np.random.default_rng(17)
    n = 40
    planet = rng.uniform(0.0, 3.0, n)
    leak = np.full(n, 120.0)  # smooth: H annihilates it
    telluric = rng.uniform(0.0, 1.0, n)
    data = _channels(planet, leak)
    mask_kwargs = dict(
        telluric_transmission_per_channel=telluric,
        telluric_mask_threshold=0.4,
    )
    budget = matched_filter_ccf(
        data,
        integration_time_s=250.0,
        eps_systematic=0.0,
        servo_bw_hz=100.0,
        with_fringe_tracking=True,
        continuum_filter_width_channels=7,
        **mask_kwargs,
    )
    (rv_score,) = ccf_vs_rv(
        lambda _rv: _channels(planet, leak),
        [0.0],
        data_channels=data,
        continuum_filter_width_channels=7,
        integration_time_s=250.0,
        **mask_kwargs,
    )
    assert budget.n_channels_used == int(np.count_nonzero(telluric >= 0.4))
    assert rv_score >= budget.ccf_snr_photon_limited * (1.0 - 1e-12)


def test_rv_scan_peak_is_not_guaranteed_under_a_structured_leak():
    # Executable form of the limitation. Cauchy-Schwarz equality is with the
    # OBSERVED spectrum, not the planet template, so when the leak carries
    # structure H cannot remove, the scan locks onto the leak instead of the
    # planet. Asserts only what is true: the peak is elsewhere, and the true
    # velocity is not even competitive.
    planet = np.array([
        0.832684, 0.252337, 0.708147, 1.957615, 0.362219, 1.973113,
        0.068271, 0.745769, 0.495582, 0.036953, 0.324502, 0.642361,
    ])
    leak = np.array([
        0.562142, 22.238429, 20.869430, 0.203414, 46.334429, 7.654591,
        13.283852, 36.362523, 12.645643, 35.714879, 36.467156, 25.478085,
    ])
    n = planet.size
    data = _channels(planet, leak)
    shifts = np.arange(-4.0, 5.0)
    scores = ccf_vs_rv(
        lambda rv: _channels(_shifted_into(planet, int(rv), n), leak),
        shifts,
        data_channels=data,
        continuum_filter_width_channels=3,
        integration_time_s=1.0,
    )
    true_score = float(scores[shifts == 0.0][0])
    assert shifts[int(np.argmax(scores))] != 0.0
    assert true_score == pytest.approx(-5.9878, rel=1e-3)
    assert float(scores.max()) == pytest.approx(9.7991, rel=1e-3)


def test_telluric_masking_is_not_monotonic_in_snr_with_the_filter_active():
    # Executable form of the note's warning. With H active, dropping a channel
    # changes the operator itself -- window membership, edge renormalization,
    # and island boundaries -- so a noisy channel can cost more in propagated
    # variance than it returns in signal. Here masking the last channel raises
    # the SNR from 5.12 to 6.56.
    planet = np.array(
        [2.576062, 1.979607, 0.169474, 1.361213, 1.801669, 2.995946, 0.346343, 0.488785]
    )
    leak = np.array(
        [13.305713, 41.203689, 8.748524, 23.372507, 32.866405, 7.757541, 5.281987, 44.416256]
    )
    channels = _channels(planet, leak)
    kwargs = dict(
        integration_time_s=100.0,
        eps_systematic=0.0,
        servo_bw_hz=100.0,
        with_fringe_tracking=True,
        continuum_filter_width_channels=3,
    )
    full = matched_filter_ccf(
        channels,
        telluric_transmission_per_channel=None,
        telluric_mask_threshold=0.0,
        **kwargs,
    ).ccf_snr
    telluric = np.ones(planet.size)
    telluric[-1] = 0.0
    masked = matched_filter_ccf(
        channels,
        telluric_transmission_per_channel=telluric,
        telluric_mask_threshold=0.5,
        **kwargs,
    ).ccf_snr
    assert full == pytest.approx(5.122431, rel=1e-4)
    assert masked == pytest.approx(6.556789, rel=1e-4)
    assert masked > full  # dropping a channel RAISED the SNR


def test_island_filter_does_not_bridge_a_telluric_gap():
    # Two flat admitted segments either side of a six-channel masked band. They
    # are not neighbours in wavelength, and compacting them into one vector had
    # the width-3 boxcar infer a ramp between them, manufacturing +-30
    # residuals in a spectrum with no line anywhere. Filtering each island
    # separately annihilates each flat segment exactly.
    n = 12
    planet = np.zeros(n)
    planet[0:3] = 10.0
    planet[9:12] = 100.0
    telluric = np.zeros(n)
    telluric[0:3] = 1.0
    telluric[9:12] = 1.0
    channels = _channels(planet, np.zeros(n))
    mask_kwargs = dict(
        telluric_transmission_per_channel=telluric,
        telluric_mask_threshold=0.5,
    )
    budget = matched_filter_ccf(
        channels,
        integration_time_s=1.0,
        eps_systematic=0.0,
        servo_bw_hz=100.0,
        with_fringe_tracking=True,
        continuum_filter_width_channels=3,
        **mask_kwargs,
    )
    (rv_score,) = ccf_vs_rv(
        lambda _rv: _channels(planet, np.zeros(n)),
        [0.0],
        data_channels=channels,
        continuum_filter_width_channels=3,
        integration_time_s=1.0,
        **mask_kwargs,
    )
    assert budget.n_channels_used == 6
    # Exact to rounding: the residue is the boxcar's last-ulp error on a flat
    # segment, not manufactured structure. Bridged, these were +-30.
    assert budget.signal_counts == pytest.approx(0.0, abs=1e-9)
    assert budget.ccf_snr < 1e-12
    assert rv_score == 0.0
    assert np.max(np.abs(budget.template_weights)) < 1e-12


def test_ccf_vs_rv_bypass_uses_the_unconstrained_optimum():
    # width <= 0 means H = I, whose null space is empty, so there is no
    # zero-sum constraint and no mean to remove: the optimum is a = T/v.
    # Applying the constraint anyway returned 0 for a single channel.
    rng = np.random.default_rng(2)
    for n in (1, 2, 5):
        planet = rng.uniform(0.5, 5.0, n)
        leak = rng.uniform(0.5, 5.0, n)
        data = _channels(planet, leak)
        (got,) = ccf_vs_rv(
            lambda _rv: _channels(planet, leak),
            [0.0],
            data_channels=data,
            continuum_filter_width_channels=0,
            integration_time_s=7.0,
        )
        rate = planet + leak
        weights = planet / rate
        expected = (
            math.sqrt(7.0)
            * float(weights @ rate)
            / math.sqrt(float(np.sum(weights * weights * rate)))
        )
        assert got == pytest.approx(expected, rel=1e-12)
        assert got > 0.0


def test_ccf_vs_rv_survives_extreme_but_finite_variances():
    # 1/v overflows on its own at v ~ 1e-308; normalizing the inverse variances
    # and the filter vector before squaring keeps the score finite.
    for magnitude in (1e-308, 1e-150, 1.0, 1e150):
        planet = np.array([2.0, 1.0]) * magnitude
        data = _channels(planet, np.zeros(2))
        (got,) = ccf_vs_rv(
            lambda _rv: _channels(np.array([1.0, 2.0]), np.zeros(2)),
            [0.0],
            data_channels=data,
            continuum_filter_width_channels=3,
            integration_time_s=1e308,
        )
        direction = np.array([-1.0, 1.0])
        expected = (
            math.sqrt(1e308)
            * float(direction @ planet)
            / math.sqrt(float(np.sum(direction * direction * planet)))
        )
        assert np.isfinite(got)
        assert got == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    ("planet_unit", "leak", "time_s", "expected", "label"),
    [
        (
            [1e-200, 0.0, 0.0],
            1e100,
            1e308,
            8.373669637337405e96,
            "A underflows unscaled",
        ),
        (
            [1e200, 0.0, 0.0],
            1e100,
            1e-100,
            2.754554230312972e-99,
            "C2 overflows unscaled",
        ),
        ([1.0, 0.0, 0.0], 1e-200, 1.0, 25.0, "w^2 overflows unscaled"),
    ],
)
def test_ccf_contrast_solver_scales_the_coefficient_accumulation(
    planet_unit, leak, time_s, expected, label
):
    # The hardened root formula was not enough: A, B2 and C2 themselves
    # over/underflow while being accumulated from raw rates. Normalizing by
    # alpha = max|P1|, beta = sqrt(min v0 max v0) and gamma = max|w| fixes it.
    planet_array = np.array(planet_unit, dtype=np.float64)
    leak_array = np.full(planet_array.size, leak)
    zeros = np.zeros(planet_array.size)
    solution = ccf_contrast_at_target_snr(
        planet_array,
        leak_array,
        zeros,
        zeros,
        **_solver_kwargs(
            integration_time_s=time_s,
            target_snr=5.0,
            eps_systematic=0.0,
            continuum_filter_width_channels=3,
        ),
    )
    contrast = solution.contrast_at_target_snr
    assert np.isfinite(contrast)
    assert contrast > 0.0
    assert contrast == pytest.approx(expected, rel=1e-12)


def test_ccf_contrast_solver_handles_a_target_snr_that_squares_to_zero():
    # n^2 underflows for a 1e-200 target, which used to zero the constant term
    # and return 0. n now appears at most once per product.
    solution = ccf_contrast_at_target_snr(
        np.array([1.0, 2.0, 3.0]),
        np.full(3, 100.0),
        np.zeros(3),
        np.zeros(3),
        **_solver_kwargs(
            integration_time_s=3600.0,
            target_snr=1e-200,
            eps_systematic=0.0,
            continuum_filter_width_channels=3,
        ),
    )
    assert solution.contrast_at_target_snr > 0.0
    assert np.isfinite(solution.contrast_at_target_snr)


def test_ccf_contrast_root_survives_a_discriminant_that_doubling_would_overflow():
    # Discriminates the hypot(P/2, sqrt(Q)) rewrite from the old
    # hypot(P, 2 sqrt(Q)): here sqrt(Q) ~ 9.2e307, so doubling it overflows to
    # inf and the old form raised "not representable" on a finite root.
    alpha = 1.1e-306
    planet_unit = np.array([alpha, 0.5 * alpha, alpha])
    solution = ccf_contrast_at_target_snr(
        planet_unit,
        np.full(3, 1.0),
        np.zeros(3),
        np.zeros(3),
        **_solver_kwargs(
            integration_time_s=1.0,
            target_snr=5.0,
            eps_systematic=0.0,
            continuum_filter_width_channels=3,
        ),
    )
    contrast = solution.contrast_at_target_snr
    assert np.isfinite(contrast)
    assert contrast == pytest.approx(9.225286e307, rel=1e-5)
    assert not np.isfinite(2.0 * contrast)  # the quantity the old form doubled


def test_published_fields_reproduce_the_photon_variance():
    # F5: template_weights and the per-channel rates are not sufficient --
    # applying a public adjoint to the raw-grid weights cannot know where the
    # islands are. With admitted_mask and noise_weights published, a consumer
    # reconstructs V_ph exactly.
    n = 24
    rng = np.random.default_rng(31)
    planet = rng.uniform(0.0, 2.0, n)
    leak = rng.uniform(20.0, 90.0, n)
    telluric = np.ones(n)
    telluric[6:11] = 0.0  # an interior gap, so there are two islands
    channels = _channels(planet, leak)
    time_s = 400.0
    budget = matched_filter_ccf(
        channels,
        integration_time_s=time_s,
        eps_systematic=0.0,
        servo_bw_hz=100.0,
        with_fringe_tracking=True,
        continuum_filter_width_channels=5,
        telluric_transmission_per_channel=telluric,
        telluric_mask_threshold=0.5,
    )
    rate = channels.total_rate_hz()
    reconstructed = time_s * float(
        np.sum(budget.noise_weights**2 * rate * budget.admitted_mask)
    )
    assert reconstructed == pytest.approx(budget.photon_var_counts, rel=1e-12)
    assert np.all(budget.template_weights[~budget.admitted_mask] == 0.0)
    assert np.all(budget.noise_weights[~budget.admitted_mask] == 0.0)
    assert int(budget.admitted_mask.sum()) == budget.n_channels_used

    # The naive raw-grid adjoint disagrees, which is why u is published.
    naive = continuum_filter_adjoint(budget.template_weights, 5)
    assert not np.allclose(naive, budget.noise_weights)
