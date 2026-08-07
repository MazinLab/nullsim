"""Tests for nullsim.physics.hdc (matched-filter CCF budget)."""
import numpy as np
import pytest

from nullsim.physics.hdc import (
    ChannelSpectra,
    build_channel_edges,
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
