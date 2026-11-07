"""Tests for nullsim.physics.spectra (HDC spectral data layer)."""
import math

import numpy as np
import pytest

from nullsim.physics.spectra import (
    C_KMS,
    TemplateSpectrum,
    constant_r_grid,
    convolve_to_resolving_power,
    doppler_shift,
    interp_to_grid,
    load_template_csv,
    provenance_number,
    resample_flux_conserving,
)


def _flat_spectrum(kind="albedo", n=100):
    wl = np.linspace(1.0, 1.5, n)
    return TemplateSpectrum(wavelength_um=wl, values=np.ones(n), kind=kind)


def test_template_spectrum_rejects_descending_wavelengths():
    with pytest.raises(ValueError):
        TemplateSpectrum(
            wavelength_um=np.array([1.2, 1.1, 1.0]),
            values=np.ones(3),
            kind="albedo",
        )


def test_template_spectrum_rejects_transmission_above_one():
    with pytest.raises(ValueError):
        TemplateSpectrum(
            wavelength_um=np.array([1.0, 1.1]),
            values=np.array([0.5, 1.5]),
            kind="transmission",
        )


def test_template_spectrum_rejects_unknown_kind():
    with pytest.raises(ValueError):
        TemplateSpectrum(
            wavelength_um=np.array([1.0, 1.1]),
            values=np.ones(2),
            kind="emissivity",
        )


def test_load_template_csv_parses_values_and_provenance(tmp_path):
    p = tmp_path / "spec.csv"
    p.write_text(
        "# source: unit test\n# kind: albedo\nwavelength_um,value\n1.0,0.3\n1.1,0.4\n"
    )
    spec = load_template_csv(p)
    assert spec.kind == "albedo"
    assert "source: unit test" in spec.provenance
    np.testing.assert_allclose(spec.wavelength_um, [1.0, 1.1])
    np.testing.assert_allclose(spec.values, [0.3, 0.4])


def test_load_template_csv_requires_kind_somewhere(tmp_path):
    p = tmp_path / "spec.csv"
    p.write_text("wavelength_um,value\n1.0,0.3\n1.1,0.4\n")
    with pytest.raises(ValueError):
        load_template_csv(p)
    assert load_template_csv(p, kind="albedo").kind == "albedo"


def test_constant_r_grid_spacing_matches_resolving_power():
    grid = constant_r_grid(1.0, 1.5, 1000.0)
    dln = np.diff(np.log(grid))
    np.testing.assert_allclose(dln, 1.0 / 1000.0, rtol=1e-12)
    assert grid[0] == pytest.approx(1.0)
    assert grid[-1] >= 1.5


def test_interp_to_grid_clamps_beyond_template_edges():
    spec = TemplateSpectrum(
        wavelength_um=np.array([1.0, 1.5]),
        values=np.array([0.2, 0.4]),
        kind="albedo",
    )
    interpolated = interp_to_grid(spec, np.array([0.5, 1.25, 2.0]))
    np.testing.assert_allclose(interpolated, [0.2, 0.3, 0.4])


def test_resample_flux_conserving_preserves_band_integral():
    rng = np.random.default_rng(7)
    wl = np.linspace(1.0, 1.5, 5001)
    values = 1.0 + 0.5 * rng.standard_normal(5001)
    edges = np.linspace(1.05, 1.45, 17)
    binned = resample_flux_conserving(wl, values, edges)
    integral_binned = float(np.sum(binned * np.diff(edges)))
    from scipy.integrate import cumulative_trapezoid

    cum = cumulative_trapezoid(values, wl, initial=0.0)
    integral_direct = float(np.diff(np.interp([1.05, 1.45], wl, cum))[0])
    assert integral_binned == pytest.approx(integral_direct, rel=1e-10)


def test_resample_flux_conserving_rejects_out_of_range_edges():
    wl = np.linspace(1.0, 1.5, 100)
    with pytest.raises(ValueError):
        resample_flux_conserving(wl, np.ones(100), np.array([0.9, 1.2]))


def test_doppler_shift_moves_line_by_expected_fraction():
    # 30 km/s redshift = 1.0007e-4 fractional shift.
    spec = _flat_spectrum()
    shifted = doppler_shift(spec, 30.0)
    np.testing.assert_allclose(
        shifted.wavelength_um / spec.wavelength_um, 1.0 + 30.0 / C_KMS
    )
    assert doppler_shift(spec, -30.0).wavelength_um[0] < spec.wavelength_um[0]


def test_convolve_to_resolving_power_delivers_requested_width():
    r_grid, r_out = 300_000.0, 10_000.0
    grid = constant_r_grid(1.0, 1.3, r_grid)
    values = np.zeros(grid.size)
    values[grid.size // 2] = 1.0
    blurred = convolve_to_resolving_power(grid, values, r_out)
    lam0 = grid[grid.size // 2]
    # FWHM in wavelength should be lam0 / r_out.
    half = blurred.max() / 2.0
    above = grid[blurred >= half]
    fwhm = above[-1] - above[0]
    assert fwhm == pytest.approx(lam0 / r_out, rel=0.05)
    # Flux conserved (log grid ~ uniform enough locally).
    assert blurred.sum() == pytest.approx(values.sum(), rel=1e-6)


def test_convolve_rejects_non_log_uniform_grid():
    wl = np.linspace(1.0, 1.5, 1000)  # linear, not log-uniform
    with pytest.raises(ValueError):
        convolve_to_resolving_power(wl, np.ones(1000), 100.0)


def test_convolve_accounts_for_input_resolution():
    r_grid = 300_000.0
    grid = constant_r_grid(1.0, 1.3, r_grid)
    values = np.zeros(grid.size)
    values[grid.size // 2] = 1.0
    # Convolving to 10k in one hop must equal 30k then 30k->10k in two hops.
    one_hop = convolve_to_resolving_power(grid, values, 10_000.0, r_in=math.inf)
    mid = convolve_to_resolving_power(grid, values, 30_000.0)
    two_hop = convolve_to_resolving_power(grid, mid, 10_000.0, r_in=30_000.0)
    np.testing.assert_allclose(two_hop, one_hop, atol=1e-9)


def test_resample_flux_conserving_is_exact_within_a_source_cell():
    # A tent over x = [0, 1, 2]. The bin [0.5, 1.5] straddles the peak and lies
    # strictly inside two half-cells, so no bin edge lands on a source sample.
    # Truth: int_0.5^1 x dx + int_1^1.5 (2 - x) dx = 0.375 + 0.375 = 0.75.
    # Linear interpolation of the cumulative returns 0.5, a 33% deficit.
    wavelength = np.array([0.0, 1.0, 2.0]) + 1.0
    values = np.array([0.0, 1.0, 0.0])
    edges = np.array([1.5, 2.5])
    (bin_mean,) = resample_flux_conserving(wavelength, values, edges)
    assert bin_mean * (edges[1] - edges[0]) == pytest.approx(0.75, rel=1e-12)


def test_resample_flux_conserving_matches_the_analytic_integral_on_subcells():
    # Sub-cell bins on a random piecewise-linear source, against the exact
    # quadratic antiderivative evaluated independently.
    rng = np.random.default_rng(11)
    wavelength = np.linspace(1.0, 2.0, 9)
    values = rng.uniform(0.5, 4.0, wavelength.size)
    edges = np.array([1.03, 1.09, 1.4, 1.65, 1.9375])
    got = resample_flux_conserving(wavelength, values, edges)

    def exact_integral(a, b):
        total = 0.0
        fine = np.linspace(a, b, 200_001)
        total = np.trapezoid(np.interp(fine, wavelength, values), fine)
        return total

    expected = np.array(
        [
            exact_integral(a, b) / (b - a)
            for a, b in zip(edges[:-1], edges[1:], strict=True)
        ]
    )
    assert got == pytest.approx(expected, rel=1e-8)


def test_convolve_does_not_invent_flux_at_the_grid_edges():
    # Edge replication fills the pad with copies of the edge sample, so a delta
    # on the first channel comes back with 2.23x its flux. Zero padding models
    # "no flux outside the supplied grid": the delta keeps only the in-band half
    # of its kernel, and nothing is created.
    grid = constant_r_grid(1.0, 1.1, 20_000.0)
    interior = np.zeros(grid.size)
    interior[grid.size // 2] = 1.0
    assert convolve_to_resolving_power(grid, interior, 2000.0).sum() == pytest.approx(
        1.0, rel=1e-6
    )

    edge = np.zeros(grid.size)
    edge[0] = 1.0
    edge_flux = convolve_to_resolving_power(grid, edge, 2000.0).sum()
    # A delta on the first sample keeps the right half of its kernel plus half
    # of its own centre sample: 0.5 + g(0)/2 with g(0) = 1/(sigma sqrt(2 pi)).
    sigma_samples = 20_000.0 / (2000.0 * 2.0 * np.sqrt(2.0 * np.log(2.0)))
    expected = 0.5 + 0.5 / (sigma_samples * np.sqrt(2.0 * np.pi))
    assert expected == pytest.approx(0.546972, rel=1e-5)
    assert edge_flux == pytest.approx(expected, rel=1e-6)
    assert edge_flux < 1.0  # never invents flux, which is the actual bug fixed


def test_convolve_leaves_a_constant_flat_away_from_the_edges():
    # Zero padding attenuates within ~8 sigma of each end, by construction.
    # Away from that margin a constant must survive untouched.
    grid = constant_r_grid(1.0, 1.5, 20_000.0)
    flat = np.full(grid.size, 3.0)
    out = convolve_to_resolving_power(grid, flat, 2000.0)
    sigma_samples = 20_000.0 / (2000.0 * 2.0 * np.sqrt(2.0 * np.log(2.0)))
    margin = int(np.ceil(8.0 * sigma_samples)) + 1
    assert out[margin:-margin] == pytest.approx(3.0, rel=1e-9)


def test_convolve_edge_mode_keeps_a_flat_transmission_flat_to_the_edges():
    # Review 2026-11-07 row 50: a transmission continues past the grid, so a
    # flat T must survive broadening everywhere, edge samples included. Zero
    # padding gave the first sample 0.5 + g(0)/2 of its value (0.9 -> ~0.49
    # here) and read as absorption.
    grid = constant_r_grid(1.0, 1.1, 20_000.0)
    flat = np.full(grid.size, 0.9)
    out = convolve_to_resolving_power(grid, flat, 100.0, outside="edge")
    assert out == pytest.approx(0.9, rel=1e-12)
    assert convolve_to_resolving_power(grid, flat, 100.0)[0] < 0.5


def test_provenance_number_does_not_confuse_a_key_with_its_suffix():
    # The reason the two resolving powers are separate keys at all: a naive
    # substring search for "resolving_power" happily matches
    # "sampling_resolving_power" and hands back the grid density where the
    # line-spread function was meant -- three times too optimistic here.
    provenance = (
        "kind: albedo; sampling_resolving_power: 90000; "
        "lsf_resolving_power: 30000; samples_per_fwhm: 3"
    )
    assert provenance_number(provenance, "lsf_resolving_power") == 30000.0
    assert provenance_number(provenance, "sampling_resolving_power") == 90000.0
    # A bare "resolving_power" key is genuinely absent from this header and
    # must read as absent rather than borrowing one of its namesakes.
    assert provenance_number(provenance, "resolving_power") is None


def test_provenance_number_reads_a_key_at_the_start_of_the_string():
    assert provenance_number("resolving_power: 30000; kind: albedo", "resolving_power") == 30000.0


def test_provenance_number_returns_none_for_missing_or_unparsable_values():
    assert provenance_number("kind: albedo", "lsf_resolving_power") is None
    assert provenance_number("lsf_resolving_power: none", "lsf_resolving_power") is None
