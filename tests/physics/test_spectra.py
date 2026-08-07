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
