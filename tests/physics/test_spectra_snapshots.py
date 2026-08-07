"""Integrity tests for packaged HDC spectra snapshots."""

import numpy as np
import pytest

from nullsim.physics.spectra import load_template_csv, packaged_spectrum_path

NAMES_KINDS = [
    ("earth", "albedo"),
    ("venus", "albedo"),
    ("G", "stellar_flux"),
    ("K", "stellar_flux"),
    ("M", "stellar_flux"),
    ("maunakea", "transmission"),
    ("armazones", "transmission"),
]


@pytest.mark.parametrize("name,kind", NAMES_KINDS)
def test_packaged_spectrum_loads_with_expected_kind_and_coverage(name, kind):
    spec = load_template_csv(packaged_spectrum_path(name))
    assert spec.kind == kind
    assert spec.wavelength_um[0] <= 0.5 + 1e-6
    assert spec.wavelength_um[-1] >= 1.8 - 1e-6
    assert np.all(np.isfinite(spec.values))
    for key in ("source", "generated", "resolving_power"):
        assert key in spec.provenance


@pytest.mark.parametrize("name,kind", NAMES_KINDS)
def test_packaged_spectrum_grid_is_constant_r(name, kind):
    spec = load_template_csv(packaged_spectrum_path(name))
    dln = np.diff(np.log(spec.wavelength_um))
    np.testing.assert_allclose(dln, dln[0], rtol=1e-4)
    assert 1.0 / dln[0] == pytest.approx(30_000.0, rel=0.01)


def test_stellar_spectra_have_line_structure():
    # An M dwarf at R=30k must show >5% RMS line contrast after continuum
    # removal; a lineless snapshot would silently kill the whole CCF signal.
    from scipy.ndimage import median_filter

    spec = load_template_csv(packaged_spectrum_path("M"))
    cont = median_filter(spec.values, size=501, mode="nearest")
    resid = spec.values / np.maximum(cont, 1e-300) - 1.0
    assert float(np.std(resid)) > 0.05


def test_packaged_spectrum_path_rejects_unknown_name():
    with pytest.raises(KeyError):
        packaged_spectrum_path("jupiter")
