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


def test_packaged_spectrum_path_rejects_unknown_name():
    with pytest.raises(KeyError):
        packaged_spectrum_path("jupiter")


REQUIRED_PROVENANCE_KEYS = (
    "source",
    "generated",
    "kind",
    "resolving_power",
    "generator_command",
)


@pytest.mark.parametrize("name,kind", NAMES_KINDS)
def test_packaged_spectrum_carries_every_required_provenance_key(name, kind):
    spec = load_template_csv(packaged_spectrum_path(name))
    missing = [
        key for key in REQUIRED_PROVENANCE_KEYS if f"{key}:" not in spec.provenance
    ]
    assert not missing, f"{name} is missing provenance keys {missing}"


# Minimum RMS line contrast after continuum removal, per spectral type. Hotter
# stars have shallower, sparser lines, so the floor has to track the type; the
# values below sit roughly a factor of two under what the shipped files
# actually deliver (G 0.060, K 0.083, M 0.111 as generated). A file that fell
# below these would carry too little structure to drive a cross-correlation.
STELLAR_LINE_RMS_FLOOR = {"G": 0.025, "K": 0.035, "M": 0.05}


@pytest.mark.parametrize("name", sorted(STELLAR_LINE_RMS_FLOOR))
def test_stellar_spectra_carry_enough_line_contrast(name):
    from scipy.ndimage import median_filter

    spec = load_template_csv(packaged_spectrum_path(name))
    cont = median_filter(spec.values, size=501, mode="nearest")
    resid = spec.values / np.maximum(cont, 1e-300) - 1.0
    assert float(np.std(resid)) > STELLAR_LINE_RMS_FLOOR[name]


def test_packaged_spectra_are_not_duplicates_of_each_other():
    # Guards the failure mode where a generator silently falls back to a
    # default target (PSG's exoplanet mode, for instance, resolves an unknown
    # object name to an Earth template) and two families ship identical data.
    for group in (("earth", "venus"), ("G", "K", "M"), ("maunakea", "armazones")):
        loaded = {n: load_template_csv(packaged_spectrum_path(n)).values for n in group}
        for i, first in enumerate(group):
            for second in group[i + 1 :]:
                assert not np.allclose(
                    loaded[first], loaded[second], rtol=1e-3, atol=0.0
                ), f"{first} and {second} carry effectively identical values"


def test_earth_albedo_shows_the_o2_a_band_and_venus_does_not():
    # The O2 A band at 0.76 um is the single most diagnostic difference between
    # the two albedo templates; if the Venus file were an Earth fallback it
    # would show the band too.
    def band_ratio(name):
        spec = load_template_csv(packaged_spectrum_path(name))
        w, v = spec.wavelength_um, spec.values
        core = v[(w >= 0.760) & (w <= 0.765)].mean()
        cont = v[(w >= 0.740) & (w <= 0.750)].mean()
        return core / cont

    assert band_ratio("earth") < 0.6
    assert band_ratio("venus") > 0.9


@pytest.mark.parametrize("name,pwv", [("maunakea", "2.5"), ("armazones", "5.0")])
def test_telluric_provenance_records_its_precipitable_water_vapour(name, pwv):
    spec = load_template_csv(packaged_spectrum_path(name))
    assert f"skycalc_pwv_mm: {pwv}" in spec.provenance


def test_wetter_site_has_deeper_water_band():
    # Armazones is generated at 5.0 mm PWV against Maunakea's 2.5 mm, so the
    # 1.4 um water band must be deeper there. This binds each file to its own
    # request rather than merely to the SkyCalc model.
    def water_band(name):
        spec = load_template_csv(packaged_spectrum_path(name))
        w, v = spec.wavelength_um, spec.values
        return v[(w >= 1.38) & (w <= 1.42)].mean()

    assert water_band("armazones") < water_band("maunakea")


@pytest.mark.parametrize("name", ["earth", "venus"])
def test_albedo_provenance_pins_the_frozen_psg_config(name):
    # The spectra must be reproducible from a committed config, not from
    # whatever defaults the live PSG service happens to hold.
    spec = load_template_csv(packaged_spectrum_path(name))
    assert "psg_config_sha256:" in spec.provenance
    assert "psg_config: scripts/psg_config/" in spec.provenance
