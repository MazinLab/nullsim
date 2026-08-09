"""Integrity tests for packaged HDC spectra snapshots."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from nullsim.physics.spectra import (
    load_template_csv,
    packaged_spectrum_path,
    provenance_number,
)

NAMES_KINDS = [
    ("earth", "albedo"),
    ("venus", "albedo"),
    ("G", "stellar_flux"),
    ("K", "stellar_flux"),
    ("M", "stellar_flux"),
    ("maunakea", "transmission"),
    ("armazones", "transmission"),
    ("maunakea_radiance", "sky_radiance"),
    ("armazones_radiance", "sky_radiance"),
]

# What the snapshots claim about themselves. Content and grid density are
# separate facts and are checked against separate measurements.
LSF_RESOLVING_POWER = 30_000.0
SAMPLING_RESOLVING_POWER = 90_000.0

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("name,kind", NAMES_KINDS)
def test_packaged_spectrum_loads_with_expected_kind_and_coverage(name, kind):
    spec = load_template_csv(packaged_spectrum_path(name))
    assert spec.kind == kind
    assert spec.wavelength_um[0] <= 0.5 + 1e-6
    assert spec.wavelength_um[-1] >= 1.8 - 1e-6
    assert np.all(np.isfinite(spec.values))


@pytest.mark.parametrize("name,kind", NAMES_KINDS)
def test_packaged_spectrum_grid_matches_its_declared_sampling(name, kind):
    # The grid is the ground truth for sampling density; the header has to
    # agree with it, because that is the number a consumer would trust without
    # measuring.
    spec = load_template_csv(packaged_spectrum_path(name))
    dln = np.diff(np.log(spec.wavelength_um))
    np.testing.assert_allclose(dln, dln[0], rtol=1e-4)
    measured = 1.0 / float(np.median(dln))
    assert measured == pytest.approx(SAMPLING_RESOLVING_POWER, rel=0.01)
    declared = provenance_number(spec.provenance, "sampling_resolving_power")
    assert declared == pytest.approx(measured, rel=0.01)


@pytest.mark.parametrize("name,kind", NAMES_KINDS)
def test_packaged_spectrum_is_oversampled_against_its_own_lsf(name, kind):
    spec = load_template_csv(packaged_spectrum_path(name))
    lsf = provenance_number(spec.provenance, "lsf_resolving_power")
    sampling = provenance_number(spec.provenance, "sampling_resolving_power")
    assert lsf == pytest.approx(LSF_RESOLVING_POWER, rel=0.01)
    # Anything at or below two samples per FWHM cannot be reconstructed; three
    # is what these files are generated at.
    assert sampling / lsf >= 2.5


def test_packaged_spectrum_path_rejects_unknown_name():
    with pytest.raises(KeyError):
        packaged_spectrum_path("jupiter")


REQUIRED_PROVENANCE_KEYS = (
    "source",
    "generated",
    "kind",
    "lsf_resolving_power",
    "sampling_resolving_power",
    "generator_command",
)


@pytest.mark.parametrize("name,kind", NAMES_KINDS)
def test_packaged_spectrum_carries_every_required_provenance_key(name, kind):
    spec = load_template_csv(packaged_spectrum_path(name))
    missing = [
        key for key in REQUIRED_PROVENANCE_KEYS if f"{key}:" not in spec.provenance
    ]
    assert not missing, f"{name} is missing provenance keys {missing}"


# --------------------------------------------------------------------------
# Content resolving power, measured rather than asserted
# --------------------------------------------------------------------------


def _isolated_line_resolving_powers(
    wavelength_um: np.ndarray,
    values: np.ndarray,
    kind: str,
    min_depth: float = 0.05,
) -> np.ndarray:
    """Resolving power implied by the FWHM of every clean line in a spectrum.

    The point of this measurement is that a file's *content* resolution cannot
    be read off its header or its grid -- it has to be recovered from the
    lines. Working in ``x = ln lambda``, where a constant-R grid is uniform and
    a constant-R line-spread function has a constant width, each line's FWHM in
    x inverts directly to a resolving power.

    Lines are located as local extrema of the continuum-divided residual, their
    peak depth and centre refined by a three-point parabola, and their half
    depth crossings found by walking outwards while the profile still
    descends. A line is discarded if the walk turns back up before reaching
    half depth (it is blended), if its two half widths differ by more than a
    quarter of the total (also blended), or if the whole FWHM spans under two
    samples (nothing narrower than that is measurable, whatever the grid).

    The median over the surviving lines is the statistic that matters. It sits
    somewhat below the LSF resolving power because real lines have intrinsic
    width of their own, and it collapses -- by roughly a factor of two -- the
    moment a file is stored at one sample per FWHM.
    """
    from scipy.ndimage import median_filter

    x = np.log(wavelength_um)
    dx = float(np.median(np.diff(x)))
    continuum = median_filter(values, size=151, mode="nearest")
    scale = np.maximum(continuum, np.max(continuum) * 1e-8)
    residual = (values - continuum) / scale
    # Emission spectra carry their structure as peaks, everything else as dips.
    depth = residual if kind == "sky_radiance" else -residual

    n = depth.size
    candidates = (
        np.where(
            (depth[1:-1] > depth[:-2])
            & (depth[1:-1] >= depth[2:])
            & (depth[1:-1] > min_depth)
        )[0]
        + 1
    )

    resolving_powers: list[float] = []
    for i in candidates:
        if i < 30 or i > n - 31:
            continue
        left_y, peak_y, right_y = depth[i - 1], depth[i], depth[i + 1]
        curvature = left_y - 2.0 * peak_y + right_y
        if curvature >= 0.0:
            continue
        offset = 0.5 * (left_y - right_y) / curvature
        if abs(offset) > 0.5:
            continue
        centre = i + offset
        half = 0.5 * (peak_y - 0.25 * (left_y - right_y) * offset)

        crossings: list[float] = []
        blended = False
        for step in (-1, 1):
            j = i
            while True:
                nxt = j + step
                if nxt <= 0 or nxt >= n - 1 or depth[nxt] > depth[j]:
                    blended = True
                    break
                j = nxt
                if depth[j] <= half:
                    break
            if blended:
                break
            below, above = depth[j], depth[j - step]
            if above == below:
                blended = True
                break
            crossings.append(j - step * (half - below) / (above - below))
        if blended:
            continue

        left_half = centre - crossings[0]
        right_half = crossings[1] - centre
        fwhm_samples = left_half + right_half
        if fwhm_samples < 2.0:
            continue
        if abs(left_half - right_half) > 0.25 * fwhm_samples:
            continue
        resolving_powers.append(1.0 / (fwhm_samples * dx))

    return np.asarray(resolving_powers, dtype=np.float64)


@pytest.mark.parametrize("name,kind", NAMES_KINDS)
def test_packaged_spectrum_content_resolves_its_declared_lsf(name, kind):
    # The behavioural pin for the sampling fix. Nothing here reads the header:
    # the resolving power is measured off the line profiles in the file. When
    # these snapshots were stored at one sample per FWHM the same measurement
    # returned 12000, less than half what the header advertised.
    spec = load_template_csv(packaged_spectrum_path(name))
    measured = _isolated_line_resolving_powers(
        spec.wavelength_um, spec.values, spec.kind
    )
    assert measured.size >= 100, (
        f"{name}: only {measured.size} clean lines found, too few to measure "
        "a resolving power from"
    )
    typical = float(np.median(measured))
    # Real lines are broader than the instrument profile, never narrower, so
    # the median sits under the LSF -- between 22000 (Earth, whose water bands
    # blend heavily) and 28500 (the airglow forest, which is nearly all
    # unresolved lines and so measures the LSF almost directly). The floor is
    # set below the worst of those and well above the 12000 the same
    # measurement returned when these files held one sample per FWHM.
    assert 0.65 * LSF_RESOLVING_POWER <= typical <= 1.1 * LSF_RESOLVING_POWER, (
        f"{name}: measured content R = {typical:.0f} against a declared LSF "
        f"R = {LSF_RESOLVING_POWER:.0f}"
    )
    # And the narrow tail must stay well inside the sampling: content that
    # crowded the grid would mean the file is undersampled at the other end.
    assert float(np.quantile(measured, 0.9)) < 0.6 * SAMPLING_RESOLVING_POWER


# --------------------------------------------------------------------------
# Family-specific content checks
# --------------------------------------------------------------------------


# Minimum RMS line contrast after continuum removal, per spectral type. Hotter
# stars have shallower, sparser lines, so the floor has to track the type; the
# values below sit roughly a factor of two under what the shipped files
# actually deliver. A file that fell below these would carry too little
# structure to drive a cross-correlation.
STELLAR_LINE_RMS_FLOOR = {"G": 0.025, "K": 0.035, "M": 0.05}


@pytest.mark.parametrize("name", sorted(STELLAR_LINE_RMS_FLOOR))
def test_stellar_spectra_carry_enough_line_contrast(name):
    from scipy.ndimage import median_filter

    spec = load_template_csv(packaged_spectrum_path(name))
    cont = median_filter(spec.values, size=1501, mode="nearest")
    resid = spec.values / np.maximum(cont, 1e-300) - 1.0
    assert float(np.std(resid)) > STELLAR_LINE_RMS_FLOOR[name]


def test_packaged_spectra_are_not_duplicates_of_each_other():
    # Guards the failure mode where a generator silently falls back to a
    # default target (PSG's exoplanet mode, for instance, resolves an unknown
    # object name to an Earth template) and two families ship identical data.
    for group in (
        ("earth", "venus"),
        ("G", "K", "M"),
        ("maunakea", "armazones"),
        ("maunakea_radiance", "armazones_radiance"),
    ):
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


@pytest.mark.parametrize(
    "name,treatment", [("earth", "audit"), ("venus", "subtract")]
)
def test_albedo_declares_what_it_did_about_thermal_emission(name, treatment):
    # PSG's rif column is *apparent* I/F: reflected sunlight plus the planet's
    # own emission, over the stellar irradiance. Only the reflected part is an
    # albedo -- emission does not scale with a different host star, so carrying
    # it into another system is simply wrong. The contamination is a per cent
    # level effect that no amount of staring at the file recovers, so what is
    # pinned is that the generator measured it and said what it did: Venus is
    # subtracted sample by sample, Earth is audited and found negligible.
    spec = load_template_csv(packaged_spectrum_path(name))
    assert f"thermal_treatment: {treatment}" in spec.provenance
    peak = provenance_number(spec.provenance, "thermal_fraction_peak")
    assert peak is not None, f"{name} does not record a peak thermal fraction"
    assert 0.0 <= peak < 1.0
    if treatment == "audit":
        # An audited family may only ship apparent I/F as an albedo while its
        # emission stays negligible; past that the generator has to subtract.
        assert peak < 1.0e-3


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


@pytest.mark.parametrize("site", ["maunakea", "armazones"])
def test_sky_radiance_shares_its_request_with_the_transmission_snapshot(site):
    # The pair is only usable together if it describes one atmosphere. Airmass
    # and water vapour are the two request parameters that move the lines, so
    # the two files have to agree on them.
    transmission = load_template_csv(packaged_spectrum_path(site))
    radiance = load_template_csv(packaged_spectrum_path(f"{site}_radiance"))
    for key in ("skycalc_pwv_mm", "skycalc_airmass"):
        assert provenance_number(transmission.provenance, key) == pytest.approx(
            provenance_number(radiance.provenance, key)
        )


@pytest.mark.parametrize("site", ["maunakea", "armazones"])
def test_sky_radiance_is_an_emission_spectrum_with_an_oh_line_forest(site):
    # Airglow is what makes this product worth packaging: the near-IR sky is
    # dominated by narrow OH Meinel lines standing far above a faint
    # continuum. A transmission-shaped file, or one generated with the
    # emission switched off, would fail both halves.
    spec = load_template_csv(packaged_spectrum_path(f"{site}_radiance"))
    assert spec.kind == "sky_radiance"
    assert float(np.max(spec.values)) > 0.0
    window = (spec.wavelength_um >= 1.5) & (spec.wavelength_um <= 1.8)
    band = spec.values[window]
    assert float(np.max(band)) > 30.0 * float(np.median(band))


# --------------------------------------------------------------------------
# Provenance binds the science inputs
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["earth", "venus"])
def test_albedo_provenance_pins_the_frozen_psg_config_by_recomputed_digest(name):
    # A recorded hash nobody recomputes is decoration. Read the committed
    # config the header names, hash its bytes here, and require the match --
    # so editing the config without regenerating the spectrum is a test
    # failure rather than a silent divergence.
    spec = load_template_csv(packaged_spectrum_path(name))
    relative = None
    recorded = None
    for entry in spec.provenance.split("; "):
        key, _, value = entry.partition(": ")
        if key == "psg_config":
            relative = value.strip()
        elif key == "psg_config_sha256":
            recorded = value.strip()
    assert relative is not None, f"{name} does not name its PSG config"
    assert recorded is not None, f"{name} does not record a PSG config digest"

    config_path = REPO_ROOT / relative
    assert config_path.is_file(), f"{name} names a missing config {relative}"
    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    assert digest == recorded, (
        f"{relative} has changed since {name} was generated: {digest} != "
        f"{recorded}. Regenerate the snapshot or restore the config."
    )


@pytest.mark.parametrize("name", ["earth", "venus"])
def test_frozen_psg_config_pins_the_generator_block(name):
    # The whole point of freezing a config is that no part of the request
    # comes from PSG's live defaults, and PSG does not expand its own defaults
    # when it echoes a config back. GENERATOR-CONT-STELLAR is the one that
    # bites: flipped to Y it stamps the host star's absorption lines into what
    # the file calls an albedo.
    spec = load_template_csv(packaged_spectrum_path(name))
    relative = next(
        entry.partition(": ")[2].strip()
        for entry in spec.provenance.split("; ")
        if entry.startswith("psg_config: ")
    )
    text = (REPO_ROOT / relative).read_text()
    required = {
        "GENERATOR-RADUNITS": "rif",
        "GENERATOR-RESOLUTION": "30000",
        "GENERATOR-RESOLUTIONKERNEL": "Y",
        "GENERATOR-CONT-STELLAR": "N",
        "GENERATOR-TRANS-APPLY": "N",
    }
    settings = {
        line[1:].partition(">")[0]: line[1:].partition(">")[2]
        for line in text.splitlines()
        if line.startswith("<GENERATOR-")
    }
    for key, value in required.items():
        assert settings.get(key) == value, (
            f"{relative} has {key}={settings.get(key)!r}, expected {value!r}"
        )


@pytest.mark.parametrize("name", ["G", "K", "M"])
def test_stellar_provenance_records_its_phoenix_source_digests(name):
    # The PHOENIX FITS are far too large to commit, so the only way to bind
    # them is by digest: anyone who re-downloads can check they hold the same
    # bytes before claiming to have reproduced the snapshot.
    spec = load_template_csv(packaged_spectrum_path(name))
    for key in ("phoenix_flux_sha256", "phoenix_wave_sha256"):
        entry = next(
            (e for e in spec.provenance.split("; ") if e.startswith(f"{key}: ")),
            None,
        )
        assert entry is not None, f"{name} is missing {key}"
        digest = entry.rsplit(" ", 1)[-1]
        assert len(digest) == 64 and set(digest) <= set("0123456789abcdef"), (
            f"{name} {key} is not a SHA-256: {digest!r}"
        )
