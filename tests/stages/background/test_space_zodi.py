"""Tests for the space zodiacal-background stage."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

from nullsim.physics.photometry import photon_rate_per_m2_per_bin
from nullsim.physics.ruilier_cassaing import angular_coupling_factor
from nullsim.pipeline import (
    Field,
    Geometry,
    RunContext,
    Scene,
    SimulationState,
    Throughput,
    Wavefront,
)
from nullsim.stages.background import SpaceZodiBackground
from nullsim.stages.injection.single_mode_fiber import SingleModeFiber

_WAVELENGTHS_UM = (0.78, 0.82)
_ARCSEC_PER_RAD = 206_264.80624709636
# Fraction of a Gaussian mode (aperture radius / mode radius = 1.1209) that a
# circular aperture passes: the diffuse single-mode acceptance.
_ETA_DIFFUSE = 1.0 - math.exp(-2.0 * 1.120906422778534**2)
_HADAMARD = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / math.sqrt(2.0)


def _context(tmp_path: Path) -> RunContext:
    return RunContext(
        rng=np.random.default_rng(0),
        config_hash="background-space-zodi",
        sweep_coord_tuple=(),
        instance_id="zodi",
        run_dir=tmp_path,
    )


def _chip(unitary: np.ndarray, n_bright: int, n_dark: int) -> dict:
    return {
        "u_stack": np.repeat(unitary[:, :, np.newaxis], len(_WAVELENGTHS_UM), axis=2),
        "n_bright_ports": n_bright,
        "n_dark_ports": n_dark,
    }


def _state(distance_pc: float = 5.0) -> SimulationState:
    """Two 1.12 m modes behind an 8 m primary, nulled by a Hadamard chip."""

    state = SimulationState.empty({"wavelength_grid_um": _WAVELENGTHS_UM})
    return replace(
        state,
        scene=Scene(
            star_jmag=8.0,
            star_spectral_type="M3V",
            star_distance_pc=distance_pc,
            planet_separation_mas=100.0,
            exozodi_level_zodi=2.0,
        ),
        field=Field(
            amplitudes=np.ones((2, 2, 1), dtype=np.complex128),
            wavelength_grid_um=_WAVELENGTHS_UM,
        ),
        geometry=Geometry(
            diameter_m=8.0,
            area_m2=30.0,
            collecting_area_m2=30.0,
            effective_collecting_area_m2=20.0,
            n_apertures=1,
            aperture_positions_pupil_m=np.array([[-0.6, 0.0], [0.6, 0.0]]),
            # Oldest component is the primary; the chip-input modes are newer.
            components={
                "telescope": {"mode_diameter_m": 8.0},
                "pupil_modes": {"mode_diameter_m": 1.12},
            },
        ),
        throughput=Throughput(
            wavelength_grid_um=_WAVELENGTHS_UM,
            transmission_fraction=(0.5, 0.25),
        ),
        results=(
            state.results.add_component(
                "chip.kernel_mzi_mesh", _chip(_HADAMARD, n_bright=1, n_dark=1)
            ).add_component(
                "detector.mkid",
                {"port_labels": ("bright_0", "dark_0")},
            )
        ),
    )


def _apply(state: SimulationState, params: dict, tmp_path: Path) -> dict:
    result = SpaceZodiBackground(params, _context(tmp_path)).apply(state)
    return result.results.components["background.space_zodi"]


def _rate(mag: float) -> np.ndarray:
    return np.array(
        [photon_rate_per_m2_per_bin(mag, wl, 0.04, band="I") for wl in _WAVELENGTHS_UM]
    )


def _etendue_m2_arcsec2() -> np.ndarray:
    wavelength_m = np.asarray(_WAVELENGTHS_UM) * 1.0e-6
    return _ETA_DIFFUSE * wavelength_m**2 * _ARCSEC_PER_RAD**2


def test_local_zodi_uses_single_mode_etendue_per_bin(tmp_path: Path) -> None:
    """Each input collects B lambda^2 eta_d per bin, independent of mode area;
    a unitary dark row (row norm 1) receives one input's worth (Issue 2a, 2b)."""

    payload = _apply(
        _state(),
        {"band": "I", "local_zodi_level": 1.5, "local_zodi_v_mag_arcsec2": 23.0,
         "exozodi_level_zodi": 0.0},
        tmp_path,
    )

    expected = 1.5 * _rate(23.0 - 0.69) * _etendue_m2_arcsec2() * np.array([0.5, 0.25])
    np.testing.assert_allclose(payload["local_zodi_rate_hz_per_bin"], expected, rtol=1e-12)
    assert payload["local_zodi_rate_hz_per_dark_port"] == pytest.approx(expected.sum())
    assert payload["routing_throughput"] == pytest.approx(1.0)
    assert payload["local_zodi_rate_hz_per_m2_per_arcsec2"] == pytest.approx(
        1.5 * _rate(23.0 - 0.69).sum()
    )

    larger_area = replace(
        _state(), geometry=replace(_state().geometry, effective_collecting_area_m2=80.0)
    )
    resized = _apply(
        larger_area,
        {"band": "I", "local_zodi_level": 1.5, "local_zodi_v_mag_arcsec2": 23.0,
         "exozodi_level_zodi": 0.0},
        tmp_path,
    )
    assert resized["local_zodi_rate_hz_per_dark_port"] == pytest.approx(
        payload["local_zodi_rate_hz_per_dark_port"]
    )


def test_zodi_is_not_debited_by_point_source_fiber_coupling(tmp_path: Path) -> None:
    """The Ruilier-Cassaing eta (x Strehl) is a point-source factor; neither the
    SMF stage's component nor a second lookup may reduce diffuse zodi
    (Issue 2d, row 42)."""

    base = _state()
    with_wavefront = replace(
        base,
        wavefront=Wavefront(wavelength_grid_um=_WAVELENGTHS_UM, strehl_ratio=(0.6, 0.7)),
    )
    injected = SingleModeFiber({}, _context(tmp_path)).apply(with_wavefront)
    params = {"band": "I", "exozodi_level_zodi": 0.0}

    assert _apply(injected, params, tmp_path)[
        "local_zodi_rate_hz_per_dark_port"
    ] == pytest.approx(_apply(base, params, tmp_path)["local_zodi_rate_hz_per_dark_port"])


def test_exozodi_brightness_uses_host_absolute_magnitude(tmp_path: Path) -> None:
    """SB(1 AU) = 22 - (V-I)_sun + M_host - M_sun,I with M = m - 5 log10(d/10)
    (Issue 2c). The apparent-magnitude version is 5 log10(10/d) off."""

    host_i_mag = 9.4  # J = 8, M3V
    for distance_pc in (5.0, 20.0):
        payload = _apply(
            _state(distance_pc),
            {"band": "I", "local_zodi_level": 0.0, "exozodi_level_zodi": 1.0},
            tmp_path,
        )
        absolute_i = host_i_mag - 5.0 * math.log10(distance_pc / 10.0)
        expected_mag = (22.0 - 0.69) + absolute_i - (4.83 - 0.69)
        assert payload["host_star_mag_for_exozodi"] == pytest.approx(host_i_mag)
        assert payload["host_star_abs_mag_for_exozodi"] == pytest.approx(absolute_i)
        assert payload["exozodi_mag_arcsec2"] == pytest.approx(expected_mag)

        # r^-2 profile evaluated at the scene planet (100 mas x d pc in AU).
        radius_au = 0.1 * distance_pc
        assert payload[
            "exozodi_rate_hz_per_m2_per_arcsec2_at_scene_separation"
        ] == pytest.approx(_rate(expected_mag).sum() / radius_au**2)


def test_exozodi_weight_matches_quadrature_over_mode_envelope(tmp_path: Path) -> None:
    """One mode feeding one dark port: T(theta) is the angular coupling envelope
    of the 1.12 m mode (not the 8 m primary), so the weight is
    int p T theta dtheta / int T theta dtheta with p = (r/1 AU)^-2 outside the
    sublimation radius. Integrated here with adaptive quadrature."""

    distance_pc = 5.0
    base = _state(distance_pc)
    state = replace(
        base,
        field=Field(
            amplitudes=np.ones((1, 2, 1), dtype=np.complex128),
            wavelength_grid_um=_WAVELENGTHS_UM,
        ),
        geometry=replace(base.geometry, aperture_positions_pupil_m=np.zeros((1, 2))),
        results=replace(
            base.results,
            components=base.results.components
            | {"chip.kernel_mzi_mesh": _chip(np.eye(1, dtype=np.complex128), 0, 1)},
        ),
    )
    payload = _apply(
        state, {"band": "I", "local_zodi_level": 0.0, "exozodi_level_zodi": 1.0},
        tmp_path,
    )

    inner_au = payload["exozodi_inner_radius_au"]
    # Blackbody sublimation radius (278.3 K / 1500 K)^2 sqrt(L), L from M_I.
    luminosity = 10.0 ** (-0.4 * (payload["host_star_abs_mag_for_exozodi"] - 4.14))
    assert inner_au == pytest.approx((278.3 / 1500.0) ** 2 * math.sqrt(luminosity))
    theta_inner = inner_au / distance_pc / _ARCSEC_PER_RAD

    def envelope(theta: float) -> float:
        return float(np.mean(
            [angular_coupling_factor(theta, 1.12, wl * 1e-6) for wl in _WAVELENGTHS_UM]
        ))

    def radius_au(theta: float) -> float:
        return theta * _ARCSEC_PER_RAD * distance_pc

    upper = 20.0 * 0.82e-6 / 1.12
    norm = quad(lambda t: envelope(t) * t, 0.0, upper, limit=200)[0]
    # The 1/theta integrand spans decades above the sublimation edge; plain
    # quad misses 20% of it without breakpoints.
    numerator = quad(
        lambda t: envelope(t) * t / radius_au(t) ** 2,
        theta_inner,
        upper,
        limit=500,
        points=np.geomspace(theta_inner, upper, 20)[1:-1],
    )[0]
    assert payload["exozodi_profile_weight_dark"] == pytest.approx(
        numerator / norm, rel=2e-3
    )

    expected = (
        _rate(payload["exozodi_mag_arcsec2"])
        * payload["exozodi_profile_weight_dark"]
        * _etendue_m2_arcsec2()
        * np.array([0.5, 0.25])
    )
    np.testing.assert_allclose(payload["exozodi_rate_hz_per_bin"], expected, rtol=1e-12)


def test_space_zodi_can_sample_hosts_level_and_inclined_morphology(
    tmp_path: Path,
) -> None:
    params = {"band": "I", "local_zodi_level": 0.0}
    sampled = _apply(
        _state(),
        params | {
            "exozodi_level_distribution": "hosts2020_lognormal",
            "exozodi_hosts2020_median_zodi": 3.0,
            "exozodi_hosts2020_sigma_dex": 0.0,
            "exozodi_morphology": "smooth_inclined_disk",
            "exozodi_inclination_deg": 60.0,
        },
        tmp_path,
    )
    one_zodi = _apply(_state(), params | {"exozodi_level_zodi": 1.0}, tmp_path)

    assert sampled["exozodi_level_source"] == "hosts2020_lognormal"
    assert sampled["exozodi_level_zodi"] == pytest.approx(3.0)
    assert sampled["exozodi_inclination_deg"] == pytest.approx(60.0)
    assert sampled["exozodi_surface_brightness_multiplier"] == pytest.approx(2.0)
    assert sampled["exozodi_effective_level_zodi"] == pytest.approx(6.0)
    np.testing.assert_allclose(
        sampled["exozodi_rate_hz_per_bin"], 6.0 * one_zodi["exozodi_rate_hz_per_bin"]
    )


def test_space_zodi_publishes_constant_background_on_sensitivity_grid(
    tmp_path: Path,
) -> None:
    """Diffuse zodi does not depend on where the planet sits (Issue 2a)."""

    state = _state()
    state = replace(
        state,
        results=state.results.add_component(
            "sensitivity.detection_curve",
            {"separations_mas": (50.0, 100.0), "planet_throughputs": (1e-30, 0.5)},
        ),
    )
    payload = _apply(state, {"band": "I"}, tmp_path)

    np.testing.assert_allclose(payload["background_separations_mas"], [50.0, 100.0])
    np.testing.assert_allclose(
        payload["background_rate_hz_per_dark_port_per_sep"],
        [payload["background_rate_hz_per_dark_port"]] * 2,
    )
    assert payload["background_rate_hz_per_dark_port"] > 0.0


def test_space_zodi_can_use_absolute_exozodi_surface_brightness(
    tmp_path: Path,
) -> None:
    payload = _apply(
        _state(),
        {"band": "I", "local_zodi_level": 0.0, "exozodi_level_zodi": 1.0,
         "derive_exozodi_mag_from_scene_jmag": False},
        tmp_path,
    )
    assert payload["host_star_mag_for_exozodi"] is None
    assert payload["exozodi_mag_arcsec2"] == pytest.approx(22.0 - 0.69)


def test_space_zodi_splits_etendue_between_pbs_arms(tmp_path: Path) -> None:
    """Each PBS arm's dark port gets half of the input's B lambda^2 eta_d."""

    state = _state()
    dual = replace(
        state,
        field=replace(
            state.field,
            amplitudes=np.ones((2, 2, 2), dtype=np.complex128),
            components={"polarization.pbs_split": {"polarization_labels": ("A", "B")}},
        ),
    )
    params = {"band": "I", "exozodi_level_zodi": 0.0}

    assert _apply(dual, params, tmp_path)[
        "local_zodi_rate_hz_per_dark_port"
    ] == pytest.approx(0.5 * _apply(state, params, tmp_path)["local_zodi_rate_hz_per_dark_port"])
