"""Tests for the planet-throughput-vs-separation primitive."""

from __future__ import annotations

import math

import numpy as np
import pytest

from nullsim.physics.planet_throughput import (
    compute_planet_throughput_vs_separation,
    dark_port_transmission_map,
    planet_input_field,
    stellar_input_field,
)
from nullsim.physics.uv_coverage import project_positions_to_sky_plane_m


_RAD_PER_MAS = 1.0 / (206265.0 * 1000.0)


def _identity_unitary_stack(n_modes: int, n_wl: int) -> np.ndarray:
    u = np.zeros((n_modes, n_modes, n_wl), dtype=np.complex128)
    eye = np.eye(n_modes, dtype=np.complex128)
    for k in range(n_wl):
        u[:, :, k] = eye
    return u


def test_stellar_input_field_norm() -> None:
    coupling = np.array([0.4, 0.6, 0.5])
    areas = np.array([1.0, 1.5, 2.0])
    e = stellar_input_field(coupling, areas)
    np.testing.assert_allclose(np.abs(e) ** 2, coupling * areas, rtol=1e-12)


def test_planet_input_field_on_axis_matches_stellar() -> None:
    """At separation=0 the planet field equals the on-axis stellar field."""

    positions = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 2.0]])
    coupling = np.array([0.5, 0.5, 0.5])
    areas = np.array([1.0, 1.0, 1.0])
    e_planet = planet_input_field(
        positions, coupling, areas, None, 1.25e-6, 0.0, 0.0,
    )
    e_star = stellar_input_field(coupling, areas)
    np.testing.assert_allclose(e_planet, e_star, atol=1e-15)


def test_planet_input_field_baseline_phase_is_linear_in_offset() -> None:
    """For two sub-apertures separated by 1 m along East, phase difference = 2*pi*alpha/lambda."""

    positions = np.array([[0.0, 0.0], [1.0, 0.0]])
    coupling = np.array([1.0, 1.0])
    areas = np.array([1.0, 1.0])
    wl = 1.0e-6
    sep = 50.0 * _RAD_PER_MAS  # 50 mas
    pa = math.pi / 2.0  # PA = 90 deg => alpha = sep, beta = 0
    field = planet_input_field(
        positions, coupling, areas, None, wl, sep, pa,
    )
    phase_diff = np.angle(field[1]) - np.angle(field[0])
    expected_diff = (2.0 * math.pi / wl) * (1.0 * sep)
    # Wrap to [-pi, pi].
    expected_wrapped = (expected_diff + math.pi) % (2.0 * math.pi) - math.pi
    actual_wrapped = (phase_diff + math.pi) % (2.0 * math.pi) - math.pi
    assert actual_wrapped == pytest.approx(expected_wrapped, abs=1e-10)


def test_throughput_identity_unitary_on_axis_matches_stellar() -> None:
    """U=I, sep=0, all ports 'dark' -> throughput == 1."""

    n_modes = 3
    n_wl = 2
    u_stack = _identity_unitary_stack(n_modes, n_wl)
    coupling = np.array([0.4, 0.6, 0.5])
    areas = np.array([1.0, 1.5, 2.0])
    t = compute_planet_throughput_vs_separation(
        u_stack=u_stack,
        wavelengths_m=[1.25e-6, 1.30e-6],
        positions_m=np.zeros((n_modes, 2)),
        dark_indices=np.arange(n_modes),
        coupling_per_sub=coupling,
        sub_areas_m2=areas,
        sub_diameters_m=None,
        separations_rad=[0.0],
        n_pa_samples=12,
    )
    assert t.shape == (1,)
    assert t[0] == pytest.approx(1.0, rel=1e-12)


def test_throughput_accepts_rectangular_output_stack() -> None:
    """A 2N-output analysis matrix can be used for dark-port throughput."""

    unitary = np.array(
        [[1.0, 1.0], [1.0, -1.0]],
        dtype=np.complex128,
    ) / math.sqrt(2.0)
    # Add two non-dark tracker rows above the two science rows. The primitive
    # should validate dark rows against the output-port axis, not the input
    # mode count.
    u_stack = np.vstack([np.eye(2, dtype=np.complex128), unitary])[:, :, np.newaxis]

    t = compute_planet_throughput_vs_separation(
        u_stack=u_stack,
        wavelengths_m=[1.0e-6],
        positions_m=np.array([[1.0, 0.0], [-1.0, 0.0]]),
        dark_indices=np.array([3]),
        coupling_per_sub=np.ones(2),
        sub_areas_m2=np.ones(2),
        sub_diameters_m=None,
        separations_rad=[0.0],
        n_pa_samples=4,
    )

    assert t[0] == pytest.approx(0.0, abs=1.0e-15)


def test_throughput_applies_differential_input_transport_scales() -> None:
    """Delay-line transport affects routing after post-delay normalization."""

    n_modes = 2
    u_stack = _identity_unitary_stack(n_modes, 1)
    coupling = np.ones(n_modes)
    areas = np.ones(n_modes)
    transport = np.array(
        [[[1.0 + 0.0j, 0.5 + 0.0j]]],
        dtype=np.complex128,
    )

    t = compute_planet_throughput_vs_separation(
        u_stack=u_stack,
        wavelengths_m=[1.25e-6],
        positions_m=np.zeros((n_modes, 2)),
        dark_indices=np.array([0]),
        coupling_per_sub=coupling,
        sub_areas_m2=areas,
        sub_diameters_m=None,
        separations_rad=[0.0],
        n_pa_samples=4,
        input_transport_scales=transport,
    )

    assert t[0] == pytest.approx(1.0 / (1.0 + 0.25), rel=1.0e-12)


def test_throughput_averages_input_transport_states_after_normalization() -> None:
    n_modes = 2
    u_stack = _identity_unitary_stack(n_modes, 1)
    transport = np.array(
        [
            [[1.0 + 0.0j, 1.0 + 0.0j]],
            [[2.0 + 0.0j, 1.0 + 0.0j]],
        ],
        dtype=np.complex128,
    )

    t = compute_planet_throughput_vs_separation(
        u_stack=u_stack,
        wavelengths_m=[1.25e-6],
        positions_m=np.zeros((n_modes, 2)),
        dark_indices=np.array([0]),
        coupling_per_sub=np.ones(n_modes),
        sub_areas_m2=np.ones(n_modes),
        sub_diameters_m=None,
        separations_rad=[0.0],
        n_pa_samples=4,
        input_transport_scales=transport,
    )

    assert t[0] == pytest.approx((0.5 + 0.8) / 2.0, rel=1.0e-12)


def test_throughput_identity_unitary_no_angular_no_phase() -> None:
    """U=I, no sub_diameters, no positions -> throughput is constant in separation."""

    n_modes = 4
    n_wl = 1
    u_stack = _identity_unitary_stack(n_modes, n_wl)
    coupling = np.full(n_modes, 0.5)
    areas = np.full(n_modes, 1.0)
    seps = np.array([1.0, 10.0, 100.0]) * _RAD_PER_MAS
    t = compute_planet_throughput_vs_separation(
        u_stack=u_stack,
        wavelengths_m=[1.25e-6],
        positions_m=np.zeros((n_modes, 2)),
        dark_indices=np.arange(n_modes),
        coupling_per_sub=coupling,
        sub_areas_m2=areas,
        sub_diameters_m=None,
        separations_rad=seps,
        n_pa_samples=16,
    )
    np.testing.assert_allclose(t, 1.0, rtol=1e-12)


def test_throughput_kernel_null_zero_on_axis_returns_zero() -> None:
    """A unitary that routes everything to bright on-axis sends nothing to dark."""

    n_modes = 2
    n_wl = 1
    # U sends mode 0 -> bright (port 0), mode 1 -> bright (port 0) via in-phase combine
    # (1/sqrt(2)) * [[1, 1], [1, -1]] is a 50/50 combiner: equal-amplitude inputs
    # interfere constructively into port 0, destructively into port 1.
    u = np.array(
        [[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128
    ) / math.sqrt(2.0)
    u_stack = np.zeros((n_modes, n_modes, n_wl), dtype=np.complex128)
    u_stack[:, :, 0] = u
    dark_indices = np.array([1])
    t = compute_planet_throughput_vs_separation(
        u_stack=u_stack,
        wavelengths_m=[1.0e-6],
        positions_m=np.zeros((n_modes, 2)),
        dark_indices=dark_indices,
        coupling_per_sub=np.array([1.0, 1.0]),
        sub_areas_m2=np.array([1.0, 1.0]),
        sub_diameters_m=None,
        separations_rad=[0.0],
        n_pa_samples=4,
    )
    assert t[0] == pytest.approx(0.0, abs=1e-15)


def test_throughput_two_aperture_fringe_visibility_curve() -> None:
    """Symmetric 50/50 combiner: dark-port throughput = sin^2(pi B alpha / lambda)/2.

    For two apertures at (+B/2, 0) and (-B/2, 0) with equal unit amplitude,
    the planet fringe makes the dark port (the 'difference' port) see
    intensity sin^2(pi B alpha / lambda). Normalised by stellar power (=2),
    the throughput is sin^2(...)/2 averaged over PA.

    PA-averaging dilutes the fringe: <sin^2(pi B sep cos(PA)/lambda)> over
    PA in [0, 2pi) equals 1/2 - (1/2) J_0(2 pi B sep / lambda) where J_0
    is the Bessel function. We just check that the throughput is bounded
    by 0.5 and rises monotonically until the first fringe maximum.
    """

    B = 10.0  # baseline meters
    n_modes = 2
    n_wl = 1
    u = (
        np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / math.sqrt(2.0)
    )
    u_stack = np.zeros((n_modes, n_modes, n_wl), dtype=np.complex128)
    u_stack[:, :, 0] = u
    positions = np.array([[B / 2.0, 0.0], [-B / 2.0, 0.0]])
    wl = 1.25e-6
    # Fringe spacing in mas: lambda / (B * mas_per_rad). For 10 m, J band:
    # 1.25e-6 / 10 = 1.25e-7 rad = 25.8 mas. Half-fringe at ~12.9 mas.
    fringe_mas = (wl / B) / _RAD_PER_MAS
    seps = np.linspace(0.01, fringe_mas / 2.0, 6) * _RAD_PER_MAS
    t = compute_planet_throughput_vs_separation(
        u_stack=u_stack,
        wavelengths_m=[wl],
        positions_m=positions,
        dark_indices=np.array([1]),
        coupling_per_sub=np.array([1.0, 1.0]),
        sub_areas_m2=np.array([1.0, 1.0]),
        sub_diameters_m=None,
        separations_rad=seps,
        n_pa_samples=72,
    )
    # Analytic PA average for a symmetric 50/50 combiner is
    # T(sep) = (1 - J_0(2*pi*B*sep/lambda)) / 2.
    from scipy.special import j0  # noqa: WPS433  (test-only import)
    expected = 0.5 * (1.0 - j0(2.0 * math.pi * B * seps / wl))
    np.testing.assert_allclose(t, expected, rtol=5e-4)
    # Monotonically increasing on the [0, fringe_mas/2] interval.
    assert np.all(np.diff(t) > 0), f"non-monotonic throughput: {t}"
    # First value (near origin) is near zero.
    assert t[0] < 1e-4


def test_throughput_angular_coupling_attenuates_at_large_sep() -> None:
    """With small sub-apertures the angular rolloff drives T->0 at large theta."""

    n_modes = 2
    n_wl = 1
    u = (
        np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / math.sqrt(2.0)
    )
    u_stack = np.zeros((n_modes, n_modes, n_wl), dtype=np.complex128)
    u_stack[:, :, 0] = u
    positions = np.array([[5.0, 0.0], [-5.0, 0.0]])
    wl = 1.25e-6
    sub_diameters = np.array([1.5, 1.5])  # small sub-aperture, narrow envelope
    seps = np.array([0.1, 10.0, 500.0]) * _RAD_PER_MAS
    t_with = compute_planet_throughput_vs_separation(
        u_stack=u_stack,
        wavelengths_m=[wl],
        positions_m=positions,
        dark_indices=np.array([1]),
        coupling_per_sub=np.array([1.0, 1.0]),
        sub_areas_m2=np.array([1.0, 1.0]),
        sub_diameters_m=sub_diameters,
        separations_rad=seps,
        n_pa_samples=36,
    )
    t_without = compute_planet_throughput_vs_separation(
        u_stack=u_stack,
        wavelengths_m=[wl],
        positions_m=positions,
        dark_indices=np.array([1]),
        coupling_per_sub=np.array([1.0, 1.0]),
        sub_areas_m2=np.array([1.0, 1.0]),
        sub_diameters_m=None,
        separations_rad=seps,
        n_pa_samples=36,
    )
    # With angular rolloff: 500 mas point should be << 10 mas point.
    assert t_with[2] < 0.1 * t_with[1]
    # Without rolloff: throughput is pure fringe pattern (no monotonic decay).
    # The values are typically O(0.5) at this separation range.
    assert np.all(t_without >= 0.0)
    # At the smallest separation the rolloff should be negligible.
    np.testing.assert_allclose(t_with[0], t_without[0], rtol=1e-3)


def test_dark_port_transmission_map_two_aperture_fringe() -> None:
    baseline_m = 10.0
    wavelength_m = 1.0e-6
    unitary = np.array(
        [[1.0, 1.0], [1.0, -1.0]],
        dtype=np.complex128,
    ) / math.sqrt(2.0)
    u_stack = unitary[:, :, np.newaxis]
    positions = np.array([[baseline_m / 2.0, 0.0], [-baseline_m / 2.0, 0.0]])
    l_coords = np.array([0.0, wavelength_m / (4.0 * baseline_m)])

    transmission = dark_port_transmission_map(
        u_stack=u_stack,
        wavelengths_m=[wavelength_m],
        positions_m=positions,
        dark_indices=np.array([1]),
        coupling_per_sub=np.ones(2),
        sub_areas_m2=np.ones(2),
        sub_diameters_m=None,
        l_coords_rad=l_coords,
        m_coords_rad=np.array([0.0]),
        wavelength_indices=(0,),
    )

    assert transmission.shape == (1, 2)
    np.testing.assert_allclose(transmission[0], [0.0, 0.5], atol=1.0e-14)


def test_dark_port_transmission_map_accepts_rectangular_dark_row_stack() -> None:
    unitary = np.array(
        [[1.0, 1.0], [1.0, -1.0]],
        dtype=np.complex128,
    ) / math.sqrt(2.0)
    full_stack = unitary[:, :, np.newaxis]
    dark_row_stack = unitary[1:, :, np.newaxis]

    full = dark_port_transmission_map(
        u_stack=full_stack,
        wavelengths_m=[1.0e-6],
        positions_m=np.array([[1.0, 0.0], [-1.0, 0.0]]),
        dark_indices=np.array([1]),
        coupling_per_sub=np.ones(2),
        sub_areas_m2=np.ones(2),
        sub_diameters_m=None,
        l_coords_rad=np.array([0.0, 1.0e-7]),
        m_coords_rad=np.array([0.0]),
        wavelength_indices=(0,),
    )
    rectangular = dark_port_transmission_map(
        u_stack=dark_row_stack,
        wavelengths_m=[1.0e-6],
        positions_m=np.array([[1.0, 0.0], [-1.0, 0.0]]),
        dark_indices=np.array([0]),
        coupling_per_sub=np.ones(2),
        sub_areas_m2=np.ones(2),
        sub_diameters_m=None,
        l_coords_rad=np.array([0.0, 1.0e-7]),
        m_coords_rad=np.array([0.0]),
        wavelength_indices=(0,),
    )

    np.testing.assert_allclose(rectangular, full, atol=1.0e-14)


def test_dark_port_transmission_map_pa_average_matches_throughput_primitive() -> None:
    baseline_m = 8.0
    wavelength_m = 1.25e-6
    unitary = np.array(
        [[1.0, 1.0], [1.0, -1.0]],
        dtype=np.complex128,
    ) / math.sqrt(2.0)
    u_stack = unitary[:, :, np.newaxis]
    positions = np.array([[baseline_m / 2.0, 0.0], [-baseline_m / 2.0, 0.0]])
    separation_rad = 20.0 * _RAD_PER_MAS
    n_pa = 72
    pa_angles = np.linspace(0.0, 2.0 * math.pi, n_pa, endpoint=False)
    map_values = []
    for pa in pa_angles:
        l_rad = separation_rad * math.sin(float(pa))
        m_rad = separation_rad * math.cos(float(pa))
        value = dark_port_transmission_map(
            u_stack=u_stack,
            wavelengths_m=[wavelength_m],
            positions_m=positions,
            dark_indices=np.array([1]),
            coupling_per_sub=np.ones(2),
            sub_areas_m2=np.ones(2),
            sub_diameters_m=None,
            l_coords_rad=np.array([l_rad]),
            m_coords_rad=np.array([m_rad]),
            wavelength_indices=(0,),
        )[0, 0]
        map_values.append(value)
    throughput = compute_planet_throughput_vs_separation(
        u_stack=u_stack,
        wavelengths_m=[wavelength_m],
        positions_m=positions,
        dark_indices=np.array([1]),
        coupling_per_sub=np.ones(2),
        sub_areas_m2=np.ones(2),
        sub_diameters_m=None,
        separations_rad=[separation_rad],
        n_pa_samples=n_pa,
    )

    assert float(np.mean(map_values)) == pytest.approx(throughput[0], rel=1.0e-12)


def test_dark_port_transmission_map_angular_coupling_preserves_on_axis_only() -> None:
    baseline_m = 6.0
    wavelength_m = 1.25e-6
    unitary = np.array(
        [[1.0, 1.0], [1.0, -1.0]],
        dtype=np.complex128,
    ) / math.sqrt(2.0)
    u_stack = unitary[:, :, np.newaxis]
    positions = np.array([[baseline_m / 2.0, 0.0], [-baseline_m / 2.0, 0.0]])
    l_coords = np.array([0.0, 500.0 * _RAD_PER_MAS])
    common = {
        "u_stack": u_stack,
        "wavelengths_m": [wavelength_m],
        "positions_m": positions,
        "dark_indices": np.array([1]),
        "coupling_per_sub": np.ones(2),
        "sub_areas_m2": np.ones(2),
        "l_coords_rad": l_coords,
        "m_coords_rad": np.array([0.0]),
        "wavelength_indices": (0,),
    }

    without_rolloff = dark_port_transmission_map(
        **common,
        sub_diameters_m=None,
    )
    with_rolloff = dark_port_transmission_map(
        **common,
        sub_diameters_m=np.array([1.5, 1.5]),
    )

    assert with_rolloff[0, 0] == pytest.approx(without_rolloff[0, 0], abs=1.0e-15)
    assert with_rolloff[0, 1] < without_rolloff[0, 1]


def test_sky_projection_foreshortens_along_pointing_azimuth() -> None:
    # altitude 60 deg, azimuth 0 (pointing due N at 60 deg el).
    # A N-S baseline foreshortens by sin(60deg); an E-W baseline is unchanged.
    pos = np.array([[0.0, 50.0, 0.0], [50.0, 0.0, 0.0]])  # N-S and E-W arms
    proj = project_positions_to_sky_plane_m(
        positions_enu_m=pos, altitude_deg=60.0, azimuth_deg=0.0
    )
    assert proj[0, 1] == pytest.approx(50.0 * math.sin(math.radians(60.0)), rel=1e-9)
    assert proj[1, 0] == pytest.approx(50.0, rel=1e-9)


def test_sky_projection_identity_at_zenith() -> None:
    pos = np.array([[3.0, -7.0, 0.0], [11.0, 5.0, 0.0]])
    proj = project_positions_to_sky_plane_m(
        positions_enu_m=pos, altitude_deg=90.0, azimuth_deg=0.0
    )
    # At zenith, az axis -> East, alt axis -> North: identity on (E, N).
    np.testing.assert_allclose(proj[:, 0], pos[:, 0], atol=1e-12)
    np.testing.assert_allclose(proj[:, 1], pos[:, 1], atol=1e-12)


def test_sky_projection_accepts_2d_positions() -> None:
    pos2d = np.array([[0.0, 50.0], [50.0, 0.0]])
    proj = project_positions_to_sky_plane_m(
        positions_enu_m=pos2d, altitude_deg=60.0, azimuth_deg=0.0
    )
    assert proj.shape == (2, 2)
    assert proj[0, 1] == pytest.approx(50.0 * math.sin(math.radians(60.0)), rel=1e-9)


def test_sky_projection_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        project_positions_to_sky_plane_m(
            positions_enu_m=np.zeros((3, 4)), altitude_deg=60.0, azimuth_deg=0.0
        )
