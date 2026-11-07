from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import j0

from nullsim.physics.ruilier_cassaing import (
    DIFFUSE_SINGLE_MODE_ACCEPTANCE,
    OPTIMAL_PUPIL_RADIUS_TO_MODE_FIELD_RADIUS_RATIO,
    UNOBSTRUCTED_GEOMETRIC_COUPLING_LIMIT,
    angular_coupling_amplitude,
    angular_coupling_factor,
    coupling_efficiency,
    geometric_coupling_limit,
)


def test_lifted_kilo_nayra_strehl_half_source_value_to_subpercent() -> None:
    """KILO and NAYRA return 0.405 for Strehl 0.5 with Fresnel disabled."""
    assert coupling_efficiency(0.5, central_obscuration_ratio=0.0) == (
        pytest.approx(0.405, rel=6.0e-3)
    )


def test_geometric_coupling_limit_uses_optimal_ruilier_mode_ratio() -> None:
    assert OPTIMAL_PUPIL_RADIUS_TO_MODE_FIELD_RADIUS_RATIO == pytest.approx(
        1.1209,
        rel=1.0e-4,
    )


def test_coupling_efficiency_at_unity_strehl_returns_ruilier_limit() -> None:
    assert coupling_efficiency(strehl=1.0, central_obscuration_ratio=0.0) == (
        pytest.approx(0.8145, abs=1.0e-3)
    )



def test_coupling_efficiency_fresnel_loss_applied_when_flagged() -> None:
    """Tier 3.4: apply_fresnel_loss multiplies by (1 - 0.035) per KILO."""
    from nullsim.physics.ruilier_cassaing import (
        FRESNEL_REFLECTION_LOSS,
        UNOBSTRUCTED_GEOMETRIC_COUPLING_LIMIT,
    )
    eta = coupling_efficiency(strehl=1.0, apply_fresnel_loss=True)
    expected = UNOBSTRUCTED_GEOMETRIC_COUPLING_LIMIT * (1.0 - FRESNEL_REFLECTION_LOSS)
    assert eta == pytest.approx(expected, rel=1.0e-12)


def test_coupling_efficiency_fresnel_loss_default_off() -> None:
    """By default Fresnel loss is NOT applied (back-compat)."""
    from nullsim.physics.ruilier_cassaing import (
        UNOBSTRUCTED_GEOMETRIC_COUPLING_LIMIT,
    )
    eta = coupling_efficiency(strehl=1.0)
    assert eta == pytest.approx(UNOBSTRUCTED_GEOMETRIC_COUPLING_LIMIT, rel=1.0e-12)


def test_coupling_efficiency_zero_strehl_is_zero() -> None:
    assert coupling_efficiency(strehl=0.0) == 0.0


def test_coupling_efficiency_monotonically_increases_with_strehl() -> None:
    strehl = np.linspace(0.0, 1.0, 32)
    coupling = coupling_efficiency(strehl)

    assert np.all(np.diff(coupling) > 0.0)


def test_central_obscuration_monotonically_decreases_geometric_limit() -> None:
    central_obscuration_ratio = np.linspace(0.0, 0.3, 16)
    limits = geometric_coupling_limit(central_obscuration_ratio)

    assert np.all(np.diff(limits) < 0.0)


@pytest.mark.parametrize("bad_strehl", [np.nan, np.inf, -np.inf])
def test_coupling_efficiency_rejects_nan_or_infinite_strehl(
    bad_strehl: float,
) -> None:
    with pytest.raises(ValueError, match="strehl"):
        coupling_efficiency(bad_strehl)


@pytest.mark.parametrize("bad_strehl", [-1.0e-12, 1.0 + 1.0e-12])
def test_coupling_efficiency_rejects_strehl_outside_unit_interval(
    bad_strehl: float,
) -> None:
    with pytest.raises(ValueError, match="strehl"):
        coupling_efficiency(bad_strehl)


@pytest.mark.parametrize("bad_obscuration", [np.nan, np.inf, -0.1, 1.0])
def test_geometric_coupling_limit_rejects_invalid_obscuration(
    bad_obscuration: float,
) -> None:
    with pytest.raises(ValueError, match="central_obscuration_ratio"):
        geometric_coupling_limit(bad_obscuration)


def test_coupling_efficiency_rejects_nan_obscuration() -> None:
    with pytest.raises(ValueError, match="central_obscuration_ratio"):
        coupling_efficiency(0.5, central_obscuration_ratio=np.nan)


def test_coupling_efficiency_returns_float64_for_array_input() -> None:
    strehl = np.array([0, 0.5, 1], dtype=np.int16)
    coupling = coupling_efficiency(strehl)

    assert coupling.dtype == np.float64


def test_coupling_efficiency_preserves_array_shape() -> None:
    strehl = np.array([[0.0, 0.5], [0.75, 1.0]], dtype=np.float32)
    coupling = coupling_efficiency(strehl)

    assert coupling.shape == strehl.shape


def test_geometric_coupling_limit_preserves_obscuration_shape() -> None:
    central_obscuration_ratio = np.array([[0.0], [0.2], [0.3]])
    limits = geometric_coupling_limit(central_obscuration_ratio)

    assert limits.shape == central_obscuration_ratio.shape


_MAS_TO_RAD = np.pi / (180.0 * 3600.0e3)


def test_angular_coupling_factor_matches_exact_overlap_anchors() -> None:
    """kilo_keck_6 sub-apertures (3.14 m) at J (1.25 um). Anchors are the
    squared overlap ratio from scipy adaptive quadrature (review 2026-11-07
    Issue 4); KILO's Gaussian envelope gave 0.985, 0.872, 0.579, 0.220, and
    0.033 here."""

    separations_rad = np.array([10.0, 30.0, 60.0, 100.0, 150.0]) * _MAS_TO_RAD
    power = angular_coupling_factor(separations_rad, 3.14, 1.25e-6)

    np.testing.assert_allclose(
        power, [0.97124, 0.76590, 0.32274, 0.020506, 0.0029927], rtol=1e-4
    )


def test_angular_coupling_amplitude_first_zero_and_sign_change() -> None:
    first_zero = brentq(lambda s: angular_coupling_amplitude(s, 1.0, 1.0), 1.0, 2.0)

    assert first_zero == pytest.approx(1.4948, abs=1e-4)
    assert angular_coupling_amplitude(1.3, 1.0, 1.0) > 0.0
    assert angular_coupling_amplitude(1.7, 1.0, 1.0) < 0.0
    assert angular_coupling_factor(first_zero, 1.0, 1.0) < 1e-20


@pytest.mark.parametrize("obscuration", [0.0, 0.3])
@pytest.mark.parametrize("s", [0.0, 0.5, 5.0, 40.0, 120.0, 500.0, 2000.0])
def test_angular_coupling_amplitude_matches_adaptive_quadrature(
    obscuration: float, s: float
) -> None:
    rho = OPTIMAL_PUPIL_RADIUS_TO_MODE_FIELD_RADIUS_RATIO
    profile = lambda u: u * np.exp(-((rho * u) ** 2))  # noqa: E731
    num = quad(lambda u: profile(u) * j0(np.pi * s * u), obscuration, 1.0, limit=4000)[0]
    den = quad(profile, obscuration, 1.0)[0]

    got = angular_coupling_amplitude(s, 1.0, 1.0, obscuration)

    assert got == pytest.approx(num / den, rel=1e-10, abs=1e-14)


def test_angular_coupling_factor_broadcasts_and_scales_with_theta_d_over_lambda() -> None:
    theta = np.linspace(0.0, 2.0e-7, 7)[:, None]
    diameters = np.array([1.0, 2.0, 3.0])[None, :]

    power = angular_coupling_factor(theta, diameters, 1.0e-6)

    assert power.shape == (7, 3)
    np.testing.assert_allclose(
        power[:, 1], angular_coupling_factor(2.0 * theta[:, 0], 1.0, 1.0e-6)
    )
    assert isinstance(angular_coupling_factor(0.0, 1.0, 1.0e-6), float)
    assert angular_coupling_factor(0.0, 1.0, 1.0e-6) == pytest.approx(1.0)


def test_diffuse_acceptance_equals_integrated_point_source_coupling() -> None:
    """Reciprocity: ``A eta(0) int a(theta)^2 dOmega = eta_d lambda^2``.

    With ``s = theta D / lambda`` and ``A = pi D^2 / 4`` this reads
    ``(pi^2 / 2) eta(0) int_0^inf a(s)^2 s ds = 1 - exp(-2 rho^2)``. Past
    ``S = 60`` the oscillation-averaged asymptote
    ``a^2 ~ exp(-2 rho^2) / (N0^2 pi^4 s^3)`` gives the tail in closed form.
    """

    rho = OPTIMAL_PUPIL_RADIUS_TO_MODE_FIELD_RADIUS_RATIO
    x, w = np.polynomial.legendre.leggauss(16)
    edges = np.linspace(0.0, 60.0, 241)
    mid, half = 0.5 * (edges[1:] + edges[:-1]), 0.5 * np.diff(edges)
    s = (mid[:, None] + half[:, None] * x).ravel()
    body = np.sum((half[:, None] * w).ravel() * angular_coupling_factor(s, 1.0, 1.0) * s)
    n0 = (1.0 - np.exp(-(rho**2))) / (2.0 * rho**2)
    tail = np.exp(-2.0 * rho**2) / (n0**2 * np.pi**4 * edges[-1])

    integrated = np.pi**2 / 2.0 * UNOBSTRUCTED_GEOMETRIC_COUPLING_LIMIT * (body + tail)

    assert DIFFUSE_SINGLE_MODE_ACCEPTANCE == pytest.approx(1.0 - np.exp(-2.0 * rho**2))
    assert integrated == pytest.approx(DIFFUSE_SINGLE_MODE_ACCEPTANCE, rel=2e-5)
