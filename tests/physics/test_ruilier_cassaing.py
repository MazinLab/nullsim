from __future__ import annotations

import numpy as np
import pytest

from nullsim.physics.ruilier_cassaing import (
    OPTIMAL_PUPIL_RADIUS_TO_MODE_FIELD_RADIUS_RATIO,
    coupling_efficiency,
    geometric_coupling_limit,
)


def test_lifted_kilo_nayra_unobstructed_source_constant_to_subpercent() -> None:
    """KILO and NAYRA both use 0.81 as the rounded geometric source value."""
    assert geometric_coupling_limit(0.0) == pytest.approx(0.81, rel=6.0e-3)


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
