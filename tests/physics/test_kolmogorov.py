"""Tests for nullsim.physics.kolmogorov.

Pinning tests for the canonical Kolmogorov / Noll normalization constants,
gap-coverage tests for r0 wavelength scaling, asymptotic limits, edge cases,
and dtype invariance. See tests/physics/notes/kolmogorov.md for the
KILO/NAYRA upstream survey and architect-spec divergence rationale.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nullsim.physics.kolmogorov import (
    KOLMOGOROV_PHASE_SF_COEFFICIENT,
    NOLL_PISTON_REMOVED_COEFFICIENT,
    NOLL_TIP_TILT_REMOVED_COEFFICIENT,
    R0_REFERENCE_WAVELENGTH_M,
    TAU0_WIND_COEFFICIENT,
    effective_wind_speed_m_s,
    opd_rms_m,
    phase_structure_function,
    phase_variance,
    r0_scaled,
    strehl_marechal,
)


# ── Canonical-constant pinning ───────────────────────────────────────────


def test_phase_structure_function_coefficient_matches_kilo_c_temp() -> None:
    """6.88 is both Fried (1965) D_phi coefficient and KILO's C_TEMP."""
    assert KOLMOGOROV_PHASE_SF_COEFFICIENT == pytest.approx(6.88, rel=1e-12)


def test_noll_piston_removed_coefficient_is_delta_1() -> None:
    """Delta_1 = 1.0299 (D/r0)^(5/3): Noll 1976, residual after piston."""
    assert NOLL_PISTON_REMOVED_COEFFICIENT == pytest.approx(1.0299, rel=1e-12)


def test_noll_tip_tilt_removed_coefficient_is_delta_3() -> None:
    """Delta_3 = 0.134 (D/r0)^(5/3): Noll 1976, residual after piston and tilt."""
    assert NOLL_TIP_TILT_REMOVED_COEFFICIENT == pytest.approx(0.134, rel=1e-12)


def _kolmogorov_residual_coefficient(*, remove_tilt: bool) -> float:
    """Residual variance / (D/r0)^(5/3) from the Kolmogorov phase spectrum.

    sigma^2 = int Phi(f) [1 - |Q(f)|^2] 2 pi f df with
    Phi(f) = c r0^(-5/3) f^(-11/3), c = Gamma(11/6)^2 / (2 pi^(11/3))
    (24/5 Gamma(6/5))^(5/6), and the angle-averaged Zernike filters of
    Noll (1976) Eq. 8 with x = pi D f: piston (2 J1(x)/x)^2, tip plus tilt
    16 J2(x)^2 / x^2. Below x0 the piston-removed integrand is replaced by
    its leading term x^(-2/3)/4 to avoid roundoff in 1 - (2 J1/x)^2.
    """
    from scipy.integrate import quad
    from scipy.special import gamma, j1, jv

    c = gamma(11 / 6) ** 2 / (2 * math.pi ** (11 / 3)) * (24 / 5 * gamma(6 / 5)) ** (5 / 6)

    def integrand(x: float) -> float:
        filtered = 1.0 - (2.0 * j1(x) / x) ** 2
        if remove_tilt:
            filtered -= 16.0 * jv(2, x) ** 2 / x**2
        return x ** (-8.0 / 3.0) * filtered

    x0 = 1.0e-2
    head = 0.0 if remove_tilt else 0.75 * x0 ** (1.0 / 3.0)
    body = quad(integrand, x0, 60.0, limit=400)[0]
    tail = quad(integrand, 60.0, np.inf, limit=400)[0]
    return 2.0 * math.pi * c * math.pi ** (5.0 / 3.0) * (head + body + tail)


@pytest.mark.parametrize(
    ("remove_tilt", "coefficient"),
    [(False, NOLL_PISTON_REMOVED_COEFFICIENT), (True, NOLL_TIP_TILT_REMOVED_COEFFICIENT)],
)
def test_noll_coefficients_match_kolmogorov_spectrum_quadrature(
    remove_tilt: bool, coefficient: float,
) -> None:
    """Pins the piston/tip-tilt labels against an independent integral.

    The quadrature gives 1.0324 and 0.1347; Noll's tabulated 1.0299 and
    0.134 sit 0.24% and 0.5% below. A swap of labels (or the former 0.582,
    which is Delta_2, piston plus one tilt) fails by tens of percent.
    """
    assert _kolmogorov_residual_coefficient(remove_tilt=remove_tilt) == pytest.approx(
        coefficient, rel=6e-3,
    )


def test_r0_reference_wavelength_matches_kilo_convention() -> None:
    """KILO scale_r0 anchors r0 at 500 nm."""
    assert R0_REFERENCE_WAVELENGTH_M == pytest.approx(500.0e-9, rel=1e-12)


def test_effective_wind_speed_from_tau0_matches_greenwood() -> None:
    # tau0 = 0.314 r0 / v_bar (Greenwood 1977; Hardy 1998 eq. 9.21)
    # r0 = 0.15 m, tau0 = 2.83 ms -> v = 0.314*0.15/2.83e-3 = 16.643 m/s.
    assert TAU0_WIND_COEFFICIENT == pytest.approx(0.314, rel=1e-12)
    assert effective_wind_speed_m_s(
        r0_500nm_m=0.15,
        tau0_500nm_s=2.83e-3,
    ) == pytest.approx(16.643, rel=1.0e-3)


# ── r0_scaled: Kolmogorov 6/5 power law ──────────────────────────────────


def test_r0_scaled_identity_at_reference_wavelength() -> None:
    assert r0_scaled(0.15, 500.0e-9) == pytest.approx(0.15, rel=1e-12)


def test_r0_scaled_matches_kilo_scale_r0_at_h_band() -> None:
    """Exact KILO `scale_r0` formula reproduction at H-band."""
    r0_500 = 0.15  # m, typical median Maunakea seeing
    expected_r0_h = r0_500 * (1.65e-6 / 500e-9) ** 1.2
    assert r0_scaled(r0_500, 1.65e-6) == pytest.approx(expected_r0_h, rel=1e-12)


def test_r0_scaled_via_explicit_reference_wavelength() -> None:
    """Passing a non-default reference wavelength produces a 1:1 result."""
    r0_h = 0.6
    # Round-trip: scale H -> J using H as reference, then back to H
    r0_j = r0_scaled(r0_h, 1.235e-6, wavelength_ref_m=1.65e-6)
    r0_back = r0_scaled(r0_j, 1.65e-6, wavelength_ref_m=1.235e-6)
    assert r0_back == pytest.approx(r0_h, rel=1e-12)


def test_r0_scaled_array_input_dtype_invariance() -> None:
    wavelengths_m = np.array([1.0e-6, 1.5e-6, 2.0e-6], dtype=np.float32)
    out = r0_scaled(np.float32(0.15), wavelengths_m)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    assert out.shape == (3,)


# ── phase_structure_function ─────────────────────────────────────────────


def test_phase_structure_function_at_r0_returns_coefficient() -> None:
    """D_phi(r0) = 6.88 rad^2 by construction."""
    r0 = 0.5
    assert phase_structure_function(r0, r0) == pytest.approx(
        KOLMOGOROV_PHASE_SF_COEFFICIENT, rel=1e-12,
    )


def test_phase_structure_function_zero_separation_is_zero() -> None:
    """D_phi(0) = 0 (no decorrelation at zero separation)."""
    assert phase_structure_function(0.0, 0.5) == pytest.approx(0.0, abs=1e-12)


def test_phase_structure_function_five_thirds_power_law() -> None:
    """Doubling r at fixed r0 scales D_phi by 2^(5/3)."""
    r0 = 0.5
    d_small = phase_structure_function(0.2, r0)
    d_large = phase_structure_function(0.4, r0)
    assert d_large / d_small == pytest.approx(2.0 ** (5.0 / 3.0), rel=1e-12)


def test_phase_structure_function_array_broadcasting() -> None:
    r0 = 0.5
    separations_m = np.array([0.1, 0.2, 0.5, 1.0])
    out = phase_structure_function(separations_m, r0)
    expected = KOLMOGOROV_PHASE_SF_COEFFICIENT * (separations_m / r0) ** (5.0 / 3.0)
    np.testing.assert_allclose(out, expected, rtol=1e-12)


# ── phase_variance: Noll 1976 ───────────────────────────────────────────


def test_phase_variance_default_is_piston_removed_delta_1() -> None:
    """sigma^2 = 1.0299 (D/r0)^(5/3) at D = r0 (Noll Delta_1)."""
    r0 = 0.15
    assert phase_variance(r0, r0) == pytest.approx(1.0299, rel=1e-12)


def test_phase_variance_tip_tilt_removed_is_delta_3() -> None:
    """sigma^2 = 0.134 (D/r0)^(5/3) at D = r0 with tip/tilt also removed."""
    r0 = 0.15
    assert phase_variance(r0, r0, tip_tilt_removed=True) == pytest.approx(
        0.134, rel=1e-12,
    )


def test_phase_variance_scales_as_d_over_r0_to_the_five_thirds() -> None:
    r0 = 0.15
    sigma2_d1 = phase_variance(1.0, r0)
    sigma2_d2 = phase_variance(2.0, r0)
    assert sigma2_d2 / sigma2_d1 == pytest.approx(2.0 ** (5.0 / 3.0), rel=1e-12)


def test_phase_variance_goes_to_zero_as_r0_infinity() -> None:
    """For fixed D, sigma^2 -> 0 as r0 -> infinity."""
    d = 10.0
    sigma2_finite = phase_variance(d, 1.0)
    sigma2_large_r0 = phase_variance(d, 1.0e6)
    sigma2_huge_r0 = phase_variance(d, 1.0e12)
    assert sigma2_large_r0 < 1e-8 * sigma2_finite
    assert sigma2_huge_r0 < sigma2_large_r0
    assert sigma2_huge_r0 == pytest.approx(0.0, abs=1e-15)


def test_phase_variance_kilo_consistency_h_band_maunakea_median() -> None:
    """Maunakea median r0=15 cm at 500 nm -> r0(H) ~ 60 cm.

    For a Keck sub-aperture diameter D = 2.5 m (Keck N=7), the
    piston-removed Noll variance is roughly 10.3 rad^2 — large, because
    KILO's design uses a deformable mirror to drive the residual down
    via fitting/temporal/WFS-noise terms. This test just pins the raw
    Kolmogorov number so any future change to the normalization
    coefficient is caught.
    """
    r0_h = r0_scaled(0.15, 1.65e-6)
    sigma2 = phase_variance(2.5, r0_h)
    expected = 1.0299 * (2.5 / r0_h) ** (5.0 / 3.0)
    assert sigma2 == pytest.approx(expected, rel=1e-12)
    assert sigma2 > 1.0  # large-aperture regime, as expected


def test_phase_variance_array_broadcasting() -> None:
    r0 = 0.15
    diameters_m = np.array([0.5, 1.0, 2.5, 10.0])
    out = phase_variance(diameters_m, r0)
    expected = NOLL_PISTON_REMOVED_COEFFICIENT * (diameters_m / r0) ** (5.0 / 3.0)
    np.testing.assert_allclose(out, expected, rtol=1e-12)


# ── opd_rms_m ────────────────────────────────────────────────────────────


def test_opd_rms_m_matches_phase_variance_conversion() -> None:
    """sigma_OPD = sqrt(sigma_phi^2) * lambda / (2 pi)."""
    d = 2.5
    r0 = 0.6
    wavelength_m = 1.65e-6
    sigma2 = phase_variance(d, r0)
    expected = math.sqrt(sigma2) * wavelength_m / (2.0 * math.pi)
    assert opd_rms_m(d, r0, wavelength_m) == pytest.approx(expected, rel=1e-12)


def test_opd_rms_m_wavelength_independent_when_r0_tracks_wavelength() -> None:
    """When r0 is scaled with wavelength, sigma_OPD is wavelength-independent.

    KILO's `ao_piston_opd_rms` returns the OPD as wavelength-independent
    physically (it's a path-length error, not a phase). Verify that
    cancellation here.
    """
    d = 2.5
    r0_500 = 0.15
    wavelengths_m = np.array([1.0e-6, 1.5e-6, 2.0e-6, 2.5e-6])
    sigma_opd_at_each_wl = np.array([
        opd_rms_m(d, r0_scaled(r0_500, wl), wl) for wl in wavelengths_m
    ])
    np.testing.assert_allclose(
        sigma_opd_at_each_wl, sigma_opd_at_each_wl[0], rtol=1e-10,
    )


def test_opd_rms_m_tip_tilt_removed_smaller_than_default() -> None:
    """sqrt(0.134 / 1.0299) ~ 0.361 factor between tip/tilt- and piston-removed."""
    d = 1.0
    r0 = 0.5
    wavelength_m = 1.65e-6
    sigma_piston_removed = opd_rms_m(d, r0, wavelength_m)
    sigma_tip_tilt_removed = opd_rms_m(d, r0, wavelength_m, tip_tilt_removed=True)
    ratio = sigma_tip_tilt_removed / sigma_piston_removed
    assert ratio == pytest.approx(math.sqrt(0.134 / 1.0299), rel=1e-12)


# ── strehl_marechal ──────────────────────────────────────────────────────


def test_strehl_marechal_unit_at_zero_opd() -> None:
    assert strehl_marechal(0.0, 1.65e-6) == pytest.approx(1.0, rel=1e-12)


def test_strehl_marechal_quarter_wave_classical_value() -> None:
    """Marechal sigma = lambda/14 -> S ~ exp(-(2 pi / 14)^2) ~ 0.819.

    This is the textbook diffraction-limit threshold.
    """
    wavelength_m = 1.65e-6
    sigma_opd = wavelength_m / 14.0
    expected = math.exp(-((2.0 * math.pi / 14.0) ** 2))
    assert strehl_marechal(sigma_opd, wavelength_m) == pytest.approx(
        expected, rel=1e-12,
    )


def test_strehl_marechal_matches_kilo_strehl_ratio_form() -> None:
    """exp(-(2 pi sigma_OPD/lambda)^2) == exp(-sigma_phi^2), KILO form."""
    sigma_phi2 = 0.5  # rad^2
    wavelength_m = 1.65e-6
    sigma_opd = math.sqrt(sigma_phi2) * wavelength_m / (2.0 * math.pi)
    s_opd = strehl_marechal(sigma_opd, wavelength_m)
    s_kilo = math.exp(-sigma_phi2)
    assert s_opd == pytest.approx(s_kilo, rel=1e-12)


def test_strehl_marechal_monotonic_in_opd() -> None:
    sigmas_m = np.linspace(0.0, 5.0e-7, 11)
    wavelength_m = 1.65e-6
    strehls = np.array([strehl_marechal(s, wavelength_m) for s in sigmas_m])
    assert np.all(np.diff(strehls) <= 0.0)


# ── NaN / inf / zero / negative input handling ──────────────────────────


def test_phase_structure_function_nan_r0_propagates() -> None:
    assert math.isnan(phase_structure_function(0.5, math.nan))


def test_opd_rms_m_nan_r0_propagates() -> None:
    assert math.isnan(opd_rms_m(2.5, math.nan, 1.65e-6))


def test_phase_variance_zero_r0_raises() -> None:
    with pytest.raises(ValueError, match="r0_m"):
        phase_variance(1.0, 0.0)


def test_phase_variance_negative_r0_raises() -> None:
    with pytest.raises(ValueError, match="r0_m"):
        phase_variance(1.0, -0.1)


@pytest.mark.parametrize("bad_d", [0.0, -1.0, math.inf, math.nan])
def test_phase_variance_invalid_aperture_raises(bad_d: float) -> None:
    with pytest.raises(ValueError, match="aperture_diameter_m"):
        phase_variance(bad_d, 0.15)


@pytest.mark.parametrize("bad_wavelength", [0.0, -1.0e-6, math.inf, math.nan])
def test_opd_rms_m_invalid_wavelength_raises(bad_wavelength: float) -> None:
    with pytest.raises(ValueError, match="wavelength_m"):
        opd_rms_m(2.5, 0.6, bad_wavelength)


def test_phase_structure_function_negative_separation_raises() -> None:
    with pytest.raises(ValueError, match="separation_m"):
        phase_structure_function(-0.1, 0.5)


def test_strehl_marechal_negative_opd_raises() -> None:
    with pytest.raises(ValueError, match="sigma_opd_m"):
        strehl_marechal(-1.0e-9, 1.65e-6)


def test_r0_scaled_zero_reference_wavelength_raises() -> None:
    with pytest.raises(ValueError, match="wavelength_ref_m"):
        r0_scaled(0.15, 1.65e-6, wavelength_ref_m=0.0)


def test_phase_variance_nan_in_array_propagates_only_at_nan_entries() -> None:
    r0_array = np.array([0.1, np.nan, 0.3])
    out = phase_variance(1.0, r0_array)
    assert np.isnan(out[1])
    assert np.isfinite(out[0])
    assert np.isfinite(out[2])


def test_phase_variance_zero_in_array_raises() -> None:
    with pytest.raises(ValueError, match="r0_m"):
        phase_variance(1.0, np.array([0.1, 0.0, 0.3]))


# ── Dtype invariance ────────────────────────────────────────────────────


def test_phase_variance_float32_in_float32_out() -> None:
    out = phase_variance(np.float32(2.5), np.float32(0.15))
    # Scalar float32 inputs return a Python float; check via array path too.
    diameters_m = np.array([1.0, 2.5, 5.0], dtype=np.float32)
    arr_out = phase_variance(diameters_m, np.float32(0.15))
    assert arr_out.dtype == np.float32
    assert isinstance(out, float)


def test_phase_structure_function_float32_in_float32_out() -> None:
    separations_m = np.array([0.1, 0.5, 1.0], dtype=np.float32)
    out = phase_structure_function(separations_m, np.float32(0.5))
    assert out.dtype == np.float32


def test_strehl_marechal_float32_in_float32_out() -> None:
    sigmas_m = np.array([0.0, 1.0e-7, 2.0e-7], dtype=np.float32)
    out = strehl_marechal(sigmas_m, np.float32(1.65e-6))
    assert out.dtype == np.float32


def test_opd_rms_m_float32_in_float32_out() -> None:
    diameters_m = np.array([1.0, 2.5, 5.0], dtype=np.float32)
    out = opd_rms_m(diameters_m, np.float32(0.15), np.float32(1.65e-6))
    assert out.dtype == np.float32


def test_r0_scaled_python_int_promotes_to_float() -> None:
    out = r0_scaled(1, 1.65e-6)
    assert isinstance(out, float)


def test_kolmogorov_validators_reject_infinity_propagate_nan() -> None:
    """Ultra-review Medium #4 regression: the kolmogorov validators are
    intentionally NaN-propagating (for missing measurements), but ±inf
    has no physical interpretation either. Pre-fix the validators only
    checked ``isfinite + (v <= 0)``, letting ±inf through and producing
    meaningless outputs like ``phase_variance(1.0, -inf) -> 0.0`` and
    ``r0_scaled(-inf, ...) -> -inf``.
    """
    import math
    import numpy as np
    import pytest

    from nullsim.physics.kolmogorov import (
        phase_variance,
        r0_scaled,
        strehl_marechal,
    )

    # Negative r0 already raised. ±inf r0 must now also raise.
    with pytest.raises(ValueError, match="r0_m"):
        r0_scaled(-np.inf, wavelength_m=1.65e-6)
    with pytest.raises(ValueError, match="r0_m"):
        r0_scaled(np.inf, wavelength_m=1.65e-6)
    with pytest.raises(ValueError, match="r0_m"):
        phase_variance(1.0, -np.inf)
    with pytest.raises(ValueError, match="r0_m"):
        phase_variance(1.0, np.inf)
    with pytest.raises(ValueError, match="sigma_opd_m"):
        strehl_marechal(-np.inf, 1.65e-6)
    with pytest.raises(ValueError, match="sigma_opd_m"):
        strehl_marechal(np.inf, 1.65e-6)

    # NaN still propagates (the intentional missing-measurement path).
    assert math.isnan(r0_scaled(float("nan"), wavelength_m=1.65e-6))
    assert math.isnan(phase_variance(1.0, float("nan")))
    assert math.isnan(strehl_marechal(float("nan"), 1.65e-6))
