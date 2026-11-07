"""Tests for the AO-realized null Monte-Carlo primitive."""

from __future__ import annotations

import math

import numpy as np
import pytest

from nullsim.physics.realized_null import (
    band_leakage_statistics,
    bias_cross_moments,
    bias_cross_variance,
    normalized_wavelength_weights,
    piston_covariance_with_common_mode_m2,
    realized_null_depth_mc,
)


def _make_50_50_combiner(n_wl: int) -> np.ndarray:
    u = (
        np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / math.sqrt(2.0)
    )
    stack = np.zeros((2, 2, n_wl), dtype=np.complex128)
    for k in range(n_wl):
        stack[:, :, k] = u
    return stack


def _make_duplicate_pair_unitary() -> np.ndarray:
    rows = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, -1.0],
        ],
        dtype=np.complex128,
    )
    rows[0] /= 2.0
    rows[1] /= math.sqrt(2.0)
    rows[2] /= math.sqrt(2.0)
    rows[3] /= 2.0
    return rows[:, :, np.newaxis]


def test_zero_piston_returns_ideal_null() -> None:
    """sigma=0 means no perturbation -> samples all equal the chip-only null."""

    stack = _make_50_50_combiner(2)
    field = np.array([1.0, 1.0], dtype=np.complex128)  # in-phase => routes to bright
    res = realized_null_depth_mc(
        u_stack=stack,
        stellar_field=field,
        dark_indices=np.array([1]),
        ao_piston_opd_rms_per_sub_m=0.0,
        wavelengths_m=np.array([1.0e-6, 1.25e-6]),
        n_realizations=50,
        seed=0,
    )
    assert res.ideal_null_depth == pytest.approx(0.0, abs=1e-15)
    assert res.mean_null_depth == pytest.approx(0.0, abs=1e-15)
    assert res.std_null_depth == pytest.approx(0.0, abs=1e-15)
    assert res.null_instability == 0.0


def test_realized_null_accepts_rectangular_output_stack() -> None:
    combiner = _make_50_50_combiner(1)[:, :, 0]
    stack = np.vstack([np.eye(2, dtype=np.complex128), combiner])[:, :, np.newaxis]
    field = np.array([1.0, 1.0], dtype=np.complex128)

    res = realized_null_depth_mc(
        u_stack=stack,
        stellar_field=field,
        dark_indices=np.array([3]),
        ao_piston_opd_rms_per_sub_m=0.0,
        wavelengths_m=np.array([1.0e-6]),
        n_realizations=1,
        seed=0,
    )

    assert res.ideal_null_depth == pytest.approx(0.0, abs=1.0e-15)
    assert res.mean_null_depth == pytest.approx(0.0, abs=1.0e-15)


def test_small_piston_quadratic_in_sigma_phi() -> None:
    """For tiny sigma, mean null depth scales as sigma_phi^2 (linear-order leading)."""

    stack = _make_50_50_combiner(1)
    field = np.array([1.0, 1.0], dtype=np.complex128)
    wl = 1.25e-6
    # sigma_phi = 2*pi*sigma_OPD/lambda. Pick sigma_phi = 0.01 rad.
    sigma_phi = 0.01
    sigma_OPD = sigma_phi * wl / (2.0 * math.pi)
    res = realized_null_depth_mc(
        u_stack=stack,
        stellar_field=field,
        dark_indices=np.array([1]),
        ao_piston_opd_rms_per_sub_m=sigma_OPD,
        wavelengths_m=np.array([wl]),
        n_realizations=20000,
        seed=42,
    )
    # Analytic: dark port intensity = |1/sqrt(2) * (1*exp(i phi_0) - 1*exp(i phi_1))|^2
    # = (1 - cos(phi_0 - phi_1)). E[1 - cos(delta phi)] = 1 - exp(-Var[delta phi]/2)
    # = 1 - exp(-(2 sigma_phi^2)/2) = 1 - exp(-sigma_phi^2) ~ sigma_phi^2.
    # Normalized by total power (=2 always for this unitary): T = (1 - exp(-sigma_phi^2))/2
    expected = (1.0 - math.exp(-sigma_phi ** 2)) / 2.0
    assert res.mean_null_depth == pytest.approx(expected, rel=0.05)


def test_correlated_piston_covariance_can_be_common_mode_only() -> None:
    stack = _make_50_50_combiner(1)
    field = np.array([1.0, 1.0], dtype=np.complex128)
    wl = 1.25e-6
    sigma_opd = 20.0e-9

    independent = realized_null_depth_mc(
        u_stack=stack,
        stellar_field=field,
        dark_indices=np.array([1], dtype=np.intp),
        ao_piston_opd_rms_per_sub_m=0.0,
        ao_piston_opd_covariance_m2=np.eye(2, dtype=np.float64) * sigma_opd**2,
        wavelengths_m=np.array([wl]),
        n_realizations=8000,
        seed=3,
    )
    common_mode = realized_null_depth_mc(
        u_stack=stack,
        stellar_field=field,
        dark_indices=np.array([1], dtype=np.intp),
        ao_piston_opd_rms_per_sub_m=0.0,
        ao_piston_opd_covariance_m2=np.full((2, 2), sigma_opd**2, dtype=np.float64),
        wavelengths_m=np.array([wl]),
        n_realizations=8000,
        seed=3,
    )

    assert independent.mean_null_depth > 1.0e-3
    assert common_mode.mean_null_depth == pytest.approx(0.0, abs=1.0e-15)


def test_common_differential_covariance_structure() -> None:
    """Rank-1 inter-telescope mode loads the +/-1 telescope-sign direction."""

    sigma_iid_m = 5.0e-9
    sigma_diff_m = 30.0e-9
    signs = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float64)

    covariance = piston_covariance_with_common_mode_m2(
        sigma_iid_m=sigma_iid_m,
        sigma_diff_m=sigma_diff_m,
        telescope_signs=signs,
    )

    # Off-diagonal IID terms vanish; the common mode adds sigma_diff^2/4 with
    # the sign of s_i s_j.
    assert covariance.shape == (4, 4)
    assert covariance == pytest.approx(
        np.eye(4) * sigma_iid_m**2
        + np.outer(signs, signs) * (sigma_diff_m**2 / 4.0)
    )

    weight = signs / math.sqrt(signs.size)
    eigen_along_signs = float(weight @ covariance @ weight)
    assert eigen_along_signs == pytest.approx(
        sigma_iid_m**2 + signs.size * sigma_diff_m**2 / 4.0
    )


def test_common_differential_covariance_reproduces_two_beam_null() -> None:
    """A pure differential piston yields the two-beam analytic null floor."""

    stack = _make_50_50_combiner(1)
    field = np.array([1.0, 1.0], dtype=np.complex128)
    wl = 1.25e-6
    sigma_diff_m = 30.0e-9

    covariance = piston_covariance_with_common_mode_m2(
        sigma_iid_m=0.0,
        sigma_diff_m=sigma_diff_m,
        telescope_signs=np.array([1.0, -1.0], dtype=np.float64),
    )
    result = realized_null_depth_mc(
        u_stack=stack,
        stellar_field=field,
        dark_indices=np.array([1], dtype=np.intp),
        ao_piston_opd_rms_per_sub_m=0.0,
        ao_piston_opd_covariance_m2=covariance,
        wavelengths_m=np.array([wl]),
        n_realizations=40000,
        seed=11,
    )

    # The two telescopes carry +/- sigma_diff/2 of a single common Gaussian,
    # so the differential OPD has RMS sigma_diff. The 50/50 dark port intensity
    # is E[1 - cos(delta_phi)] = 1 - exp(-sigma_phi_diff^2 / 2), normalized by
    # the total power (=2) of this unitary.
    sigma_phi_diff = 2.0 * math.pi * sigma_diff_m / wl
    expected = (1.0 - math.exp(-(sigma_phi_diff**2) / 2.0)) / 2.0
    assert result.mean_null_depth == pytest.approx(expected, rel=0.05)


def test_common_differential_covariance_rejects_non_unit_signs() -> None:
    """Sign vectors must be exactly +/-1 per mode."""

    with pytest.raises(ValueError, match="telescope_signs"):
        piston_covariance_with_common_mode_m2(
            sigma_iid_m=1.0e-9,
            sigma_diff_m=1.0e-9,
            telescope_signs=np.array([1.0, 0.0], dtype=np.float64),
        )


def test_realized_null_with_dark_indices_only_dark_port_counted() -> None:
    """Three-mode test: only the listed dark index contributes to leakage."""

    # Random unitary (3x3) so dark/bright partition is meaningful.
    rng = np.random.default_rng(0)
    A = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    q, _ = np.linalg.qr(A)
    stack = q[:, :, np.newaxis]
    field = np.array([1.0, 1.0, 1.0], dtype=np.complex128)
    res_dark0 = realized_null_depth_mc(
        u_stack=stack,
        stellar_field=field,
        dark_indices=np.array([0]),
        ao_piston_opd_rms_per_sub_m=0.0,
        wavelengths_m=np.array([1.25e-6]),
        n_realizations=1,
        seed=0,
    )
    res_dark12 = realized_null_depth_mc(
        u_stack=stack,
        stellar_field=field,
        dark_indices=np.array([1, 2]),
        ao_piston_opd_rms_per_sub_m=0.0,
        wavelengths_m=np.array([1.25e-6]),
        n_realizations=1,
        seed=0,
    )
    # The sum across all ports = 1 (unitary preserves power).
    assert res_dark0.mean_null_depth + res_dark12.mean_null_depth == pytest.approx(
        1.0, rel=1e-12,
    )


def test_seed_makes_output_deterministic() -> None:
    stack = _make_50_50_combiner(1)
    field = np.array([1.0, 1.0], dtype=np.complex128)
    res_a = realized_null_depth_mc(
        u_stack=stack,
        stellar_field=field,
        dark_indices=np.array([1]),
        ao_piston_opd_rms_per_sub_m=1e-9,
        wavelengths_m=np.array([1.25e-6]),
        n_realizations=100,
        seed=7,
    )
    res_b = realized_null_depth_mc(
        u_stack=stack,
        stellar_field=field,
        dark_indices=np.array([1]),
        ao_piston_opd_rms_per_sub_m=1e-9,
        wavelengths_m=np.array([1.25e-6]),
        n_realizations=100,
        seed=7,
    )
    assert res_a.mean_null_depth == res_b.mean_null_depth
    assert res_a.std_null_depth == res_b.std_null_depth


def test_kilo_keck_j_band_realized_null_in_target_range() -> None:
    """sigma_phi ~ 7 mrad (KILO J=4) -> realized null ~ 1e-4 with kernel-null chip."""

    n_modes = 24
    # Make a unitary that perfectly nulls 24 in-phase identical inputs at
    # dark ports: project orthogonal to (1,1,...,1)/sqrt(24).
    rng = np.random.default_rng(0)
    # Use Gram-Schmidt to form an orthonormal basis where the first column
    # is the in-phase vector (-> bright port 0).
    bright_vec = np.ones(n_modes, dtype=np.complex128) / math.sqrt(n_modes)
    other = rng.normal(size=(n_modes, n_modes - 1)) + 1j * rng.normal(
        size=(n_modes, n_modes - 1)
    )
    basis = np.column_stack([bright_vec, other])
    q, _ = np.linalg.qr(basis)
    # Transpose so columns are inputs and rows are outputs (U @ E).
    u = q.conj().T  # (n_modes, n_modes)
    stack = u[:, :, np.newaxis]
    field = np.ones(n_modes, dtype=np.complex128)  # in-phase star at unit amplitude
    wl = 1.25e-6
    # KILO targets sigma_phi ~ 6.79e-3 rad at J-band; convert to OPD.
    sigma_phi = 6.79e-3
    sigma_OPD = sigma_phi * wl / (2.0 * math.pi)
    res = realized_null_depth_mc(
        u_stack=stack,
        stellar_field=field,
        dark_indices=np.arange(1, n_modes),
        ao_piston_opd_rms_per_sub_m=sigma_OPD,
        wavelengths_m=np.array([wl]),
        n_realizations=2000,
        seed=11,
    )
    # Ideal null on the dark ports is zero by construction.
    assert res.ideal_null_depth == pytest.approx(0.0, abs=1e-13)
    # The realized null with sigma_phi ~ 7 mrad should land near
    # E[leakage / total]. For M=24 inputs all with iid phase noise,
    # leakage / total ~= 1 - |<exp(i phi)>|^2 = 1 - exp(-sigma_phi^2)
    # ~= sigma_phi^2 ~= 4.6e-5. KILO's much higher 1.4e-4 reflects
    # imperfect chip null + transport effects we are NOT modelling here.
    expected = 1.0 - math.exp(-sigma_phi ** 2)
    assert res.mean_null_depth == pytest.approx(expected, rel=0.10), (
        f"mean_null_depth={res.mean_null_depth:.3e}, expected ~{expected:.3e}"
    )



def test_fiber_phase_noise_increases_null_floor() -> None:
    """Tier 4.4: IID per-mode fiber phase noise lifts the null floor.

    KILO ``FiberConfig.phase_noise_rms_rad ≈ 0.01`` adds a residual
    leakage floor independent of the AO piston. Use a chip that nulls
    the field to ~0 by construction (route in-phase to bright port 0)
    so the lift due to fiber phase noise is the only signal in the
    null depth.
    """
    n_modes = 4
    bright_col = np.ones(n_modes, dtype=np.complex128) / math.sqrt(n_modes)
    rng = np.random.default_rng(0)
    A = rng.normal(size=(n_modes, n_modes)) + 1j * rng.normal(size=(n_modes, n_modes))
    q, _ = np.linalg.qr(A)
    cols = q.copy()
    cols[:, 0] = bright_col
    cols, _ = np.linalg.qr(cols)
    u = cols.conj().T  # rows are outputs; row 0 captures the bright field
    u_stack = u[:, :, np.newaxis]
    field = bright_col
    dark = np.array([1, 2, 3], dtype=np.intp)

    res_no_fiber = realized_null_depth_mc(
        u_stack=u_stack,
        stellar_field=field,
        dark_indices=dark,
        ao_piston_opd_rms_per_sub_m=0.0,
        wavelengths_m=np.array([1.25e-6]),
        n_realizations=4000,
        seed=0,
        fiber_phase_noise_rms_rad=0.0,
    )
    res_fiber = realized_null_depth_mc(
        u_stack=u_stack,
        stellar_field=field,
        dark_indices=dark,
        ao_piston_opd_rms_per_sub_m=0.0,
        wavelengths_m=np.array([1.25e-6]),
        n_realizations=4000,
        seed=0,
        fiber_phase_noise_rms_rad=0.05,
    )
    # Adding fiber phase noise must STRICTLY increase the realized null
    # floor (the chip's bright-port mode is perturbed off-axis).
    assert res_no_fiber.mean_null_depth < 1.0e-12
    assert res_fiber.mean_null_depth > res_no_fiber.mean_null_depth
    assert res_fiber.mean_null_depth > 1.0e-5


def test_mode_group_indices_correlate_split_input_perturbations() -> None:
    """Duplicated chip inputs from one aperture share physical atmosphere.

    The test unitary's dark ports only see differences between duplicate
    copies of each physical aperture. Grouped perturbations therefore keep
    those dark ports nulled, while independent post-split draws leak.
    """

    u_stack = _make_duplicate_pair_unitary()
    field = np.ones(4, dtype=np.complex128)
    dark = np.array([1, 2], dtype=np.intp)
    group_indices = np.array([0, 0, 1, 1], dtype=np.intp)
    wl = 1.25e-6
    sigma_phi = 0.1
    sigma_opd = sigma_phi * wl / (2.0 * math.pi)

    independent = realized_null_depth_mc(
        u_stack=u_stack,
        stellar_field=field,
        dark_indices=dark,
        ao_piston_opd_rms_per_sub_m=sigma_opd,
        wavelengths_m=np.array([wl]),
        n_realizations=5000,
        seed=0,
        scintillation_index_per_sub=0.1,
    )
    grouped = realized_null_depth_mc(
        u_stack=u_stack,
        stellar_field=field,
        dark_indices=dark,
        ao_piston_opd_rms_per_sub_m=sigma_opd,
        wavelengths_m=np.array([wl]),
        n_realizations=5000,
        seed=0,
        scintillation_index_per_sub=0.1,
        mode_group_indices=group_indices,
    )

    assert independent.mean_null_depth > 1.0e-3
    assert grouped.mean_null_depth == pytest.approx(0.0, abs=1.0e-15)


def test_mode_group_indices_reject_non_contiguous_groups() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        realized_null_depth_mc(
            u_stack=_make_duplicate_pair_unitary(),
            stellar_field=np.ones(4, dtype=np.complex128),
            dark_indices=np.array([1, 2], dtype=np.intp),
            ao_piston_opd_rms_per_sub_m=0.0,
            wavelengths_m=np.array([1.25e-6]),
            n_realizations=1,
            seed=0,
            mode_group_indices=np.array([0, 0, 2, 2], dtype=np.intp),
        )


def test_fiber_phase_noise_rejects_negative_value() -> None:
    """fiber_phase_noise_rms_rad must be non-negative."""
    with pytest.raises(ValueError, match="fiber_phase_noise_rms_rad"):
        realized_null_depth_mc(
            u_stack=np.eye(2, dtype=np.complex128)[:, :, np.newaxis],
            stellar_field=np.ones(2, dtype=np.complex128) / math.sqrt(2),
            dark_indices=np.array([1], dtype=np.intp),
            ao_piston_opd_rms_per_sub_m=0.0,
            wavelengths_m=np.array([1.25e-6]),
            n_realizations=10,
            seed=0,
            fiber_phase_noise_rms_rad=-0.01,
        )


def test_negative_n_realizations_per_azimuth_rejected() -> None:
    """Negative MC counts used to silently produce ideal/no-instability output."""
    with pytest.raises(ValueError, match="n_realizations_per_azimuth"):
        realized_null_depth_mc(
            u_stack=np.eye(2, dtype=np.complex128)[:, :, np.newaxis],
            stellar_field=np.ones(2, dtype=np.complex128) / math.sqrt(2),
            dark_indices=np.array([1], dtype=np.intp),
            ao_piston_opd_rms_per_sub_m=0.0,
            wavelengths_m=np.array([1.25e-6]),
            n_realizations_per_azimuth=-5,
            seed=0,
        )


def test_band_statistics_two_bin_toy_by_hand() -> None:
    """Photon-weighted band leakage N = 0.25 L0 + 0.75 L1 per paired draw.

    Two azimuths x two draws. By hand: N = [3.25, 0.75 | 1.25, 4.25], so
    mean 2.375, within-azimuth variances 1.5625 and 2.25, and
    sigma_MC = sqrt(1.90625).
    """

    samples = np.array([[1.0, 3.0, 2.0, 2.0], [4.0, 0.0, 1.0, 5.0]])
    weights = normalized_wavelength_weights(np.array([1.0, 3.0]), 2)
    mean, rms = band_leakage_statistics(samples, weights, n_azimuths=2)
    assert mean == pytest.approx(2.375, rel=1e-15)
    assert rms == pytest.approx(math.sqrt(1.90625), rel=1e-15)


def test_null_instability_is_cv_of_photon_weighted_band_leakage() -> None:
    """Issue 18: the CV is that of the band sum of paired draws, not the
    mean of per-bin CVs. Two bins of a random 4-mode chip share each piston
    draw, so per-bin leakages correlate and the band CV differs from the
    per-bin average. Check against the returned samples directly.
    """

    rng = np.random.default_rng(1)
    n_modes = 4
    q, _ = np.linalg.qr(
        rng.normal(size=(n_modes, n_modes)) + 1j * rng.normal(size=(n_modes, n_modes))
    )
    weights = np.array([5.0, 1.0])
    res = realized_null_depth_mc(
        u_stack=q[:, :, np.newaxis].repeat(2, axis=2),
        stellar_field=np.ones(n_modes, dtype=np.complex128) / 2.0,
        dark_indices=np.array([1, 2, 3], dtype=np.intp),
        ao_piston_opd_rms_per_sub_m=10.0e-9,
        wavelengths_m=np.array([1.25e-6, 1.65e-6]),
        n_realizations=2000,
        seed=42,
        wavelength_weights=weights,
    )
    samples = np.asarray(res.samples_by_wavelength)
    band = (weights / weights.sum()) @ samples
    assert res.mean_null_depth == pytest.approx(np.mean(band), rel=1e-12)
    assert res.null_fluctuation_rms == pytest.approx(np.std(band), rel=1e-12)
    assert res.null_instability == pytest.approx(np.std(band) / np.mean(band), rel=1e-12)
    per_bin_cv = np.mean(np.std(samples, axis=1) / np.mean(samples, axis=1))
    assert abs(res.null_instability / per_bin_cv - 1.0) > 0.02


def test_static_azimuth_differences_are_not_instability() -> None:
    """Row 39: azimuths with different static nulls but no fluctuation give
    zero CV. Pooling azimuths first made the CV jump to ~1 at any nonzero
    piston, however small."""

    res = realized_null_depth_mc(
        u_stack=_make_50_50_combiner(1),
        stellar_field=np.array([[1.0, 1.0], [1.0, 0.8]], dtype=np.complex128),
        dark_indices=np.array([1], dtype=np.intp),
        ao_piston_opd_rms_per_sub_m=1.0e-27,
        wavelengths_m=np.array([1.25e-6]),
        n_realizations_per_azimuth=50,
        seed=0,
    )
    assert res.mean_null_depth > 1e-3
    assert res.null_instability < 1e-12


def test_azimuth_marginalization_pools_samples() -> None:
    """Phase 1: passing stellar_field with shape (n_az, M) pools
    n_az * n_realizations_per_azimuth samples per wavelength."""

    u_stack = _make_50_50_combiner(1)
    field = np.array([1.0 + 0j, 1.0 + 0j])
    dark = np.array([1], dtype=np.intp)

    fields_per_az = np.broadcast_to(field, (6, 2)).copy()
    res = realized_null_depth_mc(
        u_stack=u_stack,
        stellar_field=fields_per_az,
        dark_indices=dark,
        ao_piston_opd_rms_per_sub_m=10.0e-9,
        wavelengths_m=np.array([1.25e-6]),
        n_realizations_per_azimuth=200,
        seed=0,
    )
    assert res.n_azimuths == 6
    assert res.n_realizations_per_azimuth == 200
    assert res.n_samples == 6 * 200  # pooled per wavelength
    # null_depth_dark alias must match mean_null_depth.
    assert res.null_depth_dark == res.mean_null_depth


def test_wavelength_dependent_stellar_fields_change_ideal_null() -> None:
    """Shape (n_az, n_wl, M) lets deterministic chromatic transport vary by λ."""

    u_stack = _make_50_50_combiner(2)
    dark = np.array([1], dtype=np.intp)
    fields_per_az_wl = np.array(
        [
            [
                [1.0 + 0.0j, 1.0 + 0.0j],
                [1.0 + 0.0j, -1.0 + 0.0j],
            ],
        ],
        dtype=np.complex128,
    )

    res = realized_null_depth_mc(
        u_stack=u_stack,
        stellar_field=fields_per_az_wl,
        dark_indices=dark,
        ao_piston_opd_rms_per_sub_m=0.0,
        wavelengths_m=np.array([1.25e-6, 1.65e-6]),
        n_realizations_per_azimuth=0,
        seed=0,
    )

    assert res.ideal_null_depth == pytest.approx(0.5)
    assert res.mean_null_depth == pytest.approx(0.5)
    assert res.n_azimuths == 1


def test_per_azimuth_transport_imbalance_inflates_null() -> None:
    """Per-azimuth transmission imbalance breaks the 50/50 combiner's
    null even without AO piston, so the pooled mean lies above the
    transport-free reference."""

    u_stack = _make_50_50_combiner(1)
    field = np.array([1.0 + 0j, 1.0 + 0j])
    dark = np.array([1], dtype=np.intp)

    # Balanced field at all 6 azimuths -> chip null is perfect.
    balanced = np.broadcast_to(field, (6, 2)).copy()
    res_balanced = realized_null_depth_mc(
        u_stack=u_stack,
        stellar_field=balanced,
        dark_indices=dark,
        ao_piston_opd_rms_per_sub_m=0.0,
        wavelengths_m=np.array([1.25e-6]),
        n_realizations_per_azimuth=100,
        seed=0,
    )

    # Imbalanced per-azimuth transmissions -> 50/50 combiner leaks dark.
    az_factors = np.array([1.0, 0.95, 0.9, 0.85, 0.9, 0.95])
    imbalanced = np.empty((6, 2), dtype=np.complex128)
    for a, factor in enumerate(az_factors):
        imbalanced[a] = field * np.array([1.0, factor], dtype=np.complex128)
    res_imbalanced = realized_null_depth_mc(
        u_stack=u_stack,
        stellar_field=imbalanced,
        dark_indices=dark,
        ao_piston_opd_rms_per_sub_m=0.0,
        wavelengths_m=np.array([1.25e-6]),
        n_realizations_per_azimuth=100,
        seed=0,
    )
    assert res_imbalanced.mean_null_depth > res_balanced.mean_null_depth


def test_n_realizations_required() -> None:
    """The MC needs either n_realizations or n_realizations_per_azimuth."""

    u_stack = _make_50_50_combiner(1)
    field = np.array([1.0 + 0j, 1.0 + 0j])
    dark = np.array([1], dtype=np.intp)
    with pytest.raises(ValueError, match="n_realizations"):
        realized_null_depth_mc(
            u_stack=u_stack,
            stellar_field=field,
            dark_indices=dark,
            ao_piston_opd_rms_per_sub_m=10.0e-9,
            wavelengths_m=np.array([1.25e-6]),
            seed=0,
        )


_DFT3 = np.exp(-2j * np.pi * np.outer(np.arange(3), np.arange(3)) / 3) / math.sqrt(3)
_BRACEWELL = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / math.sqrt(2.0)


@pytest.mark.parametrize(
    ("unitary", "field", "piston_rad", "sigma_i", "old_fails"),
    [
        # (a) Bracewell, static amplitude imbalance, piston, amplitude noise
        (_BRACEWELL, np.array([1.0, 0.98], dtype=np.complex128), 0.003, 0.01, False),
        # (b) Bracewell with a 0.02 rad static phase error and piston
        (_BRACEWELL, np.array([1.0, np.exp(0.02j)]), 0.003, 0.0, True),
        # (c) three-input DFT nulled row, amplitude noise only
        (_DFT3, np.ones(3, dtype=np.complex128), 0.0, 0.01, True),
    ],
    ids=["bracewell", "static_phase_error", "dft3_amplitude"],
)
def test_bias_cross_term_matches_bias_averaged_direct_monte_carlo(
    unitary: np.ndarray,
    field: np.ndarray,
    piston_rad: float,
    sigma_i: float,
    old_fails: bool,
) -> None:
    """Issue 19: setpoint bias b, i.i.d. across inputs with RMS 0.02 rad.

    The reference applies every sign pattern b in {+-beta}^M as a
    deterministic input phase, so E[b b^T] = beta^2 I exactly, and averages
    the direct MC variance over them. The prediction sigma_MC^2 +
    4 L_b S / D (L_b = beta^2 D) agrees to the linearization residual
    (~4e-4 here). The earlier phase-only form 2 sigma_phase^2 / L_phase,dyn
    L_b misses (b), whose variance holds the static-error x piston beat, and
    (c), where amplitude noise couples to a phase bias in a complex row.
    """

    import itertools

    wavelength_m, beta = 1.25e-6, 0.02

    def run(input_field: np.ndarray, sigma_amp: float, fields: bool = False):
        return realized_null_depth_mc(
            u_stack=unitary[:, :, np.newaxis],
            stellar_field=input_field,
            dark_indices=np.array([1], dtype=np.intp),
            ao_piston_opd_rms_per_sub_m=piston_rad * wavelength_m / (2.0 * math.pi),
            scintillation_index_per_sub=sigma_amp,
            wavelengths_m=np.array([wavelength_m]),
            n_realizations_per_azimuth=20_000,
            seed=1,
            return_dark_fields=fields,
        )

    weights = np.ones(1)
    unbiased = run(field, sigma_i, fields=True)
    _, d_moment = bias_cross_moments(unbiased, weights)
    bias_leakage = beta**2 * d_moment
    predicted = unbiased.null_fluctuation_rms**2 + bias_cross_variance(
        unbiased, weights, bias_leakage,
    )
    direct = np.mean([
        run(field * np.exp(1j * beta * np.array(signs)), sigma_i).null_fluctuation_rms**2
        for signs in itertools.product((-1.0, 1.0), repeat=field.size)
    ])
    assert predicted == pytest.approx(direct, rel=2e-3)
    assert direct > 1.3 * unbiased.null_fluctuation_rms**2  # the bias matters

    phase_leg = run(field, 0.0)
    dynamic = phase_leg.mean_null_depth - phase_leg.ideal_null_depth
    old = unbiased.null_fluctuation_rms**2 + (
        2.0 * phase_leg.null_fluctuation_rms**2 / dynamic * bias_leakage
        if dynamic > 0.0
        else 0.0
    )
    if old_fails:
        assert abs(old / direct - 1.0) > 0.5
