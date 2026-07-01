"""Diagnostic-plot statistics tests for clements, jones, and fresnel primitives.

These ``@pytest.mark.stats`` tests check physical/numerical consistency of the
three new physics primitives and write per-test PNGs under
``tests/output/stats/`` so a human can inspect the curves. A passing test with
a visually broken plot indicates the tolerance is too loose.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend so tests work without a display.

import matplotlib.pyplot as plt
import numpy as np
import pytest

from nullsim.physics.clements import (
    decompose_clements,
    reconstruct_clements,
)
from nullsim.physics.fresnel import angular_spectrum_propagate
from nullsim.physics.jones import (
    apply_jones,
    half_wave_plate,
    jones_to_stokes,
    linear_polarizer,
    quarter_wave_plate,
)

_STATS_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "stats"


def _stats_output_path(filename: str) -> Path:
    """Return an absolute path for a diagnostic plot under tests/output/stats."""
    _STATS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return _STATS_OUTPUT_DIR / filename


def _haar_unitary(n_modes: int, seed: int) -> np.ndarray:
    """Draw a Haar-distributed unitary (Mezzadri 2007, Eq. 5.12)."""
    rng = np.random.default_rng(seed)
    z = (
        rng.standard_normal(size=(n_modes, n_modes))
        + 1j * rng.standard_normal(size=(n_modes, n_modes))
    ) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    diag_signs = np.diag(r) / np.abs(np.diag(r))
    return q @ np.diag(diag_signs)


# ---------------------------------------------------------------------------
# Clements: round-trip residual heatmap
# ---------------------------------------------------------------------------


@pytest.mark.stats
def test_clements_round_trip_residual_heatmap_n8() -> None:
    """|U - U_reconstructed| must be ~machine epsilon for a Haar-random U_8.

    A clean dark heatmap in log scale indicates uniform precision across the
    mesh. Bright pixels would reveal which mesh column lost precision.
    """
    n_modes = 8
    U = _haar_unitary(n_modes, seed=20260515)

    decomp = decompose_clements(U)
    U_reconstructed = reconstruct_clements(decomp)
    residual = np.abs(U - U_reconstructed)

    assert np.max(residual) < 1.0e-12

    # Use a finite floor so log10 does not blow up at exact-zero entries.
    log_residual = np.log10(residual + 1.0e-18)

    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    image = ax.imshow(log_residual, cmap="magma", aspect="equal")
    ax.set_title("Clements N=8 round-trip residual log10|U - U_rec|")
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    fig.colorbar(image, ax=ax, label="log10 residual")
    fig.tight_layout()
    fig.savefig(_stats_output_path("clements_round_trip_residual_n8.png"), dpi=120)
    plt.close(fig)


@pytest.mark.stats
def test_clements_identity_theta_histogram() -> None:
    """For I_8 every theta angle must sit at 0 (mod pi), giving one cluster at 0.

    The architect spec calls for a bimodal cluster at 0 and pi. The current
    implementation chooses the zero branch when the column is already aligned,
    so every angle collapses to 0; this is still consistent with the "0 or pi
    (mod 2 pi)" specification because pi mod pi == 0.
    """
    decomp = decompose_clements(np.eye(8, dtype=np.complex128))
    theta_mod_pi = np.mod(decomp.theta, np.pi)

    assert np.max(np.abs(theta_mod_pi)) < 1.0e-12

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.hist(decomp.theta, bins=21, range=(-0.05, np.pi + 0.05))
    ax.set_xlabel("theta (rad)")
    ax.set_ylabel("count")
    ax.set_title("Clements I_8 theta distribution (should cluster at 0 or pi)")
    fig.tight_layout()
    fig.savefig(_stats_output_path("clements_identity_theta_histogram.png"), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Jones: HWP angle sweep, polarizer Malus law, Jones-to-Stokes consistency
# ---------------------------------------------------------------------------


@pytest.mark.stats
def test_half_wave_plate_angle_sweep_swaps_h_and_v() -> None:
    """Sweep HWP fast-axis angle and verify |E|^2 swap between H and V.

    For input vertical [0, 1], the HWP at angle alpha produces an output whose
    horizontal-intensity component is sin^2(2 alpha) — a sinusoidal swap with
    period pi/2, peaking to 1 at alpha = pi/4 (the REQUIRED HWP behavior).
    """
    vertical = np.array([0.0, 1.0], dtype=np.complex128)
    alpha_rad = np.linspace(0.0, np.pi, 91)

    horizontal_power = np.array(
        [
            np.abs(apply_jones(half_wave_plate(alpha), vertical)[0]) ** 2
            for alpha in alpha_rad
        ]
    )
    expected = np.sin(2.0 * alpha_rad) ** 2

    np.testing.assert_allclose(horizontal_power, expected, atol=1.0e-12)

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.plot(np.degrees(alpha_rad), horizontal_power, "o", label="|E_H(alpha)|^2")
    ax.plot(np.degrees(alpha_rad), expected, "-", label="sin^2(2 alpha)")
    ax.axvline(45.0, color="gray", linestyle="--", label="alpha = 45 deg")
    ax.set_xlabel("HWP fast-axis angle (deg)")
    ax.set_ylabel("Output horizontal intensity")
    ax.set_title("HWP sweep on vertical input: expected H/V swap at 45 deg")
    ax.legend()
    fig.tight_layout()
    fig.savefig(_stats_output_path("jones_hwp_angle_sweep.png"), dpi=120)
    plt.close(fig)


@pytest.mark.stats
def test_quarter_wave_plate_yields_right_circular_from_linear_45() -> None:
    """QWP at 0 deg converts linear-45 to right-circular polarization.

    The Stokes vector of right-circular polarization is [1, 0, 0, +1]
    (V_circ > 0 in the nullsim convention).
    """
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    linear_45 = np.array([inv_sqrt2, inv_sqrt2], dtype=np.complex128)

    rcp = apply_jones(quarter_wave_plate(0.0), linear_45)
    stokes = jones_to_stokes(rcp)

    np.testing.assert_allclose(rcp, [inv_sqrt2, 1j * inv_sqrt2], atol=1.0e-12)
    np.testing.assert_allclose(stokes, [1.0, 0.0, 0.0, 1.0], atol=1.0e-12)


@pytest.mark.stats
def test_linear_polarizer_obeys_malus_law_under_angle_sweep() -> None:
    """Polarizer at theta on horizontal input gives I(theta) = cos^2(theta).

    The architect's spec requires a clean cos^2 curve. Tolerance is tight
    because the formula is exact in floating point.
    """
    horizontal = np.array([1.0, 0.0], dtype=np.complex128)
    theta_rad = np.linspace(0.0, np.pi, 91)

    transmitted_intensity = np.array(
        [
            float(
                np.abs(apply_jones(linear_polarizer(theta), horizontal)[0]) ** 2
                + np.abs(apply_jones(linear_polarizer(theta), horizontal)[1]) ** 2
            )
            for theta in theta_rad
        ]
    )
    expected = np.cos(theta_rad) ** 2

    np.testing.assert_allclose(transmitted_intensity, expected, atol=1.0e-12)

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.plot(np.degrees(theta_rad), transmitted_intensity, "o", label="numerical")
    ax.plot(np.degrees(theta_rad), expected, "-", label="cos^2(theta)")
    ax.set_xlabel("polarizer angle (deg)")
    ax.set_ylabel("Transmitted intensity")
    ax.set_title("Malus's law: linear polarizer on horizontal input")
    ax.legend()
    fig.tight_layout()
    fig.savefig(_stats_output_path("jones_polarizer_malus.png"), dpi=120)
    plt.close(fig)


@pytest.mark.stats
def test_jones_to_stokes_four_canonical_states_match_textbook() -> None:
    """Bar-chart consistency check of S_1, S_2, S_3 for four canonical states.

    Tolerance is 1e-12 because the conversion is closed-form.
    """
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    states = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [inv_sqrt2, inv_sqrt2],
            [inv_sqrt2, 1j * inv_sqrt2],
        ],
        dtype=np.complex128,
    )
    expected = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    stokes = jones_to_stokes(states)
    np.testing.assert_allclose(stokes, expected, atol=1.0e-12)

    state_labels = ["H", "V", "L+45", "RCP"]
    component_labels = ["S1 (Q)", "S2 (U)", "S3 (V)"]
    bar_positions = np.arange(len(state_labels))

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    width = 0.27
    for component_index, label in enumerate(component_labels):
        ax.bar(
            bar_positions + (component_index - 1) * width,
            stokes[:, component_index + 1],
            width=width,
            label=label,
        )
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(state_labels)
    ax.set_ylim(-1.2, 1.2)
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_title("Stokes [S1, S2, S3] across four canonical Jones states")
    ax.legend()
    fig.tight_layout()
    fig.savefig(_stats_output_path("jones_to_stokes_canonical.png"), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fresnel: round-trip residual cut + Gaussian waist evolution
# ---------------------------------------------------------------------------


@pytest.mark.stats
def test_fresnel_gaussian_round_trip_center_cut_is_flat() -> None:
    """A center cut through the round-trip residual must be near machine eps.

    Setup mirrors the architect spec at coarse-grid scale: 64 x 64 to keep the
    test runtime modest while still using a low-NA Gaussian whose round trip
    recovers to 1e-10.
    """
    n_pixels = 64
    pixel_pitch_um = 5.0
    wavelength_um = 1.55
    distance_m = 2.0e-3
    waist_um = 90.0

    coords_um = (np.arange(n_pixels, dtype=np.float64) - n_pixels / 2) * pixel_pitch_um
    x_um, y_um = np.meshgrid(coords_um, coords_um, indexing="xy")
    U_0 = np.exp(-((x_um**2 + y_um**2) / waist_um**2)).astype(np.complex128)

    # Code Review 2026-05-16 Issue 26: this round-trip uses a narrow
    # Gaussian whose spatial spectrum is well-bounded, so the Nyquist
    # sampling check (added to angular_spectrum_propagate) is overly
    # strict here. Disable for these stats tests.
    U_z = angular_spectrum_propagate(
        U_0,
        distance_m=distance_m,
        wavelength_um=wavelength_um,
        pixel_pitch_um=pixel_pitch_um,
        check_sampling=False,
    )
    U_back = angular_spectrum_propagate(
        U_z,
        distance_m=-distance_m,
        wavelength_um=wavelength_um,
        pixel_pitch_um=pixel_pitch_um,
        check_sampling=False,
    )

    residual_amplitude = np.abs(np.abs(U_back) - np.abs(U_0))
    np.testing.assert_allclose(np.abs(U_back), np.abs(U_0), atol=1.0e-10)

    center_row = residual_amplitude[n_pixels // 2, :]

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.semilogy(coords_um, center_row + 1.0e-18, "o-")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("||U_back| - |U_0||")
    ax.set_title("Fresnel round-trip residual: center cut")
    fig.tight_layout()
    fig.savefig(_stats_output_path("fresnel_round_trip_center_cut.png"), dpi=120)
    plt.close(fig)


@pytest.mark.stats
def test_fresnel_gaussian_waist_grows_at_rayleigh_range() -> None:
    """A Gaussian beam propagated by one Rayleigh range must show w = w0 sqrt(2).

    The analytical Gaussian-beam result is
        w(z) = w0 * sqrt(1 + (z / z_R)^2),
    so at z = z_R the 1/e *intensity* radius is w0 sqrt(2). The angular-spectrum
    propagator must reproduce this to ~2% on a moderate grid (256 x 256, dx =
    10 um, lambda = 1.65 um, w0 = 200 um).
    """
    n_pixels = 256
    pixel_pitch_um = 10.0
    wavelength_um = 1.65
    waist_um = 200.0

    wavelength_m = wavelength_um * 1.0e-6
    waist_m = waist_um * 1.0e-6
    rayleigh_range_m = np.pi * waist_m**2 / wavelength_m

    coords_um = (np.arange(n_pixels, dtype=np.float64) - n_pixels / 2) * pixel_pitch_um
    x_um, y_um = np.meshgrid(coords_um, coords_um, indexing="xy")
    U_0 = np.exp(-((x_um**2 + y_um**2) / waist_um**2)).astype(np.complex128)

    # Code Review Issue 26: bypass the strict Nyquist check for this
    # Gaussian-waist stats test (spectrum is narrow enough that aliasing
    # is harmless at the Rayleigh range).
    U_z = angular_spectrum_propagate(
        U_0,
        distance_m=rayleigh_range_m,
        wavelength_um=wavelength_um,
        pixel_pitch_um=pixel_pitch_um,
        check_sampling=False,
    )

    # Fit the 1/e intensity radius from the radial profile along the +x axis.
    intensity_center_row = np.abs(U_z[n_pixels // 2, n_pixels // 2 :]) ** 2
    intensity_center_row /= intensity_center_row[0]
    radial_um = coords_um[n_pixels // 2 :] - coords_um[n_pixels // 2]
    # Linear interpolation on log(I) gives a clean estimate of the 1/e radius.
    log_intensity = np.log(intensity_center_row + 1.0e-30)
    # Find first crossing of -2 (since intensity ~ exp(-2 (r/w)^2) using
    # nullsim's amplitude form U = exp(-r^2/w^2)).
    crossing_index = int(np.argmax(log_intensity < -2.0))
    assert crossing_index > 0
    # Linear interpolate between samples crossing the threshold.
    r_lo, r_hi = radial_um[crossing_index - 1], radial_um[crossing_index]
    l_lo, l_hi = log_intensity[crossing_index - 1], log_intensity[crossing_index]
    fitted_waist_um = r_lo + (r_hi - r_lo) * (-2.0 - l_lo) / (l_hi - l_lo)

    expected_waist_um = waist_um * np.sqrt(2.0)
    relative_error = abs(fitted_waist_um - expected_waist_um) / expected_waist_um
    assert relative_error < 0.02, (
        f"Gaussian waist at z_R deviates: fit={fitted_waist_um:.2f} um, "
        f"expected={expected_waist_um:.2f} um, rel_err={relative_error:.4f}"
    )

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.plot(radial_um, intensity_center_row, "o", label="numerical |U(r, z_R)|^2")
    analytic = np.exp(-2.0 * (radial_um / expected_waist_um) ** 2)
    ax.plot(radial_um, analytic, "-", label=f"exp(-2 (r/w)^2), w={expected_waist_um:.0f}")
    ax.axhline(np.exp(-2.0), color="gray", linestyle="--", label="1/e^2")
    ax.axvline(fitted_waist_um, color="red", linestyle=":", label=f"fit w={fitted_waist_um:.1f}")
    ax.set_xlabel("r (um)")
    ax.set_ylabel("Normalized intensity")
    ax.set_title("Gaussian-beam waist at Rayleigh range")
    ax.legend()
    fig.tight_layout()
    fig.savefig(_stats_output_path("fresnel_gaussian_waist_at_zR.png"), dpi=120)
    plt.close(fig)
