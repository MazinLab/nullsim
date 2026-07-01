"""Scientific validation tests with diagnostic plots.

These `@pytest.mark.stats` tests verify physics-model consistency for the
ported physics primitives. Failures here indicate the implementation has
drifted from the source physics, not a code-style or contract issue.

Diagnostic plots are written to ``tests/output/stats/`` so a human can
audit the curves when a tolerance-bounded check passes but looks suspicious.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend so tests work without a display.

import matplotlib.pyplot as plt
import numpy as np
import pytest

from nullsim.physics.photometry import photon_rate_per_m2_per_bin
from nullsim.physics.ruilier_cassaing import (
    UNOBSTRUCTED_GEOMETRIC_COUPLING_LIMIT,
    coupling_efficiency,
    geometric_coupling_limit,
)


_STATS_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "stats"


def _stats_output_path(filename: str) -> Path:
    """Return an absolute path for a diagnostic plot under tests/output/stats."""
    _STATS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return _STATS_OUTPUT_DIR / filename


# ---------------------------------------------------------------------------
# Photometry stats: magnitude-rate relation
# ---------------------------------------------------------------------------


@pytest.mark.stats
def test_photon_rate_jmag_zero_to_ten_spans_four_decades() -> None:
    """rate(0)/rate(10) ≈ 10^4 within float tolerance (Pogson scale)."""
    wavelength_um = 1.235
    bandwidth_um = 0.162

    rate_j0 = photon_rate_per_m2_per_bin(0.0, wavelength_um, bandwidth_um)
    rate_j5 = photon_rate_per_m2_per_bin(5.0, wavelength_um, bandwidth_um)
    rate_j10 = photon_rate_per_m2_per_bin(10.0, wavelength_um, bandwidth_um)

    # rate(0) > rate(5) > rate(10): strict ordering.
    assert rate_j0 > rate_j5 > rate_j10
    # Pogson: ratio per 5 magnitudes is exactly 100, per 10 magnitudes is 10000.
    assert rate_j0 / rate_j5 == pytest.approx(100.0, rel=1.0e-12)
    assert rate_j0 / rate_j10 == pytest.approx(1.0e4, rel=1.0e-12)


@pytest.mark.stats
def test_photon_rate_log_log_slope_is_negative_zero_point_four() -> None:
    """A magnitude vs log10(rate) plot must be a straight line with slope -0.4.

    Tolerance is generous (1e-10) because the photon-rate formula is an exact
    closed-form expression; any deviation here means the magnitude-to-flux
    relation drifted from the Pogson definition.
    """
    wavelength_um = 1.235
    bandwidth_um = 0.162
    magnitudes = np.linspace(-2.0, 12.0, 29)

    log_rates = np.array(
        [
            math.log10(
                photon_rate_per_m2_per_bin(mag, wavelength_um, bandwidth_um)
            )
            for mag in magnitudes
        ]
    )

    # Linear fit; the slope is the physical quantity under test.
    slope, intercept = np.polyfit(magnitudes, log_rates, deg=1)

    assert slope == pytest.approx(-0.4, abs=1.0e-10)

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.plot(magnitudes, log_rates, "o", label="photon_rate_per_m2_per_bin")
    ax.plot(
        magnitudes,
        slope * magnitudes + intercept,
        "-",
        label=f"linear fit slope={slope:.4f}",
    )
    ax.set_xlabel("Vega J magnitude")
    ax.set_ylabel("log10(rate [photons s^-1 m^-2 bin^-1])")
    ax.set_title("Pogson scale: log10(rate) vs magnitude")
    ax.legend()
    fig.tight_layout()
    fig.savefig(_stats_output_path("photon_rate_pogson_slope.png"), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Ruilier-Cassaing stats: coupling curves
# ---------------------------------------------------------------------------


@pytest.mark.stats
def test_coupling_efficiency_curve_passes_through_ruilier_limit() -> None:
    """coupling_efficiency(strehl) is the straight line eta_geo * strehl.

    The curve must pass through (0, 0) and (1, 0.8145), and be monotonically
    non-decreasing in Strehl. A diagnostic plot is saved so any departure from
    a perfect line (e.g. an accidental eta_geo factor that depends on strehl)
    is visible.
    """
    strehl = np.linspace(0.0, 1.0, 11)
    coupling = coupling_efficiency(strehl)

    assert coupling[0] == 0.0
    assert coupling[-1] == pytest.approx(
        UNOBSTRUCTED_GEOMETRIC_COUPLING_LIMIT,
        abs=1.0e-3,
    )
    # Strictly non-decreasing.
    assert np.all(np.diff(coupling) >= 0.0)

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.plot(strehl, coupling, "o-", label="coupling_efficiency")
    ax.axhline(
        UNOBSTRUCTED_GEOMETRIC_COUPLING_LIMIT,
        color="gray",
        linestyle="--",
        label="Ruilier 0.8145 limit",
    )
    ax.set_xlabel("Strehl ratio")
    ax.set_ylabel("On-axis SMF coupling efficiency")
    ax.set_title("Ruilier-Cassaing coupling vs Strehl (epsilon = 0)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        _stats_output_path("ruilier_cassaing_coupling_vs_strehl.png"),
        dpi=120,
    )
    plt.close(fig)


@pytest.mark.stats
def test_geometric_coupling_limit_drops_monotonically_with_obscuration() -> None:
    """eta_geo(epsilon) must monotonically decrease and start at 0.8145.

    A non-physical bump in the obscuration penalty would indicate the
    Gaussian overlap formula has been mistyped (e.g. missing 1 - epsilon^2
    denominator).
    """
    obscuration = np.linspace(0.0, 0.3, 7)
    limits = geometric_coupling_limit(obscuration)

    assert limits[0] == pytest.approx(
        UNOBSTRUCTED_GEOMETRIC_COUPLING_LIMIT,
        abs=1.0e-3,
    )
    # Monotonically decreasing (no non-physical bumps).
    assert np.all(np.diff(limits) < 0.0)

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.plot(obscuration, limits, "o-", label="geometric_coupling_limit")
    ax.axhline(
        UNOBSTRUCTED_GEOMETRIC_COUPLING_LIMIT,
        color="gray",
        linestyle="--",
        label="unobstructed 0.8145",
    )
    ax.set_xlabel("Linear central obscuration ratio epsilon")
    ax.set_ylabel("Geometric coupling limit eta_geo")
    ax.set_title("Ruilier-Cassaing eta_geo vs central obscuration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        _stats_output_path("ruilier_cassaing_eta_geo_vs_obscuration.png"),
        dpi=120,
    )
    plt.close(fig)
