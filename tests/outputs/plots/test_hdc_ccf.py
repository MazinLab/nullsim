"""Tests for the HDC cross-correlation diagnostic plot output."""

from __future__ import annotations

import logging

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from nullsim.outputs import RunCollection, RunRecord
from matplotlib import pyplot as plt

from nullsim.outputs.plots.hdc_ccf import (
    _find_payload,
    _snr_squared_share,
    _variance_rate_per_channel,
    hdc_ccf,
    plot_hdc_ccf,
)
from nullsim.pipeline import Results, RunResult, SimulationState


def _payload() -> dict:
    centers = np.linspace(1.50, 1.75, 24)
    planet = 1.0e-3 * (1.0 + 0.3 * np.sin(np.linspace(0.0, 12.0, centers.size)))
    leak = 5.0e-2 * np.ones(centers.size)
    sky = 1.0e-4 * np.ones(centers.size)
    dark = 2.0e-3
    variance = planet + leak + sky + dark
    # Signed weights: an absorption-dominated filtered template goes negative,
    # and the plot must not clamp or take absolute values.
    weights = np.sin(np.linspace(0.0, 12.0, centers.size)) * planet / variance
    r_scan = np.geomspace(100.0, 30000.0, 10)
    snr_vs_r = np.linspace(0.5, 9.0, r_scan.size)
    snr_vs_r_photon = snr_vs_r * 1.08
    # Infeasible low-R points legitimately record NaN on narrow bands.
    snr_vs_r[:2] = np.nan
    snr_vs_r_photon[:2] = np.nan
    rv_kms = np.linspace(-60.0, 60.0, 25)
    snr_vs_rv = 9.0 * np.exp(-0.5 * ((rv_kms - 12.0) / 6.0) ** 2) - 0.4
    return {
        "ccf_snr": 8.4,
        "ccf_snr_photon_limited": 9.1,
        "ccf_signal_counts": 4.2e4,
        "ccf_photon_var_counts": 2.1e7,
        "ccf_systematic_var_counts": 3.0e6,
        "systematic_ccf_leakage": 12.0,
        "ccf_systematic_rejection": 2.4e-3,
        # Self-subtraction can legitimately exceed 1; never rendered capped.
        "self_subtraction_factor": 1.35,
        "n_effective_lines": 143.7,
        "n_channels_used": 22,
        "n_channels": 24,
        "telluric_masked_fraction": 0.083,
        "ccf_channel_centers_um": centers,
        "ccf_planet_rate_hz_per_channel": planet,
        "ccf_leak_rate_hz_per_channel": leak,
        "ccf_sky_rate_hz_per_channel": sky,
        "ccf_variance_rate_hz_per_channel": variance,
        "dark_rate_hz_per_channel": dark,
        "template_weights_per_channel": weights,
        "snr_vs_resolving_power": snr_vs_r,
        "snr_vs_resolving_power_photon_limited": snr_vs_r_photon,
        "snr_vs_resolving_power_r": r_scan,
        "snr_vs_rv": snr_vs_rv,
        "snr_vs_rv_kms": rv_kms,
        "integration_time_s": 3600.0,
        "resolving_power_used": 3000.0,
        "resolving_power_source": "characterization.spectro AWG",
        "planet_radial_velocity_kms": 12.0,
        "with_fringe_tracking": True,
    }


def _record(components: dict, run_dir: Path) -> RunRecord:
    state = SimulationState.empty({"wavelength_grid_um": (1.5, 1.75)})
    state = SimulationState(
        scene=state.scene,
        geometry=state.geometry,
        field=state.field,
        opd_budget=state.opd_budget,
        throughput=state.throughput,
        wavefront=state.wavefront,
        photon_rates=state.photon_rates,
        results=Results(components=components),
    )
    return RunRecord(
        config=SimpleNamespace(),
        result=RunResult(
            state=state,
            diagnostics={},
            timings=(),
            cache_hit_count=0,
            config_hash="plot",
        ),
        run_dir=run_dir,
        coord_tuple=(),
    )


def test_plot_hdc_ccf_title_carries_headline_scalars() -> None:
    fig = plot_hdc_ccf(_payload())

    title = fig._suptitle.get_text()
    assert "8.4" in title
    assert "3000" in title or "3.00e+03" in title
    assert "144" in title or "143.7" in title
    plt.close(fig)


def test_rv_panel_is_labeled_a_detection_statistic_not_a_correlation() -> None:
    """snr_vs_rv is an unbounded sigma-like matched-filter score. Labeling it
    as a correlation coefficient would imply a [-1, 1] bound it does not have.
    """

    fig = plot_hdc_ccf(_payload())

    label = fig.axes[3].get_ylabel().lower()
    assert "sigma" in label
    assert "detection statistic" in label
    assert "correlation" not in label
    plt.close(fig)


def test_negative_template_weights_survive_to_the_plotted_bars() -> None:
    """Absorption-dominated filtered templates give signed weights; the bar
    panel must plot them signed, with no abs() or clamp at zero."""

    payload = _payload()
    weights = np.asarray(payload["template_weights_per_channel"], dtype=float)
    assert np.any(weights < 0.0), "fixture must exercise the signed case"

    fig = plot_hdc_ccf(payload)
    heights = np.array(
        [patch.get_height() for patch in fig.axes[1].patches], dtype=float
    )

    assert np.allclose(np.sort(heights), np.sort(weights))
    plt.close(fig)


def test_snr_squared_share_weights_by_variance_not_by_weight_alone() -> None:
    """Channel i contributes w_i^2 v_i = Ptilde_i^2 / v_i to the photon-limited
    SNR^2, so a channel with the same weight but a larger variance rate carries
    a larger share. Ranking by w_i^2 alone would call them equal."""

    weights = np.array([1.0, 1.0, -2.0])
    variance = np.array([1.0, 4.0, 1.0])

    share = _snr_squared_share(weights, variance)

    assert share is not None
    assert share == pytest.approx(np.array([1.0, 4.0, 4.0]) / 9.0)
    assert share[1] > share[0]


def test_variance_rate_falls_back_to_the_stage_definition() -> None:
    """Payloads predating ccf_variance_rate_hz_per_channel rebuild v exactly:
    the stage charges one fixed dark rate to every channel."""

    payload = _payload()
    expected = np.asarray(payload.pop("ccf_variance_rate_hz_per_channel"), dtype=float)

    rebuilt = _variance_rate_per_channel(payload, expected.size)

    assert rebuilt is not None
    assert rebuilt == pytest.approx(expected)
    # No dark rate and no published variance: report absence, never guess.
    payload.pop("dark_rate_hz_per_channel")
    assert _variance_rate_per_channel(payload, expected.size) is None


def test_zero_rate_channels_do_not_break_the_log_panels() -> None:
    """Empty channels are physical (masked tellurics, band edges). A log axis
    must skip them rather than raise or drop the whole panel."""

    payload = _payload()
    for key in (
        "ccf_planet_rate_hz_per_channel",
        "ccf_leak_rate_hz_per_channel",
        "ccf_sky_rate_hz_per_channel",
    ):
        values = np.asarray(payload[key], dtype=float).copy()
        values[:3] = 0.0
        payload[key] = values

    fig = plot_hdc_ccf(payload)
    line = fig.axes[0].get_lines()[0]

    assert np.all(np.isnan(line.get_ydata()[:3]))
    plt.close(fig)


def test_photon_limited_overlay_is_a_scanned_curve_not_a_horizontal_line() -> None:
    """The photon limit is R dependent, so it is drawn from
    snr_vs_resolving_power_photon_limited across the axis."""

    payload = _payload()
    expected = np.asarray(
        payload["snr_vs_resolving_power_photon_limited"], dtype=float
    )

    fig = plot_hdc_ccf(payload)
    curves = [
        line.get_ydata()
        for line in fig.axes[2].get_lines()
        if np.asarray(line.get_ydata()).size == expected.size
    ]

    assert any(
        np.allclose(curve, expected, equal_nan=True) for curve in curves
    ), "photon-limited scan is not plotted as a curve"
    plt.close(fig)


def test_photon_limited_falls_back_to_an_operating_point_marker() -> None:
    """Older payloads carry only the scalar photon limit: mark the operating
    point rather than drawing a line that asserts R independence."""

    payload = _payload()
    payload.pop("snr_vs_resolving_power_photon_limited")

    fig = plot_hdc_ccf(payload)
    labels = [text.get_text() for text in fig.axes[2].get_legend().get_texts()]

    assert any("photon limit at R used" in label for label in labels)
    plt.close(fig)


def test_find_payload_matches_stage_component_name() -> None:
    payload = _payload()
    record = _record({"characterization.hdc_ccf_1": payload}, Path("."))

    found = _find_payload(record)

    assert found is not None
    assert found["ccf_snr"] == payload["ccf_snr"]
    assert found["n_effective_lines"] == payload["n_effective_lines"]


def test_registered_hdc_ccf_writes_plot(tmp_path: Path) -> None:
    record = _record({"characterization.hdc_ccf": _payload()}, tmp_path)
    collection = RunCollection(
        config=SimpleNamespace(),
        records=(record,),
        output_dir=tmp_path,
    )

    written = hdc_ccf(collection, tmp_path, ("png",))

    out = tmp_path / "plots" / "hdc_ccf.png"
    assert out in written
    assert out.exists()
    assert out.stat().st_size > 3_000


def test_hdc_ccf_skips_quietly_without_component(tmp_path: Path) -> None:
    record = _record({"sensitivity.detection_curve": {"contrast": 1.0}}, tmp_path)
    collection = RunCollection(
        config=SimpleNamespace(),
        records=(record,),
        output_dir=tmp_path,
    )

    assert hdc_ccf(collection, tmp_path, ("png",)) == ()
    assert not (tmp_path / "plots" / "hdc_ccf.png").exists()


def test_plot_hdc_ccf_tolerates_nan_and_empty_scans() -> None:
    payload = _payload()
    payload["snr_vs_resolving_power"] = np.full(10, np.nan)
    payload["snr_vs_rv"] = np.empty(0)
    payload["snr_vs_rv_kms"] = np.empty(0)

    payload["snr_vs_resolving_power_photon_limited"] = np.full(10, np.nan)

    fig = plot_hdc_ccf(payload)

    assert len(fig.axes) >= 4
    plt.close(fig)


def _spectro_component(*, converged: bool) -> dict:
    """Minimal spectro payload the shared convergence predicate can judge."""

    return {
        "optimization_success": converged,
        "enforce_null_constraint": True,
        "optimization_constraint_margin": 1.0e-6 if converged else -1.0,
        "null_floor": 3.0e-5,
    }


def test_hdc_ccf_plot_skips_when_upstream_spectro_did_not_converge(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Same contract the characterization_summary table applies to its columns."""

    payload = _payload()
    payload["upstream_spectro_component"] = "characterization.spectro"
    record = _record(
        {
            "characterization.spectro": _spectro_component(converged=False),
            "characterization.hdc_ccf": payload,
        },
        tmp_path,
    )
    collection = RunCollection(
        config=SimpleNamespace(), records=(record,), output_dir=tmp_path
    )

    with caplog.at_level(logging.WARNING):
        written = hdc_ccf(collection, tmp_path, ("png",))

    assert written == ()
    assert not (tmp_path / "plots" / "hdc_ccf.png").exists()
    assert "did not" in caplog.text and "converge" in caplog.text


def test_hdc_ccf_plot_renders_when_upstream_spectro_converged(
    tmp_path: Path,
) -> None:
    payload = _payload()
    payload["upstream_spectro_component"] = "characterization.spectro"
    record = _record(
        {
            "characterization.spectro": _spectro_component(converged=True),
            "characterization.hdc_ccf": payload,
        },
        tmp_path,
    )
    collection = RunCollection(
        config=SimpleNamespace(), records=(record,), output_dir=tmp_path
    )

    written = hdc_ccf(collection, tmp_path, ("png",))

    assert (tmp_path / "plots" / "hdc_ccf.png") in written


def test_hdc_ccf_plot_renders_when_upstream_cannot_be_identified(
    tmp_path: Path,
) -> None:
    """No upstream key, or a name not on the state: render rather than hide."""

    payload = _payload()
    payload["upstream_spectro_component"] = "characterization.spectro.absent"
    record = _record({"characterization.hdc_ccf": payload}, tmp_path)
    collection = RunCollection(
        config=SimpleNamespace(), records=(record,), output_dir=tmp_path
    )

    assert (tmp_path / "plots" / "hdc_ccf.png") in hdc_ccf(
        collection, tmp_path, ("png",)
    )


def test_rv_panel_warns_when_the_peak_is_not_a_velocity_measurement() -> None:
    """A labelled peak reads as a measurement whether or not it is one.

    The scan maximizes the match to the OBSERVED spectrum, so a leak the
    continuum filter cannot remove moves the peak. When the stage grades the
    localization as anything but reliable, the panel has to say so on its face.
    """

    payload = _payload()
    payload["rv_leak_contamination"] = 3.5
    payload["rv_leak_contamination_effective"] = 3.5
    payload["rv_scan_localization_grade"] = "unreliable"
    fig = plot_hdc_ccf(payload)
    texts = [t.get_text() for t in fig.axes[3].texts]
    assert any("unreliable" in t for t in texts)
    assert any("3.5" in t for t in texts)
    plt.close(fig)

    payload["rv_scan_localization_grade"] = "reliable"
    fig = plot_hdc_ccf(payload)
    assert not [
        t for t in fig.axes[3].texts if "may not be the true velocity" in t.get_text()
    ]
    plt.close(fig)
