"""Tests for the HDC cross-correlation diagnostic plot output."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from nullsim.outputs import RunCollection, RunRecord
from nullsim.outputs.plots.hdc_ccf import (
    _find_payload,
    hdc_ccf,
    plot_hdc_ccf,
)
from nullsim.pipeline import Results, RunResult, SimulationState


def _payload() -> dict:
    centers = np.linspace(1.50, 1.75, 24)
    planet = 1.0e-3 * (1.0 + 0.3 * np.sin(np.linspace(0.0, 12.0, centers.size)))
    leak = 5.0e-2 * np.ones(centers.size)
    sky = 1.0e-4 * np.ones(centers.size)
    weights = planet / leak
    r_scan = np.geomspace(100.0, 30000.0, 10)
    snr_vs_r = np.linspace(0.5, 9.0, r_scan.size)
    # Infeasible low-R points legitimately record NaN on narrow bands.
    snr_vs_r[:2] = np.nan
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
        "template_weights_per_channel": weights,
        "snr_vs_resolving_power": snr_vs_r,
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


def test_plot_hdc_ccf_renders_four_panels() -> None:
    fig = plot_hdc_ccf(_payload())

    # Four panels plus the twinned axis on the spectrum panel.
    assert len(fig.axes) >= 5


def test_plot_hdc_ccf_title_carries_headline_scalars() -> None:
    fig = plot_hdc_ccf(_payload())

    title = fig._suptitle.get_text()
    assert "8.4" in title
    assert "3000" in title or "3.00e+03" in title
    assert "144" in title or "143.7" in title


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

    fig = plot_hdc_ccf(payload)

    assert len(fig.axes) >= 4
