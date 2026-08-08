"""Checks for the detection-mode HDC example config."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nullsim.cli import main
from nullsim.config import expand_sweeps, load_config
from nullsim.pipeline import run

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "examples" / "test" / "hdc_detection_smoke.toml"


def test_hdc_detection_smoke_example_loads_and_is_single_cell() -> None:
    config = load_config(str(CONFIG_PATH))
    cells = expand_sweeps(config)

    assert len(cells) == 1


def test_hdc_detection_smoke_example_inherits_the_broadband_comparison_basis() -> None:
    config = load_config(str(CONFIG_PATH))
    stages = {
        str(stage.id): stage.model_dump(mode="python")
        for stage in config.pipeline.stages
    }

    scene = stages["scene"]
    assert scene["star_spectral_type"] == "M2V"
    assert scene["planet_radial_velocity_kms"] == 30.0

    hdc = stages["hdc_detection"]
    assert hdc["type"] == "performance.hdc"
    assert hdc["telluric_template"] == "maunakea"
    # Leaving both unset is the stage's fairness guarantee: the CCF curve and
    # the broadband curve it is divided by must describe the same observation.
    # Setting either here would either agree redundantly or be refused.
    assert "integration_time_s" not in hdc
    assert "target_snr" not in hdc

    # The two upstream components performance.hdc reads.
    assert stages["performance"]["type"] == "performance.standard"
    assert stages["sensitivity"]["type"] == "sensitivity.detection_curve"
    assert "detection_space" in config.output.plots


def test_hdc_detection_smoke_example_passes_cli_validate() -> None:
    assert main(["validate", str(CONFIG_PATH)]) == 0


def test_hdc_detection_smoke_example_runs_and_produces_a_ccf_contrast_curve() -> None:
    config = load_config(str(CONFIG_PATH))
    no_cache = config.model_copy(
        update={"cache": config.cache.model_copy(update={"enabled": False})}
    )

    result = run(no_cache)
    payload = next(
        component
        for name, component in result.state.results.components.items()
        if name.startswith("performance.hdc")
    )
    broadband_payload = next(
        component
        for name, component in result.state.results.components.items()
        if name.startswith("performance.standard")
    )

    separations = np.asarray(payload["separations_mas"], dtype=float)
    ccf = np.asarray(payload["ccf_contrast_at_target"], dtype=float)
    broadband = np.asarray(payload["broadband_contrast_at_target"], dtype=float)
    gain = np.asarray(payload["ccf_contrast_gain"], dtype=float)

    assert separations.shape == ccf.shape == broadband.shape == gain.shape
    assert np.array_equal(separations, np.asarray(load_separations(), dtype=float))

    # NaN anchors are legal in general (a non-detection is a real outcome),
    # but the shipped config is chosen so every anchor detects -- which is what
    # its header claims -- so assert the stronger contract here. A NaN
    # appearing means the config drifted out of the regime it advertises.
    assert np.all(np.isfinite(ccf)), (
        "the shipped config claims every anchor detects; non-detections at "
        f"{np.asarray(separations)[~np.isfinite(ccf)]} mas mean it drifted"
    )
    assert np.all(ccf > 0.0)
    assert np.all(np.isfinite(broadband))
    assert np.all(np.isfinite(gain))

    # The comparison basis is inherited, not configured. Pinning it against
    # the broadband payload is what proves the ratio is apples to apples.
    assert payload["comparison_time_s"] == (
        broadband_payload["contrast_curve_reference_time_s"]
    )
    assert payload["comparison_target_snr"] == broadband_payload["target_snr"]

    # This config exists to show a systematics-limited regime where the
    # matched filter wins; a photon-limited pipeline would put the gain below
    # 1 and the overlay would be pointless.
    assert payload["eps_systematic_used"] > 0.0
    assert payload["with_fringe_tracking_used"] is False
    assert np.all(gain > 1.0)


def load_separations() -> list[float]:
    """The sensitivity anchors the config declares, read back from the TOML."""

    config = load_config(str(CONFIG_PATH))
    for stage in config.pipeline.stages:
        if str(stage.id) == "sensitivity":
            return list(stage.model_dump(mode="python")["separations_mas"])
    raise AssertionError("hdc_detection_smoke.toml has no sensitivity stage")
