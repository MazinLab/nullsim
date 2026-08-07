"""Checks for the systematics-dominated HDC characterization example.

The smoke config is photon limited, so cross-correlation has nothing to
reject and loses to broadband (ccf_snr 70.49 against band SNR 387.41) --
the matched filter discards the continuum photons the band SNR counts.
This example is tuned into the opposite regime: the leaked-starlight
systematic dominates the variance budget, and the matched filter now beats
the band SNR because the continuum filter throws away the smooth part of
the leak that carries that systematic.

The comparison is only meaningful if both sides see the same observation,
so the fairness controls are asserted alongside the two headline
inequalities: equal integration time, and an HDC noise model inherited from
the spectro payload rather than set independently.
"""

from __future__ import annotations

import math
from pathlib import Path

from nullsim.cli import main
from nullsim.config import expand_sweeps, load_config
from nullsim.pipeline import run

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "examples" / "test" / "hdc_ccf_systematics.toml"


def _run_no_cache():
    config = load_config(str(CONFIG_PATH))
    no_cache = config.model_copy(
        update={"cache": config.cache.model_copy(update={"enabled": False})}
    )
    return run(no_cache)


def _component(result, prefix: str) -> dict:
    return next(
        component
        for name, component in result.state.results.components.items()
        if name.startswith(prefix)
    )


def test_hdc_ccf_systematics_example_loads_and_is_single_cell() -> None:
    config = load_config(str(CONFIG_PATH))

    assert len(expand_sweeps(config)) == 1


def test_hdc_ccf_systematics_example_passes_cli_validate() -> None:
    assert main(["validate", str(CONFIG_PATH)]) == 0


def test_systematics_example_compares_hdc_and_broadband_fairly() -> None:
    """The controls that make the SNR comparison mean anything.

    Both numbers must describe one observation under one noise model. The
    HDC stage takes eps and the fringe-tracking branch off the spectro
    payload, so a config that set them independently -- or an integration
    time that drifted away from the detector's -- would hand HDC a quiet
    advantage and the headline win would be an artifact.
    """
    result = _run_no_cache()
    hdc = _component(result, "characterization.hdc_ccf")
    spectro = _component(result, "characterization.spectro")
    detector = _component(result, "detector.mkid")

    # characterization.spectro integrates for the detector's exposure.
    assert hdc["integration_time_s"] == detector["integration_time_s"]
    assert hdc["eps_systematic_used"] == spectro["eps_systematic"]
    assert hdc["with_fringe_tracking"] == spectro["with_fringe_tracking"]
    assert hdc["servo_bw_hz_used"] == spectro["servo_bw_hz"]
    assert hdc["upstream_spectro_component"].startswith("characterization.spectro")


def test_systematics_example_is_systematic_dominated_and_hdc_wins() -> None:
    result = _run_no_cache()
    hdc = _component(result, "characterization.hdc_ccf")
    spectro = _component(result, "characterization.spectro")

    # Systematics dominate the cross-correlation variance budget: without
    # this the comparison below would just be two photon-limited numbers.
    # Achieved ratio is 3.54; the floor leaves room for retuning but fails
    # loudly if the config drifts back toward the photon-limited regime.
    assert hdc["ccf_systematic_var_counts"] > 2.0 * hdc["ccf_photon_var_counts"]

    # HDC digs the planet out of the systematic wall that caps broadband.
    # Achieved ratio is 10.65.
    assert math.isfinite(hdc["ccf_snr"])
    assert hdc["ccf_snr"] > 0.0
    assert hdc["ccf_snr"] > 3.0 * spectro["snr"]
