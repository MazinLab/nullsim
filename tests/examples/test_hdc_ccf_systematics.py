"""Checks for the systematics-dominated HDC characterization example.

The smoke config is photon limited, so cross-correlation has nothing to
reject and ties broadband. This example is tuned into the opposite regime:
the leaked-starlight systematic dominates the variance budget, and the
matched filter beats the broadband band SNR because the continuum filter
throws away the smooth part of the leak that carries that systematic.
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


def test_systematics_example_is_systematic_dominated_and_hdc_wins() -> None:
    result = _run_no_cache()
    hdc = _component(result, "characterization.hdc_ccf")
    spectro = _component(result, "characterization.spectro")

    # Systematics dominate the cross-correlation variance budget: without
    # this the comparison below would just be two photon-limited numbers.
    assert hdc["ccf_systematic_var_counts"] > hdc["ccf_photon_var_counts"]

    # HDC digs the planet out of the systematic wall that caps broadband.
    assert math.isfinite(hdc["ccf_snr"])
    assert hdc["ccf_snr"] > 0.0
    assert hdc["ccf_snr"] > spectro["snr"]
