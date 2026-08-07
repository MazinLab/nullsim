"""Checks for the HDC cross-correlation example config."""

from __future__ import annotations

import math
from pathlib import Path

from nullsim.cli import main
from nullsim.config import expand_sweeps, load_config
from nullsim.pipeline import run

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "examples" / "test" / "hdc_ccf_smoke.toml"


def test_hdc_ccf_smoke_example_loads_and_is_single_cell() -> None:
    config = load_config(str(CONFIG_PATH))
    cells = expand_sweeps(config)

    assert len(cells) == 1


def test_hdc_ccf_smoke_example_wires_m_dwarf_host_and_planet_rv() -> None:
    config = load_config(str(CONFIG_PATH))
    stages = {
        str(stage.id): stage.model_dump(mode="python")
        for stage in config.pipeline.stages
    }

    # The stellar template is resolved from the scene spectral type, so an
    # M-dwarf host is what selects the packaged M2V snapshot.
    scene = stages["scene"]
    assert scene["star_spectral_type"] == "M2V"
    assert scene["planet_radial_velocity_kms"] == 30.0

    hdc = stages["hdc"]
    assert hdc["type"] == "characterization.hdc_ccf"
    assert hdc["telluric_template"] == "maunakea"
    # resolving_power is left unset so the AWG R comes from the spectro payload.
    assert "resolving_power" not in hdc
    assert stages["characterization"]["type"] == "characterization.spectro"
    assert "hdc_ccf" in config.output.plots


def test_hdc_ccf_smoke_example_passes_cli_validate() -> None:
    assert main(["validate", str(CONFIG_PATH)]) == 0


def test_hdc_ccf_smoke_example_runs_and_produces_finite_ccf_snr() -> None:
    config = load_config(str(CONFIG_PATH))
    no_cache = config.model_copy(
        update={"cache": config.cache.model_copy(update={"enabled": False})}
    )

    result = run(no_cache)
    payload = next(
        component
        for name, component in result.state.results.components.items()
        if name.startswith("characterization.hdc_ccf")
    )

    assert math.isfinite(payload["ccf_snr"])
    assert payload["ccf_snr"] > 0.0
    assert math.isfinite(payload["ccf_snr_photon_limited"])
    # The AWG branch of characterization.spectro supplies the resolving power.
    assert payload["resolving_power_used"] == 5000.0
    assert payload["resolving_power_source"].endswith("awg_resolving_power")
    assert payload["planet_radial_velocity_kms"] == 30.0
    # M2V resolves to the packaged 3500 K PHOENIX snapshot, not the G or K one.
    assert "lte03500" in payload["star_template_provenance"]
    assert "Maunakea" in payload["telluric_template_provenance"]
    assert payload["n_channels_used"] >= 1
    # The RV scan peaks at the injected radial velocity; that is the whole
    # point of the cross-correlation, and a smoke test that never checked it
    # would pass on a scan that peaked in the stellar rest frame.
    rv_grid = payload["snr_vs_rv_kms"]
    peak_rv = float(rv_grid[int(payload["snr_vs_rv"].argmax())])
    spacing = float(rv_grid[1] - rv_grid[0])
    assert abs(peak_rv - 30.0) <= spacing
