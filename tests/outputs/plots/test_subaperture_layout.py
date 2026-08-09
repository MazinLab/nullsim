"""Smoke + regression tests for the subaperture_layout figure (KILO paperv3 Fig 3)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import nullsim  # noqa: F401  # ensures plot registration runs at import time
from nullsim.config import load_config
from nullsim.outputs.plots.subaperture_layout import (
    TelescopePanel,
    _build_pupil_template,
    _extract_n_sub,
    _extract_telescopes,
    plot_subaperture_layout,
)
from nullsim.pipeline import Geometry, SimulationState
from nullsim.physics.pupil_geometry import keck_pupil


_KECK_CONFIG = (
    Path(__file__).resolve().parents[3] / "examples" / "kilo_keck.toml"
)
_HWO_CONFIG = Path(__file__).resolve().parents[3] / "examples" / "hwo_space.toml"


def test_extract_keck_inputs_from_config() -> None:
    cfg = load_config(_KECK_CONFIG)
    tels = _extract_telescopes(cfg)
    n_sub = _extract_n_sub(cfg)

    assert len(tels) == 2  # Keck I + Keck II
    for t in tels:
        assert t.diameter_m == pytest.approx(10.0)
        assert t.pupil is not None
        assert len(t.pupil.spider_vanes) == 6
        assert t.pupil.central_obstruction_diameter_m == pytest.approx(2.4)
    assert n_sub == 12


def test_extract_hwo_n_sub_from_pupil_modes_without_ao() -> None:
    cfg = load_config(_HWO_CONFIG)

    assert _extract_n_sub(cfg) == 36


def test_extract_hwo_layout_uses_state_pupil_modes() -> None:
    cfg = load_config(_HWO_CONFIG)
    positions = np.array([[0.0, 0.0], [1.0, 0.5], [-0.4, 1.2]], dtype=np.float64)
    geometry = Geometry(diameter_m=8.0).add_component(
        "pupil_modes",
        {
            "layout": "nonredundant_circular",
            "mode_diameter_m": 0.25,
            "aperture_positions_pupil_m": positions,
        },
    )
    state = replace(SimulationState.empty(), geometry=geometry)

    tels = _extract_telescopes(cfg, state)

    assert len(tels) == 1
    assert tels[0].layout_name == "nonredundant_circular"
    assert tels[0].mode_diameter_m == pytest.approx(0.25)
    np.testing.assert_allclose(tels[0].mode_positions_m, positions)


def test_unknown_pupil_template_raises() -> None:
    with pytest.raises(ValueError, match="not supported"):
        _build_pupil_template("does_not_exist")


def test_circular_template_returns_none() -> None:
    assert _build_pupil_template("circular") is None
    assert _build_pupil_template("full_pupil") is None
    assert _build_pupil_template(None) is None


def test_render_keck_pdf(tmp_path: Path) -> None:
    tels = [
        TelescopePanel(name="Keck I", diameter_m=10.0, pupil=keck_pupil()),
        TelescopePanel(name="Keck II", diameter_m=10.0, pupil=keck_pupil()),
    ]
    out = tmp_path / "subaperture_layout.pdf"
    fig = plot_subaperture_layout(tels, n_sub=12, save_path=out)
    assert out.exists()
    assert out.stat().st_size > 1024  # not empty
    assert len(fig.axes) == 2  # one panel per telescope


def test_render_single_circular_panel(tmp_path: Path) -> None:
    tels = [TelescopePanel(name="Test", diameter_m=4.0, pupil=None)]
    out = tmp_path / "single.pdf"
    plot_subaperture_layout(tels, n_sub=4, save_path=out)
    assert out.exists()
