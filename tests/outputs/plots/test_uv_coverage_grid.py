"""Smoke + regression tests for the uv_coverage_grid figure (KILO paperv3 Fig 5)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import nullsim  # noqa: F401  # ensures plot registration runs at import time
from nullsim.config import load_config
from nullsim.outputs import RunCollection
from nullsim.outputs.plots.uv_coverage_grid import (
    _extract_baselines_from_config,
    plot_uv_coverage_grid,
    uv_coverage_grid,
    _DEFAULT_WAVELENGTHS_M,
)
from nullsim.physics.uv_coverage import (
    Baseline,
    baselines_to_uv,
    compute_uv_tracks,
    enumerate_subaperture_baselines,
)


_KECK_CONFIG = (
    Path(__file__).resolve().parents[3] / "examples" / "kilo_keck.toml"
)
_HWO_CONFIG = Path(__file__).resolve().parents[3] / "examples" / "hwo_space.toml"


def test_keck_baseline_counts_match_paperv3() -> None:
    cfg = load_config(_KECK_CONFIG)
    baselines = _extract_baselines_from_config(cfg)
    n_intra = sum(1 for b in baselines if b.is_intra_telescope)
    n_inter = sum(1 for b in baselines if not b.is_intra_telescope)

    # Paperv3 Fig 5 caption: "132 intra-Keck + 144 inter-Keck baselines".
    assert len(baselines) == 276
    assert n_intra == 132
    assert n_inter == 144


def test_keck_inter_baseline_length_in_range() -> None:
    cfg = load_config(_KECK_CONFIG)
    baselines = _extract_baselines_from_config(cfg)
    inter = [b.length_m for b in baselines if not b.is_intra_telescope]
    # Inter-Keck baselines span ~80–96 m per the paper.
    assert min(inter) > 75.0
    assert max(inter) < 100.0


def test_enumerate_two_telescope_one_sub() -> None:
    tel = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    sub = np.array([[0.0, 0.0]])
    baselines = enumerate_subaperture_baselines(tel, sub, n_subapertures_per_telescope=1)
    assert len(baselines) == 1
    b = baselines[0]
    assert not b.is_intra_telescope
    assert b.length_m == pytest.approx(10.0)


def test_baselines_to_uv_zero_at_zenith_zero_baseline() -> None:
    b = Baseline(i=0, j=1, vector_enu_m=(0.0, 0.0, 0.0), is_intra_telescope=False)
    u, v = baselines_to_uv([b], 0.0, np.pi / 2, 1e-6)
    assert u[0] == pytest.approx(0.0)
    assert v[0] == pytest.approx(0.0)


def test_compute_uv_tracks_shape() -> None:
    b = Baseline(i=0, j=1, vector_enu_m=(50.0, 0.0, 0.0), is_intra_telescope=False)
    u, v = compute_uv_tracks([b], 30.0, -30.0, 30.0, n_time_steps=60, wavelength_m=1e-6)
    assert u.shape == (60, 1)
    assert v.shape == (60, 1)


def test_render_keck_grid_pdf(tmp_path: Path) -> None:
    cfg = load_config(_KECK_CONFIG)
    baselines = _extract_baselines_from_config(cfg)
    out = tmp_path / "uv_coverage_grid.pdf"
    fig = plot_uv_coverage_grid(
        baselines,
        wavelength_m=_DEFAULT_WAVELENGTHS_M["Y"],
        save_path=out,
    )
    assert out.exists()
    assert out.stat().st_size > 5000  # rasterized scatter, real content
    # 4 declinations × 3 integration configs = 12 panels
    assert len(fig.axes) == 12


def test_registered_plot_writes_placeholder_for_single_aperture(
    tmp_path: Path,
) -> None:
    cfg = load_config(_HWO_CONFIG)
    collection = RunCollection(config=cfg, records=(), output_dir=tmp_path)

    paths = uv_coverage_grid(collection, tmp_path, ("png",))

    assert len(paths) == 1
    assert paths[0].exists()
    assert paths[0].stat().st_size > 3_000
