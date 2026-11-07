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
    baselines_to_uv_with_origins,
    compute_uv_tracks,
    enumerate_subaperture_baselines,
)


_KECK_CONFIG = (
    Path(__file__).resolve().parents[3] / "examples" / "kilo_keck.toml"
)
_HWO_CONFIG = Path(__file__).resolve().parents[3] / "examples" / "hwo_space.toml"


def test_keck_baseline_counts_match_paperv3() -> None:
    cfg = load_config(_KECK_CONFIG)
    baselines, _ = _extract_baselines_from_config(cfg)
    n_intra = sum(1 for b in baselines if b.is_intra_telescope)
    n_inter = sum(1 for b in baselines if not b.is_intra_telescope)

    # Paperv3 Fig 5 caption: "132 intra-Keck + 144 inter-Keck baselines".
    assert len(baselines) == 276
    assert n_intra == 132
    assert n_inter == 144


def test_keck_inter_baseline_length_in_range() -> None:
    cfg = load_config(_KECK_CONFIG)
    baselines, _ = _extract_baselines_from_config(cfg)
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


def _two_telescope_layout() -> tuple[list[Baseline], np.ndarray]:
    tel = np.array([[0.0, 0.0, 0.0], [60.0, 50.0, 2.0]])
    sub = np.array([[-3.0, 1.0], [2.0, 4.0]])
    baselines = enumerate_subaperture_baselines(tel, sub, n_subapertures_per_telescope=2)
    return baselines, np.repeat(tel, 2, axis=0)


def test_rigid_pupil_offsets_do_not_rotate_with_hour_angle() -> None:
    """Review 2026-11-07 row 65: an intra-telescope baseline is (dE, dN)/lambda
    at every hour angle; only telescope origins foreshorten."""

    baselines, origins = _two_telescope_layout()
    wl = 1.25e-6
    ha = np.radians(np.linspace(-60.0, 60.0, 7))
    u, v = baselines_to_uv_with_origins(baselines, origins, ha, np.radians(-20.0), wl)
    intra = [k for k, b in enumerate(baselines) if b.is_intra_telescope]
    for k in intra:
        e, n, _ = baselines[k].vector_enu_m
        np.testing.assert_allclose(u[:, k], e / wl, rtol=1e-12)
        np.testing.assert_allclose(v[:, k], n / wl, rtol=1e-12)

    # Inter-telescope baseline = projected origin baseline + rigid pupil offset.
    k = next(k for k, b in enumerate(baselines) if (b.i, b.j) == (0, 3))
    origin_only = Baseline(i=0, j=3, vector_enu_m=(60.0, 50.0, 2.0), is_intra_telescope=False)
    u0, v0 = baselines_to_uv([origin_only], ha, np.radians(-20.0), wl)
    np.testing.assert_allclose(u[:, k], u0[:, 0] + (2.0 + 3.0) / wl, rtol=1e-12)
    np.testing.assert_allclose(v[:, k], v0[:, 0] + (4.0 - 1.0) / wl, rtol=1e-12)


def test_rigid_pupil_projection_reduces_to_full_projection() -> None:
    """One aperture per telescope, or a zenith transit: both conventions agree."""

    baselines, origins = _two_telescope_layout()
    wl = 1.0e-6
    lat = np.radians(19.8283)
    u, v = baselines_to_uv_with_origins(baselines, origins, 0.0, lat, wl, lat)
    u_full, v_full = baselines_to_uv(baselines, 0.0, lat, wl, lat)
    np.testing.assert_allclose(u, u_full, rtol=1e-12, atol=1e-6)
    np.testing.assert_allclose(v, v_full, rtol=1e-12, atol=1e-6)

    tel = np.array([[0.0, 0.0, 0.0], [60.0, 50.0, 2.0], [-30.0, 80.0, 0.0]])
    single = enumerate_subaperture_baselines(
        tel, np.zeros((1, 2)), n_subapertures_per_telescope=1
    )
    ha = np.radians(np.linspace(-45.0, 45.0, 5))
    u, v = baselines_to_uv_with_origins(single, tel, ha, np.radians(30.0), wl)
    u_full, v_full = baselines_to_uv(single, ha, np.radians(30.0), wl)
    np.testing.assert_allclose(u, u_full, rtol=1e-12)
    np.testing.assert_allclose(v, v_full, rtol=1e-12)


def test_render_keck_grid_pdf(tmp_path: Path) -> None:
    cfg = load_config(_KECK_CONFIG)
    baselines, origins = _extract_baselines_from_config(cfg)
    out = tmp_path / "uv_coverage_grid.pdf"
    fig = plot_uv_coverage_grid(
        baselines,
        wavelength_m=_DEFAULT_WAVELENGTHS_M["Y"],
        aperture_origins_enu_m=origins,
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
