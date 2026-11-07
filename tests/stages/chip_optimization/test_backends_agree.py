"""Cross-backend agreement tests for chip_optimization (SPEC §15.3 Pass 3).

The xfail-strict test below is a tripwire that documents the KILO source path
of the behavioral reference for ``TestClementsBackendsAgree``. It is NOT a
verbatim copy of KILO test code — KILO tests the kernel builders directly,
while nullsim's tests run the full optimizer-stage pipeline. The executable
nullsim-native cross-backend test below carries the same intent (scipy and
torch must agree to within numerical precision, regardless of which one is
installed).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from nullsim.pipeline import (
    Field,
    Geometry,
    RunContext,
    Scene,
    SimulationState,
    Throughput,
)


def _context(tmp_path: Path, *, instance_id: str, seed: int = 0) -> RunContext:
    return RunContext(
        rng=np.random.default_rng(seed),
        config_hash=f"chip-opt-backends-{instance_id}",
        sweep_coord_tuple=(),
        instance_id=instance_id,
        run_dir=tmp_path,
    )


def _uniform_state(n_modes: int, wl_um: float = 1.65) -> SimulationState:
    wavelength_grid_um = (wl_um,)
    amplitudes = np.ones((n_modes, 1, 1), dtype=np.complex128) / np.sqrt(n_modes)
    state = SimulationState.empty({"wavelength_grid_um": wavelength_grid_um})
    field = Field(
        amplitudes=amplitudes,
        wavelength_grid_um=wavelength_grid_um,
        mode_labels=tuple(f"mode_{i}" for i in range(n_modes)),
    )
    throughput = Throughput(
        wavelength_grid_um=wavelength_grid_um,
        transmission_fraction=(1.0,),
    )
    return replace(state, field=field, throughput=throughput)


def _broadband_kernel_state() -> SimulationState:
    n_modes = 4
    wavelength_grid_um = (1.20, 1.25, 1.30)
    state = SimulationState.empty({"wavelength_grid_um": wavelength_grid_um})
    field = Field(
        amplitudes=np.ones((n_modes, len(wavelength_grid_um), 1), dtype=np.complex128)
        / np.sqrt(n_modes),
        wavelength_grid_um=wavelength_grid_um,
        mode_labels=tuple(f"mode_{i}" for i in range(n_modes)),
    )
    throughput = Throughput(
        wavelength_grid_um=wavelength_grid_um,
        transmission_fraction=tuple(1.0 for _ in wavelength_grid_um),
    )
    geometry = Geometry(
        aperture_positions_pupil_m=np.array(
            [[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
            dtype=np.float64,
        ),
        effective_collecting_area_m2=4.0,
    )
    return replace(state, field=field, throughput=throughput, geometry=geometry)


def _agreement_ok(n1: float, n2: float, *, rtol: float, abs_floor: float) -> bool:
    """Treat two null values as agreeing when EITHER they are both deeper
    than ``abs_floor`` (numerical noise dominates differences below that)
    OR they agree within ``rtol`` relative tolerance.
    """

    if abs(n1) < abs_floor and abs(n2) < abs_floor:
        return True
    denom = max(abs(n1), abs(n2), 1e-300)
    return abs(n1 - n2) / denom < rtol


def _run_backend(
    backend: str,
    *,
    n_modes: int,
    n_dark_ports: int,
    seed: int,
    max_iter: int,
    tmp_path: Path,
) -> float:
    context = _context(tmp_path, instance_id=f"chip_opt_{backend}")
    if backend == "scipy":
        from nullsim.stages.chip_optimization.scipy import ChipOptimizationScipy

        stage_cls = ChipOptimizationScipy
    elif backend == "torch":
        from nullsim.stages.chip_optimization.torch import ChipOptimizationTorch

        stage_cls = ChipOptimizationTorch
    else:  # pragma: no cover — defensive
        raise ValueError(f"unknown backend {backend!r}")

    stage = stage_cls(
        {
            "n_modes": n_modes,
            "n_bright_ports": n_modes - n_dark_ports,
            "n_dark_ports": n_dark_ports,
            "wl_design_um": 1.65,
            "max_iter": max_iter,
            "seed": seed,
        },
        context,
    )
    result = stage.apply(_uniform_state(n_modes=n_modes))
    matches = [
        payload
        for name, payload in result.results.components.items()
        if name.startswith("chip_optimization")
    ]
    assert len(matches) == 1
    return float(matches[0]["optimization_loss_final"])


def _run_backend_with_params(
    backend: str,
    *,
    params: dict[str, object],
    state: SimulationState,
    tmp_path: Path,
) -> float:
    context = _context(tmp_path, instance_id=f"chip_opt_{backend}")
    if backend == "scipy":
        from nullsim.stages.chip_optimization.scipy import ChipOptimizationScipy

        stage_cls = ChipOptimizationScipy
    elif backend == "torch":
        from nullsim.stages.chip_optimization.torch import ChipOptimizationTorch

        stage_cls = ChipOptimizationTorch
    else:  # pragma: no cover — defensive
        raise ValueError(f"unknown backend {backend!r}")

    result = stage_cls(params, context).apply(state)
    matches = [
        payload
        for name, payload in result.results.components.items()
        if name.startswith("chip_optimization")
    ]
    assert len(matches) == 1
    return float(matches[0]["optimization_loss_final"])


def _torch_available() -> bool:
    try:
        import torch  # type: ignore  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    # Importing the stage module also triggers registry side effects.
    try:
        from nullsim.stages.chip_optimization import torch as _backend  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


@pytest.mark.stats
@pytest.mark.parametrize("n_modes", [4, 8])
def test_backends_agree_on_final_null_depth(n_modes: int, tmp_path: Path) -> None:
    """For each m in {4, 8}, scipy + (torch if importable) starting from
    seed=0 must agree on the final null depth within 1% relative
    tolerance. Backends not installed are simply skipped from the comparison.
    """

    n_dark_ports = max(1, n_modes // 2)
    seed = 0
    max_iter = 200

    nulls: dict[str, float] = {}
    nulls["scipy"] = _run_backend(
        "scipy",
        n_modes=n_modes,
        n_dark_ports=n_dark_ports,
        seed=seed,
        max_iter=max_iter,
        tmp_path=tmp_path,
    )
    if _torch_available():
        nulls["torch"] = _run_backend(
            "torch",
            n_modes=n_modes,
            n_dark_ports=n_dark_ports,
            seed=seed,
            max_iter=max_iter,
            tmp_path=tmp_path,
        )

    if len(nulls) < 2:
        pytest.skip(f"Need >=2 backends to compare; only {list(nulls)} available.")

    # Both backends finding nulls deeper than 1e-8 are operationally
    # indistinguishable for any realistic instrument; we only require
    # 1% relative agreement when at least one backend is shallower.
    abs_floor = 1.0e-8
    rtol = 1.0e-2
    backends = list(nulls)
    for i, b1 in enumerate(backends):
        for b2 in backends[i + 1 :]:
            n1, n2 = nulls[b1], nulls[b2]
            assert _agreement_ok(n1, n2, rtol=rtol, abs_floor=abs_floor), (
                f"backend nulls disagree at m={n_modes}: {b1}={n1:.6e}, {b2}={n2:.6e}"
            )


@pytest.mark.stats
def test_backends_agree_on_broadband_kernel_objective_above_abs_floor(
    tmp_path: Path,
) -> None:
    """Exercise the relative-tolerance branch on a broadband kernel objective."""

    if not _torch_available():
        pytest.skip("torch unavailable")

    params: dict[str, object] = {
        "n_modes": 4,
        "n_bright_ports": 2,
        "n_dark_ports": 2,
        "wl_design_um": 1.25,
        "seed": 2,
        "max_iter": 20,
        "n_restarts": 1,
        "single_lambda_bootstrap": False,
        "kernel_order": 2,
        # Keep this test in the shallower-than-floor regime so _agreement_ok
        # has to exercise the relative tolerance, not the absolute floor.
        "target_null_depth": 1.0e9,
    }
    state = _broadband_kernel_state()
    scipy_loss = _run_backend_with_params(
        "scipy",
        params=params,
        state=state,
        tmp_path=tmp_path,
    )
    torch_loss = _run_backend_with_params(
        "torch",
        params=params,
        state=state,
        tmp_path=tmp_path,
    )

    abs_floor = 1.0e-8
    assert max(abs(scipy_loss), abs(torch_loss)) > abs_floor
    assert _agreement_ok(scipy_loss, torch_loss, rtol=1.0e-2, abs_floor=abs_floor)


def test_torch_batched_clements_matches_single_builder() -> None:
    if not _torch_available():
        pytest.skip("torch unavailable")

    import torch

    from nullsim.stages.chip_optimization.torch import (
        _clements_unitary_ri,
        _clements_unitary_ri_batched,
    )

    n_modes = 5
    n_mzi = n_modes * (n_modes - 1) // 2
    n_params = 2 * n_mzi + n_modes
    rng = np.random.default_rng(11)
    params_np = rng.uniform(0.0, 2.0 * np.pi, size=(4, n_params))
    params_t = torch.from_numpy(params_np.astype(np.float64))

    batch_r, batch_i = _clements_unitary_ri_batched(
        params_t,
        n_modes,
        torch_mod=torch,
    )
    for index in range(params_np.shape[0]):
        single_r, single_i = _clements_unitary_ri(
            params_t[index],
            n_modes,
            torch_mod=torch,
        )
        torch.testing.assert_close(batch_r[index], single_r)
        torch.testing.assert_close(batch_i[index], single_i)


def test_torch_spectral_p_norm_matches_numpy_reducer() -> None:
    if not _torch_available():
        pytest.skip("torch unavailable")

    import torch

    from nullsim.stages.chip_optimization import (
        ChipOptimizationParams,
        _reduce_spectral_leakage_numpy,
    )
    from nullsim.stages.chip_optimization.torch import (
        _reduce_spectral_leakage_torch,
    )

    params = ChipOptimizationParams(
        n_modes=4,
        n_bright_ports=2,
        n_dark_ports=2,
        wl_design_um=1.25,
        spectral_loss_mode="p_norm",
        spectral_loss_p=8.0,
    )
    leakage = np.array([1.0e-6, 1.0e-6, 1.0e-3], dtype=np.float64)
    expected = _reduce_spectral_leakage_numpy(leakage, params)
    actual = _reduce_spectral_leakage_torch(
        torch.from_numpy(leakage),
        params,
        torch_mod=torch,
    )

    assert float(actual) == pytest.approx(expected, rel=1.0e-12)


def test_torch_spectral_edge_weighted_mean_matches_numpy_reducer() -> None:
    if not _torch_available():
        pytest.skip("torch unavailable")

    import torch

    from nullsim.stages.chip_optimization import (
        ChipOptimizationParams,
        _reduce_spectral_leakage_numpy,
    )
    from nullsim.stages.chip_optimization.torch import (
        _reduce_spectral_leakage_torch,
    )

    params = ChipOptimizationParams(
        n_modes=4,
        n_bright_ports=2,
        n_dark_ports=2,
        wl_design_um=1.25,
        spectral_loss_mode="edge_weighted_mean",
        spectral_edge_weight_alpha=4.0,
        spectral_edge_weight_power=2.0,
    )
    wavelength_grid_um = (1.20, 1.25, 1.30)
    leakage = np.array([1.0e-3, 1.0e-6, 1.0e-6], dtype=np.float64)
    expected = _reduce_spectral_leakage_numpy(
        leakage,
        params,
        wavelength_grid_um=wavelength_grid_um,
    )
    actual = _reduce_spectral_leakage_torch(
        torch.from_numpy(leakage),
        params,
        wavelength_grid_um=wavelength_grid_um,
        torch_mod=torch,
    )

    assert float(actual) == pytest.approx(expected, rel=1.0e-12)


def test_torch_tracker_observability_loss_matches_numpy_reference() -> None:
    if not _torch_available():
        pytest.skip("torch unavailable")

    import torch

    from nullsim.physics.fringe_observability import (
        differential_piston_basis,
        intensity_response_matrix,
    )
    from nullsim.stages.chip_optimization.torch import (
        _tracker_observability_crlb_loss_torch,
        _tracker_observability_loss_torch,
    )

    rng = np.random.default_rng(123)
    n_modes = 4
    n_wavelengths = 2
    bright_indices = np.array([0, 1, 2], dtype=np.intp)
    wavelengths_m = np.array([1.20e-6, 1.30e-6], dtype=np.float64)
    raw = rng.normal(size=(n_modes, n_modes)) + 1j * rng.normal(size=(n_modes, n_modes))
    q, _r = np.linalg.qr(raw)
    u_stack = np.stack([q, q.conj()], axis=2).astype(np.complex128)
    fields = (
        rng.normal(size=(n_modes, n_wavelengths, 1))
        + 1j * rng.normal(size=(n_modes, n_wavelengths, 1))
    ).astype(np.complex128)
    piston_basis = differential_piston_basis(n_modes)
    target = 1.0e-3
    eps = 1.0e-12

    response = intensity_response_matrix(
        u_stack=u_stack,
        input_fields=fields,
        wavelengths_m=wavelengths_m,
        port_indices=bright_indices,
        wavelength_indices=np.arange(n_wavelengths, dtype=np.intp),
    )
    projected = response.response_power_per_m @ piston_basis
    fisher_proxy_per_nm = (
        projected / np.sqrt(np.maximum(response.reference_power[:, np.newaxis], eps))
    ) * 1.0e-9
    singular = np.linalg.svd(fisher_proxy_per_nm, compute_uv=False)
    deficits = np.maximum(np.log(target) - np.log(singular + eps), 0.0)
    expected = float(np.sum(deficits**2) / piston_basis.shape[1])
    target_crlb_nm = 100.0
    expected_crlb = float(
        np.mean((1.0 / np.maximum(singular, eps) / target_crlb_nm) ** 2)
    )

    actual = _tracker_observability_loss_torch(
        u_r=torch.from_numpy(np.moveaxis(u_stack.real, 2, 0).copy()),
        u_i=torch.from_numpy(np.moveaxis(u_stack.imag, 2, 0).copy()),
        field_r_by_wl=torch.from_numpy(np.moveaxis(fields.real, 1, 0).copy()),
        field_i_by_wl=torch.from_numpy(np.moveaxis(fields.imag, 1, 0).copy()),
        wavelengths_m=torch.from_numpy(wavelengths_m),
        bright_indices=torch.from_numpy(bright_indices.astype(np.int64)),
        piston_basis=torch.from_numpy(piston_basis),
        min_singular_per_nm=target,
        eps=eps,
        torch_mod=torch,
    )
    actual_crlb = _tracker_observability_crlb_loss_torch(
        u_r=torch.from_numpy(np.moveaxis(u_stack.real, 2, 0).copy()),
        u_i=torch.from_numpy(np.moveaxis(u_stack.imag, 2, 0).copy()),
        field_r_by_wl=torch.from_numpy(np.moveaxis(fields.real, 1, 0).copy()),
        field_i_by_wl=torch.from_numpy(np.moveaxis(fields.imag, 1, 0).copy()),
        wavelengths_m=torch.from_numpy(wavelengths_m),
        bright_indices=torch.from_numpy(bright_indices.astype(np.int64)),
        piston_basis=torch.from_numpy(piston_basis),
        target_rms_nm=target_crlb_nm,
        eps=eps,
        torch_mod=torch,
    )

    assert float(actual) == pytest.approx(expected, rel=1.0e-12, abs=1.0e-12)
    assert float(actual_crlb) == pytest.approx(
        expected_crlb,
        rel=1.0e-12,
        abs=1.0e-12,
    )


def test_torch_amplitude_mismatch_loss_matches_numpy_reference() -> None:
    if not _torch_available():
        pytest.skip("torch unavailable")

    import torch

    from nullsim.physics.fringe_observability import differential_piston_basis
    from nullsim.stages.chip_optimization.torch import (
        _amplitude_mismatch_fractions_torch,
        _amplitude_mismatch_loss_torch,
    )

    rng = np.random.default_rng(321)
    n_modes = 4
    n_wavelengths = 2
    null_indices = np.array([2, 3], dtype=np.intp)
    dump_indices = np.array([1], dtype=np.intp)
    raw = rng.normal(size=(n_modes, n_modes)) + 1j * rng.normal(size=(n_modes, n_modes))
    q, _r = np.linalg.qr(raw)
    u_by_w = np.stack([q, q.conj()], axis=0).astype(np.complex128)
    fields = (
        rng.normal(size=(n_wavelengths, n_modes, 1))
        + 1j * rng.normal(size=(n_wavelengths, n_modes, 1))
    ).astype(np.complex128)
    basis = differential_piston_basis(n_modes)
    eps = 1.0e-12
    min_dump_fraction = 0.2
    probe_sigma_i = 0.02

    basis_4d = basis[np.newaxis, :, np.newaxis, :]
    d_field = 0.5 * fields[:, :, :, np.newaxis] * basis_4d
    null_fractions = []
    for sign in (-1.0, 1.0):
        scale = np.exp(0.5 * sign * probe_sigma_i * basis_4d)
        perturbed = fields[:, :, :, np.newaxis] * scale
        null_out = np.einsum(
            "won,wnpc->wopc",
            u_by_w[:, null_indices, :],
            perturbed,
        )
        dark_power = np.sum(np.abs(null_out) ** 2, axis=(1, 2))
        total_power = np.maximum(np.sum(np.abs(perturbed) ** 2, axis=(1, 2)), eps)
        null_fractions.append(dark_power / total_power)
    dump_out = np.einsum("won,wnpc->wopc", u_by_w[:, dump_indices, :], d_field)
    perturbation_power = np.maximum(np.sum(np.abs(d_field) ** 2, axis=(1, 2)), eps)
    expected_null = 0.5 * (null_fractions[0] + null_fractions[1])
    expected_dump = np.sum(np.abs(dump_out) ** 2, axis=(1, 2)) / perturbation_power
    expected_loss = float(
        np.mean(expected_null) / (probe_sigma_i**2)
        + np.mean(np.maximum(min_dump_fraction - expected_dump, 0.0) ** 2)
    )

    u_r = torch.from_numpy(u_by_w.real.copy())
    u_i = torch.from_numpy(u_by_w.imag.copy())
    field_r = torch.from_numpy(fields.real.copy())
    field_i = torch.from_numpy(fields.imag.copy())
    null_t = torch.from_numpy(null_indices.astype(np.int64))
    dump_t = torch.from_numpy(dump_indices.astype(np.int64))
    basis_t = torch.from_numpy(basis)

    actual_null, actual_dump = _amplitude_mismatch_fractions_torch(
        u_r=u_r,
        u_i=u_i,
        field_r_by_wl=field_r,
        field_i_by_wl=field_i,
        null_indices=null_t,
        dump_indices=dump_t,
        amplitude_basis=basis_t,
        probe_sigma_i=probe_sigma_i,
        eps=eps,
        torch_mod=torch,
    )
    actual_loss = _amplitude_mismatch_loss_torch(
        u_r=u_r,
        u_i=u_i,
        field_r_by_wl=field_r,
        field_i_by_wl=field_i,
        null_indices=null_t,
        dump_indices=dump_t,
        amplitude_basis=basis_t,
        min_dump_fraction=min_dump_fraction,
        probe_sigma_i=probe_sigma_i,
        eps=eps,
        torch_mod=torch,
    )

    np.testing.assert_allclose(actual_null.numpy(), expected_null, rtol=1.0e-12)
    np.testing.assert_allclose(actual_dump.numpy(), expected_dump, rtol=1.0e-12)
    assert float(actual_loss) == pytest.approx(expected_loss, rel=1.0e-12)


def test_torch_amplitude_observability_loss_matches_numpy_reference() -> None:
    if not _torch_available():
        pytest.skip("torch unavailable")

    import torch

    from nullsim.physics.fringe_observability import differential_piston_basis
    from nullsim.stages.chip_optimization.torch import (
        _amplitude_observability_loss_torch,
        _amplitude_observability_singular_values_torch,
    )

    rng = np.random.default_rng(654)
    n_modes = 4
    n_wavelengths = 2
    port_indices = np.array([0, 2, 3], dtype=np.intp)
    raw = rng.normal(size=(n_modes, n_modes)) + 1j * rng.normal(size=(n_modes, n_modes))
    q, _r = np.linalg.qr(raw)
    u_by_w = np.stack([q, q.conj()], axis=0).astype(np.complex128)
    fields = (
        rng.normal(size=(n_wavelengths, n_modes, 1))
        + 1j * rng.normal(size=(n_wavelengths, n_modes, 1))
    ).astype(np.complex128)
    basis = differential_piston_basis(n_modes)
    eps = 1.0e-12
    floor = 0.1

    u_selected = u_by_w[:, port_indices, :]
    output = np.einsum("wom,wmp->wop", u_selected, fields)
    d_field = 0.5 * fields[:, :, :, np.newaxis] * basis[np.newaxis, :, np.newaxis, :]
    d_output = np.einsum("wom,wmpc->wopc", u_selected, d_field)
    response = 2.0 * np.sum(
        np.real(np.conj(output)[..., np.newaxis] * d_output),
        axis=2,
    )
    reference_power = np.sum(np.abs(output) ** 2, axis=2)
    matrix = response / np.sqrt(np.maximum(reference_power[..., np.newaxis], eps))
    expected_singular = np.linalg.svd(
        matrix.reshape(-1, basis.shape[1]),
        compute_uv=False,
    )
    deficits = np.maximum(floor - expected_singular, 0.0) / floor
    if expected_singular.size < basis.shape[1]:
        deficits = np.concatenate(
            (deficits, np.ones(basis.shape[1] - expected_singular.size))
        )
    expected_loss = float(np.mean(deficits**2))

    u_r = torch.from_numpy(u_by_w.real.copy())
    u_i = torch.from_numpy(u_by_w.imag.copy())
    field_r = torch.from_numpy(fields.real.copy())
    field_i = torch.from_numpy(fields.imag.copy())
    ports_t = torch.from_numpy(port_indices.astype(np.int64))
    basis_t = torch.from_numpy(basis)

    actual_singular = _amplitude_observability_singular_values_torch(
        u_r=u_r,
        u_i=u_i,
        field_r_by_wl=field_r,
        field_i_by_wl=field_i,
        port_indices=ports_t,
        amplitude_basis=basis_t,
        eps=eps,
        torch_mod=torch,
    )
    actual_loss = _amplitude_observability_loss_torch(
        u_r=u_r,
        u_i=u_i,
        field_r_by_wl=field_r,
        field_i_by_wl=field_i,
        port_indices=ports_t,
        amplitude_basis=basis_t,
        min_singular_per_fractional_intensity=floor,
        eps=eps,
        torch_mod=torch,
    )

    np.testing.assert_allclose(
        actual_singular.numpy(),
        expected_singular,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert float(actual_loss) == pytest.approx(expected_loss, rel=1.0e-12)


def test_torch_joint_phase_amplitude_observability_loss_matches_numpy_reference() -> None:
    if not _torch_available():
        pytest.skip("torch unavailable")

    import torch

    from nullsim.physics.fringe_observability import differential_piston_basis
    from nullsim.stages.chip_optimization.torch import (
        _joint_phase_amplitude_observability_crlb_loss_torch,
        _joint_phase_amplitude_observability_loss_torch,
        _joint_phase_amplitude_observability_singular_values_torch,
    )

    rng = np.random.default_rng(987)
    n_modes = 4
    n_wavelengths = 2
    port_indices = np.array([0, 1, 3], dtype=np.intp)
    wavelengths_m = np.array([1.20e-6, 1.35e-6], dtype=np.float64)
    raw = rng.normal(size=(n_modes, n_modes)) + 1j * rng.normal(size=(n_modes, n_modes))
    q, _r = np.linalg.qr(raw)
    u_by_w = np.stack([q, q.conj()], axis=0).astype(np.complex128)
    fields = (
        rng.normal(size=(n_wavelengths, n_modes, 1))
        + 1j * rng.normal(size=(n_wavelengths, n_modes, 1))
    ).astype(np.complex128)
    basis = differential_piston_basis(n_modes)
    eps = 1.0e-12
    floor = 0.02
    amplitude_scale = 0.01
    crlb_target = 100.0

    u_selected = u_by_w[:, port_indices, :]
    output = np.einsum("wom,wmp->wop", u_selected, fields)
    reference_power = np.sum(np.abs(output) ** 2, axis=2)

    phase_per_m = (2.0 * np.pi / wavelengths_m)[:, np.newaxis, np.newaxis, np.newaxis]
    d_phase_field = 1j * fields[:, :, :, np.newaxis] * phase_per_m
    d_phase_field = d_phase_field * basis[np.newaxis, :, np.newaxis, :]
    d_phase_output = np.einsum("wom,wmpc->wopc", u_selected, d_phase_field)
    phase_response = 2.0 * np.sum(
        np.real(np.conj(output)[..., np.newaxis] * d_phase_output),
        axis=2,
    )
    phase_matrix = phase_response / np.sqrt(
        np.maximum(reference_power[..., np.newaxis], eps)
    )
    phase_matrix = phase_matrix.reshape(-1, basis.shape[1]) * 1.0e-9

    d_amp_field = 0.5 * fields[:, :, :, np.newaxis] * basis[np.newaxis, :, np.newaxis, :]
    d_amp_output = np.einsum("wom,wmpc->wopc", u_selected, d_amp_field)
    amplitude_response = 2.0 * np.sum(
        np.real(np.conj(output)[..., np.newaxis] * d_amp_output),
        axis=2,
    )
    amplitude_matrix = amplitude_response / np.sqrt(
        np.maximum(reference_power[..., np.newaxis], eps)
    )
    amplitude_matrix = amplitude_matrix.reshape(-1, basis.shape[1])
    matrix = np.concatenate(
        (phase_matrix, amplitude_matrix * amplitude_scale),
        axis=1,
    )
    expected_singular = np.linalg.svd(matrix, compute_uv=False)
    deficits = np.maximum(floor - expected_singular, 0.0) / floor
    if expected_singular.size < 2 * basis.shape[1]:
        deficits = np.concatenate(
            (deficits, np.ones(2 * basis.shape[1] - expected_singular.size))
        )
    expected_loss = float(np.mean(deficits**2))
    expected_crlb_loss = float(np.mean((1.0 / expected_singular / crlb_target) ** 2))

    u_r = torch.from_numpy(u_by_w.real.copy())
    u_i = torch.from_numpy(u_by_w.imag.copy())
    field_r = torch.from_numpy(fields.real.copy())
    field_i = torch.from_numpy(fields.imag.copy())
    wavelengths_t = torch.from_numpy(wavelengths_m.copy())
    ports_t = torch.from_numpy(port_indices.astype(np.int64))
    basis_t = torch.from_numpy(basis)

    actual_singular = _joint_phase_amplitude_observability_singular_values_torch(
        u_r=u_r,
        u_i=u_i,
        field_r_by_wl=field_r,
        field_i_by_wl=field_i,
        wavelengths_m=wavelengths_t,
        port_indices=ports_t,
        piston_basis=basis_t,
        amplitude_basis=basis_t,
        amplitude_scale=amplitude_scale,
        eps=eps,
        torch_mod=torch,
    )
    actual_loss = _joint_phase_amplitude_observability_loss_torch(
        u_r=u_r,
        u_i=u_i,
        field_r_by_wl=field_r,
        field_i_by_wl=field_i,
        wavelengths_m=wavelengths_t,
        port_indices=ports_t,
        piston_basis=basis_t,
        amplitude_basis=basis_t,
        amplitude_scale=amplitude_scale,
        min_singular_per_scaled_mode=floor,
        eps=eps,
        torch_mod=torch,
    )
    actual_crlb_loss = _joint_phase_amplitude_observability_crlb_loss_torch(
        u_r=u_r,
        u_i=u_i,
        field_r_by_wl=field_r,
        field_i_by_wl=field_i,
        wavelengths_m=wavelengths_t,
        port_indices=ports_t,
        piston_basis=basis_t,
        amplitude_basis=basis_t,
        amplitude_scale=amplitude_scale,
        target_per_scaled_mode=crlb_target,
        eps=eps,
        torch_mod=torch,
    )

    np.testing.assert_allclose(
        actual_singular.numpy(),
        expected_singular,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert float(actual_loss) == pytest.approx(expected_loss, rel=1.0e-12)
    # The CRLB loss squares reciprocals of the joint singular spectrum, so
    # platform-level SVD roundoff is amplified slightly more than in the
    # singular-value and hinge-loss checks above.
    assert float(actual_crlb_loss) == pytest.approx(
        expected_crlb_loss,
        rel=1.0e-11,
    )


def test_torch_tracker_observability_wavelength_indices_use_channel_reps() -> None:
    from nullsim.stages.chip_optimization.torch import (
        _tracker_observability_wavelength_indices,
    )

    np.testing.assert_array_equal(
        _tracker_observability_wavelength_indices(5, 0),
        np.array([0, 1, 2, 3, 4]),
    )
    np.testing.assert_array_equal(
        _tracker_observability_wavelength_indices(5, 1),
        np.array([2]),
    )
    np.testing.assert_array_equal(
        _tracker_observability_wavelength_indices(5, 2),
        np.array([1, 3]),
    )


def test_scipy_rejects_tracker_observability_regularizer(tmp_path: Path) -> None:
    from nullsim.stages.chip_optimization.scipy import ChipOptimizationScipy

    stage = ChipOptimizationScipy(
        {
            "n_modes": 4,
            "n_bright_ports": 2,
            "n_dark_ports": 2,
            "wl_design_um": 1.65,
            "tracker_observability_weight": 1.0e-3,
        },
        _context(tmp_path, instance_id="chip_opt_scipy_observability"),
    )

    with pytest.raises(
        ValueError,
        match="only implemented for chip_optimization.torch",
    ):
        stage.apply(_uniform_state(n_modes=4))


def test_torch_stage_applies_tracker_observability_regularizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not _torch_available():
        pytest.skip("torch unavailable")

    from nullsim.stages.chip_optimization.torch import ChipOptimizationTorch
    import nullsim.stages.chip_optimization.torch as torch_backend

    def fake_minimize(fun, x0, **_kwargs):
        value = fun(x0)
        if isinstance(value, tuple):  # fused (loss, grad) objectives
            value = value[0]
        return SimpleNamespace(fun=float(value), x=np.asarray(x0), nit=0)

    monkeypatch.setattr(torch_backend, "minimize", fake_minimize)

    stage = ChipOptimizationTorch(
        {
            "n_modes": 4,
            "n_bright_ports": 3,
            "n_dark_ports": 1,
            "wl_design_um": 1.25,
            "max_iter": 1,
            "n_restarts": 1,
            "single_lambda_bootstrap": False,
            "tracker_observability_weight": 1.0e-3,
            "tracker_observability_min_singular_per_nm": 1.0e-3,
            "tracker_observability_crlb_weight": 2.0e-3,
            "tracker_observability_crlb_target_nm": 100.0,
            "tracker_observability_rank_priority": True,
            "tracker_observability_n_spectral_channels": 1,
        },
        _context(tmp_path, instance_id="chip_opt_torch_observability"),
    )

    out = stage.apply(_uniform_state(n_modes=4, wl_um=1.25))
    payload = next(
        payload
        for name, payload in out.results.components.items()
        if name.startswith("chip_optimization")
    )

    assert payload["tracker_observability_weight"] == 1.0e-3
    assert payload["tracker_observability_min_singular_per_nm"] == 1.0e-3
    assert payload["tracker_observability_crlb_weight"] == 2.0e-3
    assert payload["tracker_observability_crlb_target_nm"] == 100.0
    assert payload["tracker_observability_rank_priority"] is True
    assert payload["tracker_observability_wavelength_indices"] == (0,)
    assert payload["tracker_observability_basis_source"] == "chip_input_modes"
    assert payload["tracker_observability_n_physical_modes"] == 4
    assert payload["tracker_observability_field_source"] == "nominal_on_axis_field"
    assert payload["tracker_observability_n_field_states"] == 1
    assert np.isfinite(payload["optimization_leakage_loss_final"])
    assert np.isfinite(payload["optimization_tracker_loss_final"])
    assert np.isfinite(payload["optimization_tracker_crlb_loss_final"])
    assert isinstance(
        payload["tracker_observability_singular_values_per_nm_final"],
        tuple,
    )
    assert np.isfinite(payload["tracker_observability_min_singular_per_nm_final"])
    assert np.isfinite(payload["tracker_observability_crlb_rms_nm_final"])
    assert np.isfinite(payload["tracker_observability_crlb_loss_final"])
    assert payload["optimization_target_null_depth"] == pytest.approx(1.0e-6)
    assert isinstance(payload["optimization_null_target_met_final"], bool)
    assert isinstance(payload["optimization_selection_key_final"], tuple)
    assert payload["n_restarts_completed"] == 1
    assert isinstance(payload["optimization_best_result_success"], bool)


def test_tracker_rank_priority_selection_uses_science_objective_after_crlb_target() -> None:
    from nullsim.stages.chip_optimization.torch import (
        _tracker_rank_priority_selection_key,
    )

    crlb_target_nm = 100.0
    # Candidate A has a better CRLB but a worse science objective. Candidate B
    # is still full-rank and under the CRLB target, so the restart chooser
    # should keep the throughput-preserving science basin.
    overoptimized_tracker = _tracker_rank_priority_selection_key(
        singular_values_per_nm=np.array([0.20, 0.20, 0.20]),
        n_dof=3,
        min_singular_per_nm=1.0e-3,
        crlb_target_nm=crlb_target_nm,
        objective_loss=1.0,
        normalized_null=8.0e-7,
        target_null_depth=1.0e-6,
        eps=1.0e-12,
    )
    science_basin = _tracker_rank_priority_selection_key(
        singular_values_per_nm=np.array([0.02, 0.02, 0.02]),
        n_dof=3,
        min_singular_per_nm=1.0e-3,
        crlb_target_nm=crlb_target_nm,
        objective_loss=-1.0,
        normalized_null=8.0e-7,
        target_null_depth=1.0e-6,
        eps=1.0e-12,
    )

    assert science_basin < overoptimized_tracker


def test_tracker_rank_priority_selection_penalizes_crlb_target_miss() -> None:
    from nullsim.stages.chip_optimization.torch import (
        _tracker_rank_priority_selection_key,
    )

    crlb_target_nm = 100.0
    weak_tracker = _tracker_rank_priority_selection_key(
        singular_values_per_nm=np.array([0.005, 0.005, 0.005]),
        n_dof=3,
        min_singular_per_nm=1.0e-3,
        crlb_target_nm=crlb_target_nm,
        objective_loss=-1.0,
        normalized_null=8.0e-7,
        target_null_depth=1.0e-6,
        eps=1.0e-12,
    )
    sufficient_tracker = _tracker_rank_priority_selection_key(
        singular_values_per_nm=np.array([0.02, 0.02, 0.02]),
        n_dof=3,
        min_singular_per_nm=1.0e-3,
        crlb_target_nm=crlb_target_nm,
        objective_loss=1.0,
        normalized_null=8.0e-7,
        target_null_depth=1.0e-6,
        eps=1.0e-12,
    )

    assert sufficient_tracker < weak_tracker


def test_torch_stage_applies_amplitude_mismatch_regularizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not _torch_available():
        pytest.skip("torch unavailable")

    from nullsim.stages.chip_optimization.torch import ChipOptimizationTorch
    import nullsim.stages.chip_optimization.torch as torch_backend

    def fake_minimize(fun, x0, **_kwargs):
        value = fun(x0)
        if isinstance(value, tuple):  # fused (loss, grad) objectives
            value = value[0]
        return SimpleNamespace(fun=float(value), x=np.asarray(x0), nit=0)

    monkeypatch.setattr(torch_backend, "minimize", fake_minimize)

    stage = ChipOptimizationTorch(
        {
            "n_modes": 4,
            "n_bright_ports": 3,
            "n_dark_ports": 1,
            "wl_design_um": 1.25,
            "max_iter": 1,
            "n_restarts": 1,
            "single_lambda_bootstrap": False,
            "amplitude_mismatch_weight": 0.5,
            "amplitude_mismatch_n_spectral_channels": 1,
            "amplitude_dump_port_index": 2,
            "amplitude_dump_min_fraction": 0.1,
        },
        _context(tmp_path, instance_id="chip_opt_torch_amplitude"),
    )

    out = stage.apply(_uniform_state(n_modes=4, wl_um=1.25))
    payload = next(
        payload
        for name, payload in out.results.components.items()
        if name.startswith("chip_optimization")
    )

    assert payload["amplitude_mismatch_weight"] == 0.5
    assert payload["amplitude_mismatch_wavelength_indices"] == (0,)
    assert payload["amplitude_mismatch_basis_source"] == "chip_input_modes"
    assert payload["amplitude_mismatch_n_physical_modes"] == 4
    assert payload["amplitude_dump_port_index"] == 2
    assert payload["amplitude_dump_min_fraction"] == 0.1
    assert np.isfinite(payload["optimization_amplitude_mismatch_loss_final"])
    assert np.isfinite(payload["amplitude_mismatch_null_fraction_mean_final"])
    assert np.isfinite(payload["amplitude_dump_fraction_mean_final"])


def test_torch_stage_applies_amplitude_observability_regularizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not _torch_available():
        pytest.skip("torch unavailable")

    from nullsim.stages.chip_optimization.torch import ChipOptimizationTorch
    import nullsim.stages.chip_optimization.torch as torch_backend

    def fake_minimize(fun, x0, **_kwargs):
        value = fun(x0)
        if isinstance(value, tuple):  # fused (loss, grad) objectives
            value = value[0]
        return SimpleNamespace(fun=float(value), x=np.asarray(x0), nit=0)

    monkeypatch.setattr(torch_backend, "minimize", fake_minimize)

    stage = ChipOptimizationTorch(
        {
            "n_modes": 4,
            "n_bright_ports": 2,
            "n_dark_ports": 2,
            "wl_design_um": 1.25,
            "max_iter": 1,
            "n_restarts": 1,
            "single_lambda_bootstrap": False,
            "amplitude_observability_weight": 0.25,
            "amplitude_observability_min_singular_per_fractional_intensity": 0.05,
            "amplitude_observability_port_set": "bright_dark",
            "amplitude_observability_n_spectral_channels": 1,
        },
        _context(tmp_path, instance_id="chip_opt_torch_amplitude_observability"),
    )

    out = stage.apply(_uniform_state(n_modes=4, wl_um=1.25))
    payload = next(
        payload
        for name, payload in out.results.components.items()
        if name.startswith("chip_optimization")
    )

    assert payload["amplitude_observability_weight"] == 0.25
    assert (
        payload["amplitude_observability_min_singular_per_fractional_intensity"]
        == 0.05
    )
    assert payload["amplitude_observability_port_set"] == "bright_dark"
    assert payload["amplitude_observability_port_indices"] == (0, 1, 2, 3)
    assert payload["amplitude_observability_wavelength_indices"] == (0,)
    assert payload["amplitude_observability_basis_source"] == "chip_input_modes"
    assert np.isfinite(payload["optimization_amplitude_observability_loss_final"])
    assert isinstance(payload["amplitude_observability_singular_values_final"], tuple)
    assert np.isfinite(payload["amplitude_observability_min_singular_final"])
    assert np.isfinite(payload["amplitude_observability_condition_final"])


def test_torch_stage_applies_joint_observability_regularizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not _torch_available():
        pytest.skip("torch unavailable")

    from nullsim.stages.chip_optimization.torch import ChipOptimizationTorch
    import nullsim.stages.chip_optimization.torch as torch_backend

    def fake_minimize(fun, x0, **_kwargs):
        value = fun(x0)
        if isinstance(value, tuple):  # fused (loss, grad) objectives
            value = value[0]
        return SimpleNamespace(fun=float(value), x=np.asarray(x0), nit=0)

    monkeypatch.setattr(torch_backend, "minimize", fake_minimize)

    stage = ChipOptimizationTorch(
        {
            "n_modes": 4,
            "n_bright_ports": 2,
            "n_dark_ports": 2,
            "wl_design_um": 1.25,
            "max_iter": 1,
            "n_restarts": 1,
            "single_lambda_bootstrap": False,
            "joint_observability_weight": 0.125,
            "joint_observability_min_singular_per_scaled_mode": 0.02,
            "joint_observability_crlb_weight": 0.05,
            "joint_observability_crlb_target_per_scaled_mode": 75.0,
            "joint_observability_science_null_guard_weight": 0.25,
            "joint_observability_science_null_guard_factor": 1.5,
            "joint_observability_port_set": "bright_dark",
            "joint_observability_n_spectral_channels": 2,
            "joint_observability_amplitude_scale": 0.015,
        },
        _context(tmp_path, instance_id="chip_opt_torch_joint_observability"),
    )

    out = stage.apply(_broadband_kernel_state())
    payload = next(
        payload
        for name, payload in out.results.components.items()
        if name.startswith("chip_optimization")
    )

    assert payload["joint_observability_weight"] == 0.125
    assert payload["joint_observability_min_singular_per_scaled_mode"] == 0.02
    assert payload["joint_observability_crlb_weight"] == 0.05
    assert payload["joint_observability_crlb_target_per_scaled_mode"] == 75.0
    assert payload["joint_observability_science_null_guard_weight"] == 0.25
    assert payload["joint_observability_science_null_guard_factor"] == 1.5
    assert payload["joint_observability_port_set"] == "bright_dark"
    assert payload["joint_observability_port_indices"] == (0, 1, 2, 3)
    assert payload["joint_observability_wavelength_indices"] == (0, 2)
    assert payload["joint_observability_amplitude_scale"] == 0.015
    assert payload["joint_observability_basis_source"] == "chip_input_modes"
    assert payload["joint_observability_n_rows"] == 8
    assert payload["joint_observability_n_columns"] == 6
    assert np.isfinite(payload["optimization_joint_observability_loss_final"])
    assert np.isfinite(payload["optimization_joint_observability_crlb_loss_final"])
    assert np.isfinite(payload["optimization_joint_science_guard_loss_final"])
    assert isinstance(payload["joint_observability_singular_values_final"], tuple)
    assert np.isfinite(payload["joint_observability_min_singular_final"])
    assert np.isfinite(payload["joint_observability_condition_final"])
    assert payload["joint_observability_n_modes_final"] == 6


def test_torch_joint_observability_rejects_underdetermined_setup(
    tmp_path: Path,
) -> None:
    if not _torch_available():
        pytest.skip("torch unavailable")

    from nullsim.stages.chip_optimization.torch import ChipOptimizationTorch

    stage = ChipOptimizationTorch(
        {
            "n_modes": 4,
            "n_bright_ports": 2,
            "n_dark_ports": 2,
            "wl_design_um": 1.25,
            "max_iter": 1,
            "n_restarts": 1,
            "single_lambda_bootstrap": False,
            "joint_observability_weight": 0.125,
            "joint_observability_port_set": "bright_dark",
            "joint_observability_n_spectral_channels": 1,
        },
        _context(tmp_path, instance_id="chip_opt_torch_joint_rank_guard"),
    )

    with pytest.raises(ValueError, match="rank-deficient by construction"):
        stage.apply(_uniform_state(n_modes=4, wl_um=1.25))


def test_torch_rejects_multichannel_tracker_when_skipping_broadband_refine(
    tmp_path: Path,
) -> None:
    if not _torch_available():
        pytest.skip("torch unavailable")

    from nullsim.stages.chip_optimization.torch import ChipOptimizationTorch

    n_modes = 4
    wavelength_grid_um = (1.20, 1.25, 1.30)
    base_state = _uniform_state(n_modes=n_modes, wl_um=1.25)
    state = replace(
        base_state,
        field=Field(
            amplitudes=np.ones(
                (n_modes, len(wavelength_grid_um), 1),
                dtype=np.complex128,
            )
            / np.sqrt(n_modes),
            wavelength_grid_um=wavelength_grid_um,
        ),
        throughput=Throughput(
            wavelength_grid_um=wavelength_grid_um,
            transmission_fraction=(1.0, 1.0, 1.0),
        ),
    )
    stage = ChipOptimizationTorch(
        {
            "n_modes": n_modes,
            "n_bright_ports": 3,
            "n_dark_ports": 1,
            "wl_design_um": 1.25,
            "broadband_refine": False,
            "tracker_observability_weight": 1.0e-3,
            "tracker_observability_n_spectral_channels": 2,
        },
        _context(tmp_path, instance_id="chip_opt_torch_tracker_guard"),
    )

    with pytest.raises(ValueError, match="broadband_refine=False"):
        stage.apply(state)


@pytest.mark.parametrize("kernel_order", [0, 2])
def test_torch_throughput_penalty_sums_dark_ports_like_kilo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kernel_order: int
) -> None:
    """KILO's throughput penalty averages over PA samples, not dark ports."""

    if not _torch_available():
        pytest.skip("torch unavailable")

    import torch

    from nullsim.physics.planet_throughput import planet_input_field
    from nullsim.stages.chip_optimization import (
        _kernel_null_field_stack,
        _n_params,
        _normalized_loss_for_target,
        _reported_null_depth_for_target,
        _require_field,
    )
    from nullsim.stages.chip_optimization.torch import (
        ChipOptimizationTorch,
        _clements_unitary_ri,
    )
    import nullsim.stages.chip_optimization.torch as torch_backend

    def fake_minimize(fun, x0, **_kwargs):
        value = fun(x0)
        if isinstance(value, tuple):  # fused (loss, grad) objectives
            value = value[0]
        return SimpleNamespace(fun=float(value), x=np.asarray(x0), nit=0)

    monkeypatch.setattr(torch_backend, "minimize", fake_minimize)

    n_modes = 4
    n_dark_ports = 2
    n_bright_ports = n_modes - n_dark_ports
    wavelength_grid_um = (1.25,)
    wl_design_um = 1.25
    seed = 7
    alpha = 0.25
    n_pa = 4
    separation_mas = 120.0
    positions = np.array(
        [[-1.0, -0.5], [0.5, -0.25], [1.25, 0.75], [2.0, 1.0]],
        dtype=np.float64,
    )
    amplitudes = np.ones((n_modes, 1, 1), dtype=np.complex128) / np.sqrt(n_modes)
    state = SimulationState.empty({"wavelength_grid_um": wavelength_grid_um})
    state = replace(
        state,
        field=Field(amplitudes=amplitudes, wavelength_grid_um=wavelength_grid_um),
        throughput=Throughput(
            wavelength_grid_um=wavelength_grid_um,
            transmission_fraction=(1.0,),
        ),
        geometry=Geometry(
            aperture_positions_pupil_m=positions,
            effective_collecting_area_m2=8.0,
        ),
    )

    stage = ChipOptimizationTorch(
        {
            "n_modes": n_modes,
            "n_bright_ports": n_bright_ports,
            "n_dark_ports": n_dark_ports,
            "wl_design_um": wl_design_um,
            "seed": seed,
            "max_iter": 1,
            "n_restarts": 1,
            "single_lambda_bootstrap": False,
            "kernel_order": kernel_order,
            "throughput_penalty_alpha": alpha,
            "throughput_penalty_seps_mas": [separation_mas],
            "throughput_penalty_n_pa": n_pa,
        },
        _context(tmp_path, instance_id="chip_opt_torch_penalty"),
    )

    out = stage.apply(state)
    payload = next(
        payload
        for name, payload in out.results.components.items()
        if name.startswith("chip_optimization")
    )

    master_rng = np.random.default_rng(seed)
    restart_seed = int(master_rng.integers(0, 2**31 - 1))
    x0 = np.random.default_rng(restart_seed).uniform(
        0.0, 2.0 * np.pi, _n_params(n_modes)
    )
    params_t = torch.from_numpy(x0.astype(np.float64))
    u_r_t, u_i_t = _clements_unitary_ri(params_t, n_modes, torch_mod=torch)
    u = u_r_t.numpy() + 1j * u_i_t.numpy()

    dark_idx = np.arange(n_bright_ports, n_modes)
    optimizer_field = _require_field(state, n_modes, wl_design_um=wl_design_um)
    field_stack = (
        _kernel_null_field_stack(optimizer_field, positions, order=kernel_order)
        if kernel_order > 0
        else optimizer_field[np.newaxis, ...]
    )
    leak = 0.0
    for row in range(field_stack.shape[0]):
        out_field = u @ field_stack[row, :, 0, 0]
        leak += float(np.sum(np.abs(out_field[dark_idx]) ** 2))

    rad_per_mas = np.pi / (180.0 * 3600.0 * 1000.0)
    sep_rad = separation_mas * rad_per_mas
    planet_dark_sum = 0.0
    mode_areas_m2 = np.full(n_modes, 2.0, dtype=np.float64)
    mode_diameters_m = np.sqrt(4.0 * mode_areas_m2 / np.pi)
    coupling_per_mode = np.abs(optimizer_field[:, 0, 0]) ** 2 / mode_areas_m2
    for pa in np.linspace(0.0, 2.0 * np.pi, n_pa, endpoint=False):
        planet_field = planet_input_field(
            positions,
            coupling_per_mode,
            mode_areas_m2,
            mode_diameters_m,
            wl_design_um * 1.0e-6,
            sep_rad,
            float(pa),
        )
        out_planet = u @ planet_field
        planet_dark_sum += float(np.sum(np.abs(out_planet[dark_idx]) ** 2))
    expected = leak - alpha * planet_dark_sum / float(n_pa)

    assert float(payload["optimization_loss_final"]) == pytest.approx(
        expected, rel=1.0e-12, abs=1.0e-12
    )
    assert float(payload["optimization_objective_final"]) == pytest.approx(
        expected, rel=1.0e-12, abs=1.0e-12
    )
    assert float(payload["optimization_leakage_loss_final"]) == pytest.approx(
        leak, rel=1.0e-12, abs=1.0e-12
    )
    assert float(payload["optimization_throughput_reward_final"]) == pytest.approx(
        planet_dark_sum / float(n_pa), rel=1.0e-12, abs=1.0e-12
    )
    assert float(payload["optimization_normalized_null_final"]) == pytest.approx(
        _reported_null_depth_for_target(
            leak,
            optimizer_field,
            n_objective_rows=field_stack.shape[0],
        ),
        rel=1.0e-12,
        abs=1.0e-12,
    )
    assert float(payload["optimization_normalized_null_legacy_final"]) == pytest.approx(
        _normalized_loss_for_target(leak, optimizer_field),
        rel=1.0e-12,
        abs=1.0e-12,
    )


def test_scipy_throughput_penalty_sums_dark_ports_like_kilo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scipy backend must honor the shared throughput-penalty params."""

    from nullsim.physics.planet_throughput import planet_input_field
    from nullsim.stages.chip_optimization import (
        _build_unitary_stack_numpy,
        _n_params,
        _require_field,
    )
    from nullsim.stages.chip_optimization.scipy import ChipOptimizationScipy
    import nullsim.stages.chip_optimization.scipy as scipy_backend

    def fake_minimize(fun, x0, **_kwargs):
        value = fun(x0)
        if isinstance(value, tuple):  # fused (loss, grad) objectives
            value = value[0]
        return SimpleNamespace(fun=float(value), x=np.asarray(x0), nit=0)

    monkeypatch.setattr(scipy_backend, "minimize", fake_minimize)

    n_modes = 4
    n_dark_ports = 2
    n_bright_ports = n_modes - n_dark_ports
    wavelength_grid_um = (1.25,)
    wl_design_um = 1.25
    seed = 7
    alpha = 0.25
    n_pa = 4
    separation_mas = 120.0
    positions = np.array(
        [[-1.0, -0.5], [0.5, -0.25], [1.25, 0.75], [2.0, 1.0]],
        dtype=np.float64,
    )
    amplitudes = np.ones((n_modes, 1, 1), dtype=np.complex128) / np.sqrt(n_modes)
    state = SimulationState.empty({"wavelength_grid_um": wavelength_grid_um})
    state = replace(
        state,
        field=Field(amplitudes=amplitudes, wavelength_grid_um=wavelength_grid_um),
        throughput=Throughput(
            wavelength_grid_um=wavelength_grid_um,
            transmission_fraction=(1.0,),
        ),
        geometry=Geometry(
            aperture_positions_pupil_m=positions,
            effective_collecting_area_m2=8.0,
        ),
    )
    params = {
        "n_modes": n_modes,
        "n_bright_ports": n_bright_ports,
        "n_dark_ports": n_dark_ports,
        "wl_design_um": wl_design_um,
        "seed": seed,
        "max_iter": 1,
        "n_restarts": 1,
        "target_null_depth": 1.0e9,
        "throughput_penalty_alpha": alpha,
        "throughput_penalty_seps_mas": [separation_mas],
        "throughput_penalty_n_pa": n_pa,
    }
    stage = ChipOptimizationScipy(
        params,
        _context(tmp_path, instance_id="chip_opt_scipy_penalty"),
    )

    out = stage.apply(state)
    payload = next(
        payload
        for name, payload in out.results.components.items()
        if name.startswith("chip_optimization")
    )

    master_rng = np.random.default_rng(seed)
    restart_seed = int(master_rng.integers(0, 2**31 - 1))
    x0 = np.random.default_rng(restart_seed).uniform(
        0.0, 2.0 * np.pi, _n_params(n_modes)
    )
    params_model = stage.params_model
    u = _build_unitary_stack_numpy(
        params_vec=x0,
        n_modes=n_modes,
        wavelength_grid_um=wavelength_grid_um,
        wl_design_um=wl_design_um,
        dac_bits=params_model.dac_bits,
        chromaticity_params=params_model,
    )[:, :, 0]
    dark_idx = np.arange(n_bright_ports, n_modes)
    optimizer_field = _require_field(state, n_modes, wl_design_um=wl_design_um)
    leak = float(np.sum(np.abs((u @ optimizer_field[:, 0, 0])[dark_idx]) ** 2))

    rad_per_mas = np.pi / (180.0 * 3600.0 * 1000.0)
    sep_rad = separation_mas * rad_per_mas
    mode_areas_m2 = np.full(n_modes, 2.0, dtype=np.float64)
    mode_diameters_m = np.sqrt(4.0 * mode_areas_m2 / np.pi)
    coupling_per_mode = np.abs(optimizer_field[:, 0, 0]) ** 2 / mode_areas_m2
    planet_dark_sum = 0.0
    for pa in np.linspace(0.0, 2.0 * np.pi, n_pa, endpoint=False):
        planet_field = planet_input_field(
            positions,
            coupling_per_mode,
            mode_areas_m2,
            mode_diameters_m,
            wl_design_um * 1.0e-6,
            sep_rad,
            float(pa),
        )
        planet_dark_sum += float(np.sum(np.abs((u @ planet_field)[dark_idx]) ** 2))
    expected = leak - alpha * planet_dark_sum / float(n_pa)

    assert float(payload["optimization_loss_final"]) == pytest.approx(
        expected, rel=1.0e-12, abs=1.0e-12
    )


def test_torch_stage_with_disk_samples_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not _torch_available():
        pytest.skip("torch unavailable")

    import nullsim.stages.chip_optimization.torch as torch_backend
    from nullsim.stages.chip_optimization.torch import ChipOptimizationTorch

    def fake_minimize(fun, x0, **_kwargs):
        value = fun(x0)
        if isinstance(value, tuple):  # fused (loss, grad) objectives
            value = value[0]
        return SimpleNamespace(fun=float(value), x=np.asarray(x0), nit=0)

    monkeypatch.setattr(torch_backend, "minimize", fake_minimize)

    n_modes = 4
    wavelength_grid_um = (1.20, 1.30)
    state = SimulationState.empty({"wavelength_grid_um": wavelength_grid_um})
    state = replace(
        state,
        scene=Scene(star_angular_diameter_mas=1.0),
        field=Field(
            amplitudes=np.ones((n_modes, 2, 1), dtype=np.complex128) / 2,
            wavelength_grid_um=wavelength_grid_um,
        ),
        throughput=Throughput(
            wavelength_grid_um=wavelength_grid_um,
            transmission_fraction=(1.0, 1.0),
        ),
        geometry=Geometry(
            aperture_positions_pupil_m=np.array(
                [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
                dtype=np.float64,
            ),
            effective_collecting_area_m2=4.0,
        ),
    )
    stage = ChipOptimizationTorch(
        {
            "n_modes": n_modes,
            "n_bright_ports": 2,
            "n_dark_ports": 2,
            "wl_design_um": 1.25,
            "seed": 3,
            "max_iter": 1,
            "n_restarts": 1,
            "single_lambda_bootstrap": False,
            "disk_n_rings": 1,
            "disk_n_azimuths": 2,
        },
        _context(tmp_path, instance_id="chip_opt_torch_disk"),
    )

    out = stage.apply(state)
    payload = next(
        payload
        for name, payload in out.results.components.items()
        if name.startswith("chip_optimization")
    )
    assert np.isfinite(payload["optimization_loss_final"])
    assert payload["torch_unitary_builder"] == "batched_clements_ri"


def test_torch_stage_with_delay_states_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not _torch_available():
        pytest.skip("torch unavailable")

    import nullsim.stages.chip_optimization.torch as torch_backend
    from nullsim.stages.chip_optimization.torch import ChipOptimizationTorch

    def fake_minimize(fun, x0, **_kwargs):
        value = fun(x0)
        if isinstance(value, tuple):  # fused (loss, grad) objectives
            value = value[0]
        return SimpleNamespace(fun=float(value), x=np.asarray(x0), nit=0)

    monkeypatch.setattr(torch_backend, "minimize", fake_minimize)

    wavelength_grid_um = (1.50, 1.55, 1.60)
    delay_diag = {
        "fiber_lengths_m": (1.0, 1.0),
        "n_eff": 1.0003,
        "phase_index_ref": 1.0003,
        "group_index_ref": 1.0004,
        "dispersion_ps_nm_km": 1.0,
        "attenuation_db_per_m": 0.0,
        "altitude_deg": 0.0,
        "azimuth_deg": 0.0,
        "fixed_delay_line_loss_db_per_telescope": None,
        "chromatic_control": "multi_wavelength_group_delay",
        "reference_wavelength_um": 1.55,
        "per_mode_field_amplitude_applied": True,
    }
    state = SimulationState.empty({"wavelength_grid_um": wavelength_grid_um})
    state = replace(
        state,
        scene=Scene(star_jmag=4.0, altitude_deg=0.0, azimuth_deg=0.0),
        field=Field(
            amplitudes=np.ones((2, 3, 1), dtype=np.complex128) / np.sqrt(2.0),
            wavelength_grid_um=wavelength_grid_um,
        ),
        throughput=Throughput(
            wavelength_grid_um=wavelength_grid_um,
            transmission_fraction=(1.0, 1.0, 1.0),
            components={
                "delay_line_loss": {
                    "wavelength_grid_um": wavelength_grid_um,
                    "transmission_fraction": (1.0, 1.0, 1.0),
                    "per_aperture": delay_diag,
                },
            },
        ),
        geometry=Geometry(
            n_apertures=2,
            aperture_positions_enu_m=np.array(
                [[0.0, 0.0, 0.0], [0.0, 10.0, 0.0]],
                dtype=np.float64,
            ),
        ),
    )
    stage = ChipOptimizationTorch(
        {
            "n_modes": 2,
            "n_bright_ports": 1,
            "n_dark_ports": 1,
            "wl_design_um": 1.55,
            "seed": 5,
            "max_iter": 1,
            "n_restarts": 1,
            "single_lambda_bootstrap": False,
            "delay_state_n_azimuths": 2,
            "tracker_observability_weight": 1.0e-3,
            "tracker_observability_n_spectral_channels": 2,
            "amplitude_observability_weight": 1.0e-3,
            "amplitude_observability_n_spectral_channels": 2,
            "joint_observability_crlb_weight": 1.0e-3,
            "joint_observability_n_spectral_channels": 2,
        },
        _context(tmp_path, instance_id="chip_opt_torch_delay_states"),
    )

    out = stage.apply(state)
    payload = next(
        payload
        for name, payload in out.results.components.items()
        if name.startswith("chip_optimization")
    )

    assert np.isfinite(payload["optimization_loss_final"])
    assert payload["delay_state_n_azimuths"] == 2
    assert payload["delay_state_objective"] == "uniform_azimuth_average"
    assert payload["delay_state_azimuths_deg"] == (0.0, 180.0)
    assert payload["delay_state_field_shape"] == (2, 2, 3, 1)
    assert payload["tracker_observability_field_source"] == "delay_state_fields"
    assert payload["tracker_observability_n_field_states"] == 2
    # Observability is judged per delay state (review 2026-11-07 Issue 21).
    for key in (
        "tracker_observability_singular_values_per_nm_per_state_final",
        "amplitude_observability_singular_values_per_state_final",
        "joint_observability_singular_values_per_state_final",
    ):
        assert len(payload[key]) == 2
    assert payload["joint_observability_n_rows"] == 4  # 2 ports x 2 channels


@pytest.mark.parametrize("backend", ["torch"])
def test_optimizer_to_kernel_handoff_reproduces_null_depth(
    backend: str, tmp_path: Path
) -> None:
    """Finding 3 regression: the torch optimizer used to emit Clements
    params in KILO's column-alternating layout, while chip.kernel_mzi_mesh
    reconstructs in physics.clements lexicographic layout. Same param
    vector → different unitary → broken handoff. After the fix, the
    backend canonicalizes its optimal_params before writing them, so the
    kernel reproduces the optimizer's null depth.
    """

    if backend == "torch" and not _torch_available():
        pytest.skip("torch unavailable")

    from nullsim.stages.chip.kernel_mzi_mesh import KernelMZIMesh

    n_modes = 4
    n_dark_ports = 2
    seed = 0
    max_iter = 200

    context = _context(tmp_path, instance_id=f"chip_opt_{backend}_handoff")
    from nullsim.stages.chip_optimization.torch import ChipOptimizationTorch

    opt_cls = ChipOptimizationTorch

    opt_stage = opt_cls(
        {
            "n_modes": n_modes,
            "n_bright_ports": n_modes - n_dark_ports,
            "n_dark_ports": n_dark_ports,
            "wl_design_um": 1.65,
            "max_iter": max_iter,
            "seed": seed,
        },
        context,
    )
    state_after_opt = opt_stage.apply(_uniform_state(n_modes=n_modes))

    chip_stage = KernelMZIMesh(
        {
            "n_modes": n_modes,
            "n_bright_ports": n_modes - n_dark_ports,
            "n_dark_ports": n_dark_ports,
            "wl_design_um": 1.65,
            "optimization_source": "auto",
            "params": None,
            "insertion_loss_db": 0.0,
        },
        _context(tmp_path, instance_id="chip"),
    )
    state_after_chip = chip_stage.apply(state_after_opt)

    port_power = np.sum(np.abs(state_after_chip.field.amplitudes) ** 2, axis=(1, 2))
    dark_idx = np.arange(n_modes - n_dark_ports, n_modes)
    bright_idx = np.arange(0, n_modes - n_dark_ports)
    kernel_null = float(port_power[dark_idx].mean() / port_power[bright_idx].mean())

    components = [
        payload
        for name, payload in state_after_opt.results.components.items()
        if name.startswith("chip_optimization")
    ]
    optimizer_loss = float(components[0]["optimization_loss_final"])

    # The kernel's null depth must reproduce the optimizer's loss to within
    # numerical roundoff from canonicalization + Clements rebuild. We use
    # the same abs_floor + rtol policy as the cross-backend agreement test.
    assert _agreement_ok(kernel_null, optimizer_loss, rtol=1.0e-2, abs_floor=1.0e-8), (
        f"kernel reproduces a different null than the optimizer: "
        f"kernel={kernel_null:.6e}, optimizer={optimizer_loss:.6e}"
    )


@pytest.mark.parametrize("backend", ["torch"])
def test_optimizer_to_kernel_handoff_broadband_uses_monochromatic_kernel(
    backend: str, tmp_path: Path
) -> None:
    """Tier 3.1 (option 2) chromatic-handoff regression test.

    History: the "Ultra-review High #5" pass forced the chip to be
    deployed monochromatically at ``wl_design`` because the canonicalize
    → physics.clements rebuild → per-λ rescale path produced a different
    unitary than the optimizer evaluated (the two Clements layouts
    compose MZIs in different orders, so phase rescaling at the
    parameter level is not invariant under reparameterization).

    Tier 3.1 (option 2) fix: the torch backend now publishes the
    chromatic ``optimal_u_stack`` they actually evaluated (KILO column-
    alternating layout + per-λ phase rescaling), and the chip stage uses
    it verbatim. So the chip is now genuinely chromatic, and the
    deployed unitary matches the optimizer's evaluation at every bin
    rather than collapsing to a monochromatic constant.

    This test pins three things:
      (1) the upstream payload carries ``optimal_u_stack``;
      (2) the chip stage's ``u_stack`` component matches that payload
          bit-for-bit; and
      (3) the per-wavelength output amplitudes are NOT identical across
          bins (the chromatic invariant — the chip varies across the
          band the way a real broadband chip does).
    """

    if backend == "torch" and not _torch_available():
        pytest.skip("torch unavailable")

    from nullsim.stages.chip.kernel_mzi_mesh import KernelMZIMesh

    n_modes = 4
    n_dark_ports = 2
    seed = 0
    max_iter = 200

    # Three-wavelength state: design at 1.65 µm, edges at 1.50 and 1.80.
    wavelength_grid_um = (1.50, 1.65, 1.80)
    amplitudes = np.ones(
        (n_modes, len(wavelength_grid_um), 1), dtype=np.complex128
    ) / np.sqrt(n_modes)
    state = SimulationState.empty({"wavelength_grid_um": wavelength_grid_um})
    state = replace(
        state,
        field=Field(
            amplitudes=amplitudes,
            wavelength_grid_um=wavelength_grid_um,
            mode_labels=tuple(f"mode_{i}" for i in range(n_modes)),
        ),
        throughput=Throughput(
            wavelength_grid_um=wavelength_grid_um,
            transmission_fraction=(1.0,) * len(wavelength_grid_um),
        ),
    )

    context = _context(tmp_path, instance_id=f"chip_opt_{backend}_broadband")
    from nullsim.stages.chip_optimization.torch import ChipOptimizationTorch

    opt_cls = ChipOptimizationTorch

    opt_stage = opt_cls(
        {
            "n_modes": n_modes,
            "n_bright_ports": n_modes - n_dark_ports,
            "n_dark_ports": n_dark_ports,
            "wl_design_um": 1.65,
            "max_iter": max_iter,
            "seed": seed,
        },
        context,
    )
    state_after_opt = opt_stage.apply(state)

    chip_opt_payloads = [
        payload
        for name, payload in state_after_opt.results.components.items()
        if name.startswith("chip_optimization")
    ]
    assert len(chip_opt_payloads) == 1
    payload = chip_opt_payloads[0]
    # Tier 3.1: the chromatic U_stack must be published by the backend.
    if backend == "torch":
        assert "optimal_u_stack" in payload, (
            "Tier 3.1 option 2: torch backend must publish optimal_u_stack."
        )
        assert payload["torch_unitary_builder"] == "batched_clements_ri"
        assert payload["torch_wavelength_batch_size"] == len(wavelength_grid_um)
        assert payload["torch_u_stack_edge_delta_max"] > 1.0e-3
        opt_u_stack = np.asarray(payload["optimal_u_stack"])
        assert opt_u_stack.shape == (n_modes, n_modes, len(wavelength_grid_um))

    chip_stage = KernelMZIMesh(
        {
            "n_modes": n_modes,
            "n_bright_ports": n_modes - n_dark_ports,
            "n_dark_ports": n_dark_ports,
            "wl_design_um": 1.65,
            "optimization_source": "auto",
            "params": None,
            "insertion_loss_db": 0.0,
        },
        _context(tmp_path, instance_id="chip_broadband"),
    )
    state_after_chip = chip_stage.apply(state_after_opt)

    chip_payloads = [
        payload
        for name, payload in state_after_chip.results.components.items()
        if name.startswith("chip.kernel_mzi_mesh")
    ]
    assert len(chip_payloads) == 1
    chip_u_stack = np.asarray(chip_payloads[0]["u_stack"])

    if backend == "torch":
        # Bit-for-bit pass-through: the chip stage published exactly the
        # U_stack the optimizer evaluated.
        assert chip_payloads[0]["precomputed_u_stack_used"] is True
        np.testing.assert_allclose(
            chip_u_stack,
            opt_u_stack,
            atol=1.0e-15,
            rtol=0.0,
        )

    # Chromatic invariant: the per-wavelength output amplitudes are NOT
    # identical across bins (a real broadband chip's response varies
    # across the band). Pre-Tier-3.1 the chip was deployed monochromatic
    # at wl_design and these were identical to machine precision.
    out_amps = state_after_chip.field.amplitudes  # (n_modes, n_wl, n_pol)
    diff_left = np.abs(out_amps[:, 0, :] - out_amps[:, 1, :]).max()
    diff_right = np.abs(out_amps[:, 2, :] - out_amps[:, 1, :]).max()
    assert diff_left > 1.0e-3, (
        f"chip output at λ=1.50 differs from λ=1.65 by only {diff_left:.2e}; "
        "expected a real chromatic spread (Tier 3.1 option 2)."
    )
    assert diff_right > 1.0e-3, (
        f"chip output at λ=1.80 differs from λ=1.65 by only {diff_right:.2e}; "
        "expected a real chromatic spread (Tier 3.1 option 2)."
    )


def test_torch_warm_start_fits_upstream_ustack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The torch backend can initialize from an upstream ideal U_stack."""

    if not _torch_available():
        pytest.skip("torch unavailable")

    import torch
    import nullsim.stages.chip_optimization.torch as torch_backend
    from nullsim.stages.chip_optimization import ChipOptimizationParams, _n_params

    def fake_minimize(fun, x0, **_kwargs):
        return SimpleNamespace(
            fun=float(fun(x0)),
            x=np.asarray(x0, dtype=np.float64),
            nit=0,
            success=True,
        )

    monkeypatch.setattr(torch_backend, "minimize", fake_minimize)

    n_modes = 4
    wavelength_grid_um = (1.20, 1.25, 1.30)
    base_state = SimulationState.empty({"wavelength_grid_um": wavelength_grid_um})
    identity_stack = np.repeat(
        np.eye(n_modes, dtype=np.complex128)[:, :, np.newaxis],
        len(wavelength_grid_um),
        axis=2,
    )
    state = replace(
        base_state,
        results=base_state.results.add_component(
            "chip_optimization",
            {
                "optimal_chip_params": tuple(0.0 for _ in range(_n_params(n_modes))),
                "optimal_u_stack": identity_stack,
                "backend_name": "ideal_kernel_projector",
            },
        ),
    )
    params = ChipOptimizationParams(
        n_modes=n_modes,
        n_bright_ports=2,
        n_dark_ports=2,
        wl_design_um=1.25,
        warm_start_source="auto",
    )

    warm, metadata = torch_backend._resolve_external_warm_start(
        state=state,
        params=params,
        wavelength_grid_um=wavelength_grid_um,
        torch_mod=torch,
        device=torch.device("cpu"),
    )

    assert warm.shape == (_n_params(n_modes),)
    assert metadata["warm_start_source"] == "chip_optimization"
    assert metadata["warm_start_kind"] == "fit_optimal_u_stack"
    assert metadata["warm_start_wavelength_index"] == 1
    assert metadata["warm_start_fit_success"] is True


def test_torch_external_warm_start_can_skip_broadband_refine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: external warm start without bootstrap still reports a loss."""

    if not _torch_available():
        pytest.skip("torch unavailable")

    import nullsim.stages.chip_optimization.torch as torch_backend
    from nullsim.stages.chip_optimization import _n_params
    from nullsim.stages.chip_optimization.torch import ChipOptimizationTorch

    def fake_minimize(fun, x0, **_kwargs):
        return SimpleNamespace(
            fun=float(fun(x0)),
            x=np.asarray(x0, dtype=np.float64),
            nit=0,
            success=True,
        )

    monkeypatch.setattr(torch_backend, "minimize", fake_minimize)

    n_modes = 4
    wavelength_grid_um = (1.20, 1.25, 1.30)
    base_state = _uniform_state(n_modes=n_modes, wl_um=1.25)
    amplitudes = np.ones(
        (n_modes, len(wavelength_grid_um), 1),
        dtype=np.complex128,
    ) / np.sqrt(n_modes)
    identity_stack = np.repeat(
        np.eye(n_modes, dtype=np.complex128)[:, :, np.newaxis],
        len(wavelength_grid_um),
        axis=2,
    )
    state = replace(
        base_state,
        field=Field(
            amplitudes=amplitudes,
            wavelength_grid_um=wavelength_grid_um,
        ),
        throughput=Throughput(
            wavelength_grid_um=wavelength_grid_um,
            transmission_fraction=(1.0,) * len(wavelength_grid_um),
        ),
        results=base_state.results.add_component(
            "chip_optimization",
            {
                "optimal_chip_params": tuple(0.0 for _ in range(_n_params(n_modes))),
                "optimal_u_stack": identity_stack,
                "backend_name": "ideal_kernel_projector",
            },
        ),
    )
    stage = ChipOptimizationTorch(
        {
            "n_modes": n_modes,
            "n_bright_ports": 2,
            "n_dark_ports": 2,
            "wl_design_um": 1.25,
            "warm_start_source": "auto",
            "single_lambda_bootstrap": False,
            "broadband_refine": False,
            "n_restarts": 1,
            "max_iter": 1,
        },
        _context(tmp_path, instance_id="chip_opt_warm_no_refine"),
    )

    out = stage.apply(state)
    payload = out.results.components["chip_optimization.chip_opt_warm_no_refine"]
    assert np.isfinite(payload["optimization_loss_final"])
    assert payload["optimization_iterations"] == 0
    assert payload["warm_start_kind"] == "fit_optimal_u_stack"


def test_torch_parametric_chromaticity_changes_deployed_ustack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Torch deployment uses the shared theta/phase chromaticity model."""

    if not _torch_available():
        pytest.skip("torch unavailable")

    import nullsim.stages.chip_optimization.torch as torch_backend
    from nullsim.stages.chip_optimization.torch import ChipOptimizationTorch

    def fake_minimize(fun, x0, **_kwargs):
        x = np.zeros_like(np.asarray(x0, dtype=np.float64))
        value = fun(x)
        if isinstance(value, tuple):  # fused (loss, grad) objectives
            value = value[0]
        return SimpleNamespace(
            fun=float(value),
            x=x,
            nit=0,
            success=True,
        )

    monkeypatch.setattr(torch_backend, "minimize", fake_minimize)

    n_modes = 2
    wavelength_grid_um = (0.8, 1.0, 1.2)
    amplitudes = np.ones(
        (n_modes, len(wavelength_grid_um), 1),
        dtype=np.complex128,
    ) / np.sqrt(n_modes)
    base_state = SimulationState.empty({"wavelength_grid_um": wavelength_grid_um})
    state = replace(
        base_state,
        field=Field(
            amplitudes=amplitudes,
            wavelength_grid_um=wavelength_grid_um,
        ),
        throughput=Throughput(
            wavelength_grid_um=wavelength_grid_um,
            transmission_fraction=(1.0,) * len(wavelength_grid_um),
        ),
    )
    stage = ChipOptimizationTorch(
        {
            "n_modes": n_modes,
            "n_bright_ports": 1,
            "n_dark_ports": 1,
            "wl_design_um": 1.0,
            "single_lambda_bootstrap": False,
            "n_restarts": 1,
            "max_iter": 1,
            "chip_chromaticity_model": "parametric",
            "theta_chromaticity_slope_rad_per_frac": 0.5,
            "phase_chromaticity_model": "fixed_opd_neff",
            "phase_neff_ref": 2.4,
            "phase_group_index": 4.0,
        },
        _context(tmp_path, instance_id="chip_opt_torch_chromatic"),
    )

    out = stage.apply(state)
    payload = next(
        payload
        for name, payload in out.results.components.items()
        if name.startswith("chip_optimization")
    )
    u_stack = np.asarray(payload["optimal_u_stack"])
    assert payload["chip_chromaticity_model"] == "parametric"
    assert payload["phase_chromaticity_model"] == "fixed_opd_neff"
    assert payload["torch_u_stack_edge_delta_max"] > 1.0e-3
    assert np.max(np.abs(u_stack[:, :, 0] - u_stack[:, :, -1])) > 1.0e-3
