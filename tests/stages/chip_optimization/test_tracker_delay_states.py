"""Tracker observability treats delay states as alternative pointings (Issue 21)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nullsim.physics.fringe_observability import (  # noqa: E402
    differential_piston_basis,
    intensity_response_matrix,
)
from nullsim.stages.chip_optimization.torch import (  # noqa: E402
    _amplitude_observability_loss_torch,
    _amplitude_observability_singular_values_torch,
    _joint_phase_amplitude_observability_crlb_loss_torch,
    _joint_phase_amplitude_observability_loss_torch,
    _joint_phase_amplitude_observability_singular_values_torch,
    _per_delay_state_singular_values_torch,
    _tracker_observability_losses_torch,
    _tracker_observability_singular_values_torch,
    _tracker_per_state_singular_values_torch,
    _tracker_rank_priority_selection_key,
    _tracker_worst_state_losses_torch,
    _worst_delay_state_loss_torch,
)

_N_MODES = 4
_BRIGHT = np.array([0, 1, 2], dtype=np.int64)
_WAVELENGTHS_M = np.array([1.20e-6, 1.30e-6])
_EPS = 1.0e-12


def _unitary(rng: np.random.Generator) -> np.ndarray:
    raw = rng.normal(size=(_N_MODES, _N_MODES)) + 1j * rng.normal(
        size=(_N_MODES, _N_MODES)
    )
    return np.linalg.qr(raw)[0]


def _field(rng: np.random.Generator) -> np.ndarray:
    """One pointing: ``(n_wl, n_modes, 1)``."""

    shape = (_WAVELENGTHS_M.size, _N_MODES, 1)
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def _channel_inputs(u_by_wl: np.ndarray, states: list[np.ndarray]) -> dict:
    """Stack pointings wavelength-major with states innermost, as the stage does."""

    n_states = len(states)
    u = np.repeat(u_by_wl, n_states, axis=0)
    fields = np.stack(states, axis=1).reshape(-1, _N_MODES, 1)
    return dict(
        u_r=torch.from_numpy(u.real.copy()),
        u_i=torch.from_numpy(u.imag.copy()),
        field_r_by_wl=torch.from_numpy(fields.real.copy()),
        field_i_by_wl=torch.from_numpy(fields.imag.copy()),
        wavelengths_m=torch.from_numpy(np.repeat(_WAVELENGTHS_M, n_states)),
        bright_indices=torch.from_numpy(_BRIGHT),
        piston_basis=torch.from_numpy(differential_piston_basis(_N_MODES)),
        eps=_EPS,
        torch_mod=torch,
    )


def _singular_numpy(u_by_wl: np.ndarray, field: np.ndarray) -> np.ndarray:
    """Independent single-pointing Fisher-proxy singular values per nm."""

    response = intensity_response_matrix(
        u_stack=np.moveaxis(u_by_wl, 0, 2),
        input_fields=np.moveaxis(field, 0, 1),
        wavelengths_m=_WAVELENGTHS_M,
        port_indices=_BRIGHT.astype(np.intp),
        wavelength_indices=np.arange(_WAVELENGTHS_M.size, dtype=np.intp),
    )
    projected = response.response_power_per_m @ differential_piston_basis(_N_MODES)
    proxy = projected / np.sqrt(
        np.maximum(response.reference_power[:, np.newaxis], _EPS)
    )
    return np.sort(np.linalg.svd(proxy * 1.0e-9, compute_uv=False))


def test_identical_delay_states_do_not_inflate_tracker_singular_values() -> None:
    rng = np.random.default_rng(5)
    u = _unitary(rng)
    u_by_wl = np.stack([u, u.conj()])
    field = _field(rng)
    n_states = 6

    inputs = _channel_inputs(u_by_wl, [field] * n_states)
    per_state = _tracker_per_state_singular_values_torch(
        n_states=n_states, **inputs
    ).numpy()
    expected = _singular_numpy(u_by_wl, field)

    assert per_state.shape == (n_states, expected.size)
    np.testing.assert_allclose(np.sort(per_state, axis=1), np.tile(expected, (6, 1)),
                               rtol=1e-10)
    # The old stacked matrix scaled every singular value by sqrt(n_states).
    stacked = np.sort(_tracker_observability_singular_values_torch(**inputs).numpy())
    np.testing.assert_allclose(stacked, np.sqrt(n_states) * expected, rtol=1e-10)

    loss_kwargs = dict(min_singular_per_nm=1.0e-2, target_rms_nm=100.0,
                       compute_crlb=True)
    single = _tracker_observability_losses_torch(
        **_channel_inputs(u_by_wl, [field]), **loss_kwargs
    )
    worst = _tracker_worst_state_losses_torch(
        n_states=n_states, **inputs, **loss_kwargs
    )
    for got, want in zip(worst, single):
        assert float(got) == pytest.approx(float(want), rel=1e-10)


def test_worst_delay_state_exposes_pointing_unobservable_alone() -> None:
    rng = np.random.default_rng(11)
    u = _unitary(rng)
    u_by_wl = np.stack([u, u.conj()])
    good = _field(rng)
    blind = _field(rng)
    blind[:, 0, :] = 0.0  # mode 0 dark at this pointing: its piston is invisible
    n_states = 2

    inputs = _channel_inputs(u_by_wl, [good, blind])
    per_state = _tracker_per_state_singular_values_torch(
        n_states=n_states, **inputs
    ).numpy()

    # Wavelength-major, state-innermost layout: each row is its own pointing.
    np.testing.assert_allclose(np.sort(per_state[0]), _singular_numpy(u_by_wl, good),
                               rtol=1e-10)
    np.testing.assert_allclose(np.sort(per_state[1]), _singular_numpy(u_by_wl, blind),
                               rtol=1e-8, atol=1e-14)
    assert per_state[1].min() < 1e-6 * per_state[0].min()
    # Stacking hid the blind pointing behind the good one.
    stacked = _tracker_observability_singular_values_torch(**inputs).numpy()
    assert stacked.min() > 0.5 * per_state[0].min()

    loss_kwargs = dict(min_singular_per_nm=1.0e-2, target_rms_nm=100.0,
                       compute_crlb=True)
    worst = _tracker_worst_state_losses_torch(
        n_states=n_states, **inputs, **loss_kwargs
    )
    blind_only = _tracker_observability_losses_torch(
        **_channel_inputs(u_by_wl, [blind]), **loss_kwargs
    )
    for got, want in zip(worst, blind_only):
        assert float(got) == pytest.approx(float(want), rel=1e-10)

    floor = 0.5 * per_state[0].min()
    key = _tracker_rank_priority_selection_key(
        singular_values_per_nm=per_state,
        n_dof=per_state.shape[1],
        min_singular_per_nm=floor,
        crlb_target_nm=100.0,
        objective_loss=0.0,
        normalized_null=0.0,
        target_null_depth=1.0,
        eps=_EPS,
    )
    assert key[0] == 1.0  # rank deficit of the blind state, not of the stack


def _amplitude_joint_kwargs(inputs: dict) -> tuple[dict, dict]:
    """Amplitude and joint helper kwargs on the same stacked channels."""

    shared = {key: inputs[key] for key in (
        "u_r", "u_i", "field_r_by_wl", "field_i_by_wl", "eps", "torch_mod"
    )}
    basis = inputs["piston_basis"]
    amplitude = dict(shared, port_indices=inputs["bright_indices"],
                     amplitude_basis=basis)
    joint = dict(amplitude, wavelengths_m=inputs["wavelengths_m"],
                 piston_basis=basis, amplitude_scale=0.02)
    return amplitude, joint


def test_amplitude_and_joint_observability_judge_each_delay_state() -> None:
    rng = np.random.default_rng(17)
    u = _unitary(rng)
    u_by_wl = np.stack([u, u.conj()])
    good = _field(rng)
    # A weak pointing with mode 0 dark (heavy delay-line loss, say).
    blind = 0.3 * good
    blind[:, 0, :] = 0.0

    # Identical states: per-state values equal the single pointing, and the
    # stacked matrix carries the sqrt(n_states) inflation.
    n_states = 6
    amp1, joint1 = _amplitude_joint_kwargs(_channel_inputs(u_by_wl, [good]))
    amp6, joint6 = _amplitude_joint_kwargs(
        _channel_inputs(u_by_wl, [good] * n_states)
    )
    for fn, single, stacked in (
        (_amplitude_observability_singular_values_torch, amp1, amp6),
        (_joint_phase_amplitude_observability_singular_values_torch, joint1, joint6),
    ):
        expected = np.sort(fn(**single).numpy())
        per_state = _per_delay_state_singular_values_torch(
            fn, n_states, **stacked
        ).numpy()
        np.testing.assert_allclose(np.sort(per_state, axis=1),
                                   np.tile(expected, (n_states, 1)), rtol=1e-10)
        np.testing.assert_allclose(np.sort(fn(**stacked).numpy()),
                                   np.sqrt(n_states) * expected, rtol=1e-10)

    # A pointing blind to one mode: the worst-state penalty is the blind
    # state's, which the stacked matrix hid behind the good pointing.
    amp2, joint2 = _amplitude_joint_kwargs(_channel_inputs(u_by_wl, [good, blind]))
    amp_b, joint_b = _amplitude_joint_kwargs(_channel_inputs(u_by_wl, [blind]))

    def floor_between_states(fn, kwargs: dict) -> float:
        good_min, blind_min = _per_delay_state_singular_values_torch(
            fn, 2, **kwargs
        ).numpy().min(axis=1)
        assert blind_min < 0.5 * good_min
        # Not the geometric mean: the blind mode's singular value is exactly 0
        # on some LAPACK builds (Linux CI), which would make the floor 0 and
        # the floor-normalized hinge 0/0.
        return float(0.5 * good_min)

    amp_floor = floor_between_states(
        _amplitude_observability_singular_values_torch, amp2
    )
    joint_floor = floor_between_states(
        _joint_phase_amplitude_observability_singular_values_torch, joint2
    )
    cases = (
        (_amplitude_observability_loss_torch, amp2, amp_b,
         dict(min_singular_per_fractional_intensity=amp_floor)),
        (_joint_phase_amplitude_observability_loss_torch, joint2, joint_b,
         dict(min_singular_per_scaled_mode=joint_floor)),
        (_joint_phase_amplitude_observability_crlb_loss_torch, joint2, joint_b,
         dict(target_per_scaled_mode=100.0)),
    )
    for fn, both, blind_only, extra in cases:
        worst = _worst_delay_state_loss_torch(fn, 2, **both, **extra)
        assert float(worst) == pytest.approx(
            float(fn(**blind_only, **extra)), rel=1e-10
        )
        assert float(worst) > float(fn(**both, **extra))
