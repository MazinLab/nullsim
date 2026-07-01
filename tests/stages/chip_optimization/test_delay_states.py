"""Delay-state replay conventions for the chip optimizer."""

from __future__ import annotations

import numpy as np
import pytest

def test_reported_null_depth_is_state_count_invariant() -> None:
    """An all-leak chip must report null depth 1.0 regardless of how many
    delay states replay the same rows. Pre-fix the state multiplicity was
    double-counted (rows weighted 1/sqrt(n) AND the denominator counting
    state-multiplied rows), deflating every reported null by exactly
    n_states — 6x optimistic in kilo_keck_6 (review 2026-06-10 Issue 1)."""

    from nullsim.stages.chip_optimization import (
        _normalization_field_array,
        _reported_null_depth_for_target,
    )

    rng = np.random.default_rng(7)
    n_modes, n_wl, n_pol = 4, 3, 1
    field = (
        rng.normal(size=(n_modes, n_wl, n_pol))
        + 1j * rng.normal(size=(n_modes, n_wl, n_pol))
    )

    for n_states in (1, 3, 6):
        # Identical delay states: replaying the same field n_states times.
        delay_state_fields = np.broadcast_to(
            field, (n_states, n_modes, n_wl, n_pol)
        ).copy()
        normalization = _normalization_field_array(field, delay_state_fields)
        state_weight = 1.0 / np.sqrt(float(n_states))
        # One objective row per state (kernel_order=0 analogue), each
        # weighted by 1/sqrt(n_states): an identity "chip" that leaks 100%
        # of stellar power at the null ports gives summed leakage equal to
        # the per-state mean power.
        rows = np.concatenate(
            [field[np.newaxis, ...] * state_weight for _ in range(n_states)],
            axis=0,
        )
        leak_per_wl = np.sum(np.abs(rows) ** 2, axis=(0, 1, 3))
        loss = float(np.mean(leak_per_wl))
        reported = _reported_null_depth_for_target(
            loss,
            normalization,
            n_objective_rows=n_states,
            n_states=n_states,
        )
        assert reported == pytest.approx(1.0, rel=1e-12), (
            f"n_states={n_states}: all-leak chip must report 1.0, "
            f"got {reported}"
        )


def test_reported_null_denominator_rejects_indivisible_rows() -> None:
    """Stacked rows must be a whole number of per-state row sets."""

    from nullsim.stages.chip_optimization import _reported_null_depth_denominator

    field = np.ones((2, 4, 2, 1), dtype=np.complex128)
    with pytest.raises(ValueError, match="multiple"):
        _reported_null_depth_denominator(field, n_objective_rows=3, n_states=2)
