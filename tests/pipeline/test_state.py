from __future__ import annotations

import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest

from nullsim.pipeline import (
    Geometry,
    Results,
    Scene,
    SimulationState,
)

EXPECTED_EMPTY_STATE_DIGEST_HEX = (
    "72ae107906862a7ec4bb109cd38e77dd275ef9919825fe523c6d6e1df081ce93"
)


def test_add_component_returns_new_immutable_instance() -> None:
    scene = Scene(star_jmag=5.0)

    updated_scene = scene.add_component("stellar_sed", np.array([1.0, 0.5]))

    assert updated_scene is not scene
    assert "stellar_sed" not in scene.components
    assert np.array_equal(updated_scene.components["stellar_sed"], np.array([1.0, 0.5]))
    with pytest.raises(FrozenInstanceError):
        updated_scene.star_jmag = 6.0  # type: ignore[misc]
    with pytest.raises(TypeError):
        updated_scene.components["planet"] = 1.0  # type: ignore[index]


def test_nested_component_payloads_are_immutable() -> None:
    results = Results(components={"diagnostic": {"value": 1.0}})
    digest_before = results.content_digest()

    with pytest.raises(TypeError):
        results.components["diagnostic"]["value"] = 2.0

    assert results.components["diagnostic"] == {"value": 1.0}
    assert results.content_digest() == digest_before


def test_empty_state_digest_matches_checked_in_value() -> None:
    assert SimulationState.empty().content_digest().hex() == EXPECTED_EMPTY_STATE_DIGEST_HEX


def test_empty_state_digest_matches_checked_in_value_across_process_restart() -> None:
    project_root = Path(__file__).parents[2]
    code = (
        "from nullsim.pipeline import SimulationState; "
        "print(SimulationState.empty().content_digest().hex())"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == EXPECTED_EMPTY_STATE_DIGEST_HEX


def test_content_digest_changes_when_any_budget_gets_a_contribution() -> None:
    state = SimulationState.empty()
    original_digest = state.content_digest()

    for budget_name in SimulationState.budget_names:
        budget = getattr(state, budget_name)
        updated_budget = budget.add_component("test_component", budget_name)
        updated_state = replace(state, **{budget_name: updated_budget})

        assert updated_state.content_digest() != original_digest


def test_content_digest_for_scene_ignores_throughput_contributions() -> None:
    state = SimulationState.empty()
    updated_state = replace(
        state,
        throughput=state.throughput.add_component("fiber", np.array([0.81, 0.82])),
    )

    assert updated_state.content_digest() != state.content_digest()
    assert updated_state.content_digest_for({"scene"}) == state.content_digest_for({"scene"})
    assert updated_state.content_digest_for({"throughput"}) != state.content_digest_for(
        {"throughput"}
    )


def test_subbudget_arrays_are_defensively_copied_and_readonly() -> None:
    baselines = np.array([1.0, 2.0], dtype=np.float64)
    geometry = Geometry(baselines_m=baselines)
    digest_before = geometry.content_digest()

    baselines[0] = 99.0

    assert geometry.content_digest() == digest_before
    assert geometry.baselines_m[0] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="read-only"):
        geometry.baselines_m[1] = 3.0


def test_wavelength_grid_centers_match_configured_bandwidth() -> None:
    """Phase 6: total integrated bandwidth must equal the configured
    wavelength_bandwidth_um (using mkid's half-bin edge extrapolation).

    Pre-Phase-6 the grid generator used ``np.linspace(start, stop,
    n_bins)`` which placed samples AT the band edges, then mkid's
    half-bin extrapolation added an extra step at each end (20 bins
    over 0.15 um integrated to ~0.158 um instead of 0.15).
    """
    import numpy as np

    state = SimulationState.empty(
        {
            "wavelength_center_um": 1.25,
            "wavelength_bandwidth_um": 0.15,
            "n_wavelength_bins": 20,
        }
    )
    wl = np.asarray(state.field.wavelength_grid_um)
    assert wl.size == 20
    # Bin centers, NOT edges: the first/last samples should be half a
    # bin in from the band edges.
    step = 0.15 / 20.0
    assert wl[0] == pytest.approx(1.25 - 0.15 / 2.0 + step / 2.0)
    assert wl[-1] == pytest.approx(1.25 + 0.15 / 2.0 - step / 2.0)
    # mkid's half-bin extrapolation now recovers the configured bw.
    edge_low = wl[0] - 0.5 * (wl[1] - wl[0])
    edge_high = wl[-1] + 0.5 * (wl[-1] - wl[-2])
    assert edge_high - edge_low == pytest.approx(0.15, rel=1e-10)
