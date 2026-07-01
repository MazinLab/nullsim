from __future__ import annotations

import numpy as np
import pytest

from nullsim.pipeline import derive_rng


def test_derive_rng_is_reproducible_for_same_stage_identity() -> None:
    first = derive_rng(123, "config", (("scene.star.jmag", 5.0),), "chip")
    second = derive_rng(123, "config", (("scene.star.jmag", 5.0),), "chip")

    assert np.array_equal(first.random(8), second.random(8))


def test_derive_rng_changes_with_instance_id_and_sweep_coordinate() -> None:
    baseline = derive_rng(123, "config", (("scene.star.jmag", 5.0),), "chip")
    other_instance = derive_rng(123, "config", (("scene.star.jmag", 5.0),), "detector")
    other_coord = derive_rng(123, "config", (("scene.star.jmag", 6.0),), "chip")

    baseline_values = baseline.random(8)
    assert not np.array_equal(baseline_values, other_instance.random(8))
    assert not np.array_equal(baseline_values, other_coord.random(8))


def test_derive_rng_is_independent_of_derivation_order() -> None:
    first_chip = derive_rng(123, "config", (), "chip").random(8)
    _ = derive_rng(123, "config", (), "detector").random(8)
    second_chip = derive_rng(123, "config", (), "chip").random(8)

    assert np.array_equal(first_chip, second_chip)


def test_derive_rng_rejects_invalid_identity() -> None:
    with pytest.raises(ValueError, match="root_seed"):
        derive_rng(-1, "config", (), "chip")

    with pytest.raises(ValueError, match="instance_id"):
        derive_rng(1, "config", (), "")



def test_stage_rng_is_local_to_stage_spec() -> None:
    """A stage's RNG draw must not change when unrelated config content
    (here: the explicit config_hash standing in for any cosmetic edit)
    changes (review 2026-06-10 Issue 7). It must change when the stage's
    own spec changes."""

    from nullsim.pipeline.runner import build_stage_instances

    def _config(config_hash: str, star_jmag: float) -> dict:
        return {
            "run": {"seed": 3},
            "config_hash": config_hash,
            "output": {"dir": "."},
            "pipeline": {
                "stages": [
                    {
                        "id": "scene",
                        "type": "scene.point_source",
                        "star_jmag": star_jmag,
                    },
                ]
            },
        }

    draw_a = build_stage_instances(_config("a" * 64, 4.0))[0].context.rng.integers(
        0, 2**32
    )
    draw_b = build_stage_instances(_config("b" * 64, 4.0))[0].context.rng.integers(
        0, 2**32
    )
    draw_c = build_stage_instances(_config("a" * 64, 8.0))[0].context.rng.integers(
        0, 2**32
    )
    assert draw_a == draw_b  # unrelated config change: same draw
    assert draw_a != draw_c  # stage spec change: new draw
