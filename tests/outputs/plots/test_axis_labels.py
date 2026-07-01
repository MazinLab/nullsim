"""Tests for compact plot legend labels."""

from __future__ import annotations

from nullsim.outputs.plots._axis_labels import compact_record_label


def test_compact_record_label_abbreviates_planet_contrast() -> None:
    label = compact_record_label(
        (("scene.planet.contrast", 1.0e-10),),
        "fallback",
    )

    assert label == "C=1e-10"
