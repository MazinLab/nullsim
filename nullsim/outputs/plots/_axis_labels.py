"""Compact sweep-axis label rendering.

NAYRA-style configs zip many physical axes (n_modes, n_bright_ports,
n_dark_ports, n_pixels_dark, ...) onto a single sub-aperture-count
sweep ``N``. Naive labels like
``jmag=5, pipeline.stages.chip_opt.n_modes=18, ...`` rapidly outgrow
the legend area. ``compact_record_label`` collapses to the
human-meaningful axes only — ``J=5, N=18``.
"""

from __future__ import annotations

from typing import Any

_AXIS_ABBREV: dict[str, str] = {
    "jmag": "J",
    "n_subapertures_per_aperture": "N",
    "kernel_order": "K",
    "contrast": "C",
}

# Axes whose values are zipped to (and entirely derived from) the
# sub-aperture count axis ``N``. Dropping them from the legend keeps
# labels short without losing information.
_AXIS_DERIVED_FROM_N: frozenset[str] = frozenset({
    "n_modes",
    "n_bright_ports",
    "n_dark_ports",
    "n_spectro_ports",
    "n_pixels_bright",
    "n_pixels_dark",
})


def short_axis_name(axis_name: str) -> str | None:
    """Return a compact axis name, or ``None`` to drop the axis.

    Returns:
        - For canonical axes (``jmag``, ``n_subapertures_per_aperture``):
          a single-letter abbreviation (``J``, ``N``).
        - For axes derived from ``N`` in zipped NAYRA-style sweeps:
          ``None`` — the caller drops the entry.
        - For anything else: the stage-prefix-stripped field name.
    """

    if axis_name.startswith("pipeline.stages."):
        leaf = axis_name.split(".", 3)[-1]
    elif axis_name.startswith("scene.star."):
        leaf = axis_name.split(".", 2)[-1]
    elif axis_name.startswith("scene.planet."):
        leaf = axis_name.split(".", 2)[-1]
    else:
        leaf = axis_name
    if leaf in _AXIS_ABBREV:
        return _AXIS_ABBREV[leaf]
    if leaf in _AXIS_DERIVED_FROM_N:
        return None
    return leaf


def compact_record_label(coord_tuple: tuple[tuple[str, Any], ...] | None,
                         fallback: str) -> str:
    """Render a sweep coord tuple as ``J=5, N=18`` for plot legends.

    Dedupes (short, value) pairs so zipped axes that share a value
    (e.g. ``n_modes=18`` under chip, chip_opt, detector, ...) appear
    once. Returns ``fallback`` when no canonical axes are present.
    """

    if not coord_tuple:
        return fallback
    seen: set[tuple[str, str]] = set()
    parts: list[str] = []
    for axis_name, axis_value in coord_tuple:
        short = short_axis_name(axis_name)
        if short is None:
            continue
        value_str = (
            f"{axis_value:g}" if isinstance(axis_value, float) else str(axis_value)
        )
        key = (short, value_str)
        if key in seen:
            continue
        seen.add(key)
        parts.append(f"{short}={value_str}")
    return ", ".join(parts) if parts else fallback
