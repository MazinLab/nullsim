"""Inject top-level config-section defaults into stage params.

The user-facing TOML lets users write physics in dedicated `[scene.star]`,
`[site.atmosphere]`, `[telescope.aperture]`, etc. sections (per SPEC §7.1).
The pipeline stages, however, receive their inputs through their own validated
`Params` model. This module bridges the two: at config-load / sweep-expansion
time, it copies the relevant top-level fields into each stage's spec.

Living in the config layer (not the runner) keeps the pipeline runner
agnostic of specific stage types.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

# Stage-spec keys that are structural (identity / extends-merge directives),
# never user physics params. Everything else on a raw stage spec counts as a
# user-explicit param for materialization-precedence purposes.
_RESERVED_STAGE_SPEC_KEYS = frozenset(
    {"id", "type", "remove", "replace", "before", "after"}
)

_PORT_TOPOLOGY_BINDINGS = {
    "n_modes": "instrument.n_modes",
    "n_bright_ports": "instrument.n_bright_ports",
    "n_dark_ports": "instrument.n_dark_ports",
    "n_spectro_ports": "instrument.n_spectro_ports",
}
_PORT_ONLY_BINDINGS = {
    key: value for key, value in _PORT_TOPOLOGY_BINDINGS.items() if key != "n_modes"
}
_INPUT_MODE_BINDING = {"n_modes": "instrument.n_input_modes"}

# Per-stage-type mapping of param-name -> dotted top-level config path.
# Add an entry here when a new stage wants user-facing config syntax in a
# dedicated [section] block instead of inline stage params.
_STAGE_CONFIG_BINDINGS: dict[str, dict[str, str]] = {
    "scene.point_source": {
        "star_jmag": "scene.star.jmag",
        "star_angular_diameter_mas": "scene.star.angular_diameter_mas",
        "star_distance_pc": "scene.star.distance_pc",
        "star_spectral_type": "scene.star.spectral_type",
        "planet_contrast": "scene.planet.contrast",
        "planet_separation_mas": "scene.planet.separation_mas",
        "position_angle_deg": "scene.planet.position_angle_deg",
        "exozodi_level_zodi": "scene.exozodi.level_zodi",
    },
    "telescope.fixed_aperture": {
        "diameter_m": "telescope.aperture.diameter_m",
        "n_apertures": "telescope.n_apertures",
        "mirror_transmission": "telescope.optics.mirror_transmission",
        "ao_bench_transmission": "telescope.optics.ao_bench_transmission",
        "central_obstruction_ratio": "telescope.aperture.obscuration_fraction",
    },
    "atmosphere.kolmogorov": {
        "r0_500nm_m": "site.atmosphere.r0_500nm_m",
        "L0_m": "site.atmosphere.L0_m",
        "wind_speed_m_s": "site.atmosphere.wind_speed_m_s",
        "atmospheric_transmission": "site.atmosphere.atmospheric_transmission",
    },
    "ao.pyramid_wfs": {
        "central_obstruction": "telescope.aperture.obscuration_fraction",
        "tau0_500nm_s": "site.atmosphere.tau0_500nm_s",
        "theta0_500nm_arcsec": "site.atmosphere.theta0_500nm_arcsec",
        "v_wind_m_s": "site.atmosphere.wind_speed_m_s",
        "pupil_central_obstruction_diameter_m": (
            "telescope.aperture.central_obstruction_diameter_m"
        ),
    },
    "injection.ideal": {
        "central_obscuration_ratio": "telescope.aperture.obscuration_fraction",
        **_INPUT_MODE_BINDING,
    },
    "injection.single_mode_fiber": {
        "central_obscuration_ratio": "telescope.aperture.obscuration_fraction",
    },
    "chip.kernel_mzi_mesh": _PORT_TOPOLOGY_BINDINGS,
    "chip_optimization.scipy": _PORT_TOPOLOGY_BINDINGS,
    "chip_optimization.torch": _PORT_TOPOLOGY_BINDINGS,
    "chip_optimization.ideal_kernel": _PORT_TOPOLOGY_BINDINGS,
    "detector.mkid": {
        **_PORT_TOPOLOGY_BINDINGS,
        "integration_time_s": "observation.integration_time_s",
    },
    "detector.snspd": {
        **_PORT_TOPOLOGY_BINDINGS,
        "integration_time_s": "observation.integration_time_s",
    },
    "detector.ideal_counter": {
        "integration_time_s": "observation.integration_time_s",
    },
    "background.sky_surface_brightness": {
        "site_profile": "site.sky.profile",
    },
    "fringe_tracking.closed_loop": {
        "wind_speed_m_s": "site.atmosphere.wind_speed_m_s",
        "outer_scale_m": "site.atmosphere.L0_m",
        **_INPUT_MODE_BINDING,
    },
    "sensitivity.detection_curve": _PORT_ONLY_BINDINGS,
    "characterization.spectro": _PORT_TOPOLOGY_BINDINGS,
    "characterization.detection_curve": _PORT_TOPOLOGY_BINDINGS,
}


def materialize_pipeline_stages(
    config_data: MutableMapping[str, Any],
    *,
    overwrite: bool = False,
    skip_stage_params: frozenset[tuple[str, str]] | None = None,
) -> None:
    """In-place fill stage specs with defaults pulled from sibling sections.

    Args:
        config_data: Top-level config (mutated in place).
        overwrite: When True, always copy from the top-level binding
            source even if the stage spec already has a value. The
            sweep-cell path passes ``overwrite=True`` because it has
            just rewritten the top-level binding source (e.g.
            ``scene.star.jmag``) and the stage spec carries a stale
            value from the prior ``load_config`` materialize. Default
            False preserves the original idempotent behavior.
        skip_stage_params: ``{(stage_id, param_name)}`` pairs the sweep
            cell wrote directly on the stage spec. The materializer must
            not overwrite these from the un-swept top-level binding
            source: a sweep over ``pipeline.stages.<id>.<param>`` is the
            authoritative value, and re-materializing from the un-swept
            top-level mirror would silently produce N identical cells.
    """
    pipeline = config_data.get("pipeline")
    if not isinstance(pipeline, MutableMapping):
        return
    stages = pipeline.get("stages")
    # Accept both list (raw TOML load) and tuple (post-pydantic
    # model_dump). The expand_sweeps round-trip serializes back to
    # tuple, so the early return on list-only used to make the sweep
    # cell's re-materialize a silent no-op.
    if not isinstance(stages, (list, tuple)):
        return
    skips = skip_stage_params or frozenset()
    for spec in stages:
        if not isinstance(spec, MutableMapping):
            continue
        bindings = _STAGE_CONFIG_BINDINGS.get(spec.get("type"))
        if not bindings:
            continue
        stage_id = spec.get("id")
        for param_name, dotted_path in bindings.items():
            if isinstance(stage_id, str) and (stage_id, param_name) in skips:
                continue
            if not overwrite and spec.get(param_name) is not None:
                continue
            value = _walk(config_data, dotted_path)
            if value is not None:
                spec[param_name] = value


def explicit_stage_params(raw_config: Any) -> frozenset[tuple[str, str]]:
    """Return ``(stage_id, param_name)`` pairs the user wrote on stage specs.

    Reads the *raw* (pre-materialization) config, so the result reflects
    only user-authored stage params — not values a prior materialize pass
    injected. The sweep-cell re-materializer uses this to preserve explicit
    stage-level overrides: without it, ``overwrite=True`` re-copies every
    binding from the top-level section and the same config produces
    different stage params with and without a ``[[sweep]]`` block
    (review 2026-06-10 Issue 6).
    """

    if not isinstance(raw_config, Mapping):
        return frozenset()
    pipeline = raw_config.get("pipeline")
    if not isinstance(pipeline, Mapping):
        return frozenset()
    stages = pipeline.get("stages")
    if not isinstance(stages, (list, tuple)):
        return frozenset()
    explicit: set[tuple[str, str]] = set()
    for spec in stages:
        if not isinstance(spec, Mapping):
            continue
        stage_id = spec.get("id")
        if not isinstance(stage_id, str) or not stage_id:
            continue
        for key in spec:
            if str(key) not in _RESERVED_STAGE_SPEC_KEYS:
                explicit.add((stage_id, str(key)))
    return frozenset(explicit)


def stage_binding_sources(stage_type: Any) -> dict[str, str]:
    """Return ``{param_name: top_level_dotted_path}`` bindings for a stage type."""

    bindings = _STAGE_CONFIG_BINDINGS.get(stage_type)
    return dict(bindings) if bindings else {}


def bound_top_level_paths() -> frozenset[str]:
    """All top-level dotted paths that materialize into some stage param."""

    return frozenset(
        dotted_path
        for bindings in _STAGE_CONFIG_BINDINGS.values()
        for dotted_path in bindings.values()
    )


def _walk(config: Any, dotted_path: str) -> Any:
    node: Any = config
    for part in dotted_path.split("."):
        if isinstance(node, MutableMapping) and part in node:
            node = node[part]
        else:
            return None
    return node


__all__ = [
    "bound_top_level_paths",
    "explicit_stage_params",
    "materialize_pipeline_stages",
    "stage_binding_sources",
]
