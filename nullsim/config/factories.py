"""Small config-factory helpers for band-sweep construction.

These helpers return plain dictionaries shaped like TOML ``[[sweep]]`` blocks
so notebooks and higher-level instrument factories can build multiband runs
without duplicating NIR band constants or sweep path spelling.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScienceBand:
    """A photometric band grid definition in microns."""

    name: str
    wavelength_center_um: float
    wavelength_bandwidth_um: float


SCIENCE_BANDS: dict[str, ScienceBand] = {
    "Y": ScienceBand("Y", 1.02, 0.10),
    "J": ScienceBand("J", 1.25, 0.16),
    "H": ScienceBand("H", 1.65, 0.30),
    "K": ScienceBand("K", 2.20, 0.34),
}


def science_band(name: str) -> ScienceBand:
    """Return a built-in science-band definition."""
    key = name.upper()
    try:
        return SCIENCE_BANDS[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown science band {name!r}; expected one of {sorted(SCIENCE_BANDS)}."
        ) from exc


def band_grid_sweep_specs(
    bands: tuple[str, ...] = ("Y", "J", "H"),
    *,
    stage_band_params: tuple[str, ...] = (),
) -> tuple[dict[str, object], ...]:
    """Return zipped sweep specs for a multiband grid.

    Args:
        bands: Band names to include.
        stage_band_params: Optional pipeline-stage param paths that should
            receive the band label in the same zip sweep, e.g.
            ``("pipeline.stages.detector.band",)``.

    Returns:
        Tuple of dictionaries compatible with ``[[sweep]]`` TOML blocks.
    """
    band_specs = tuple(science_band(band) for band in bands)
    sweeps: list[dict[str, object]] = [
        {
            "param": "grid.wavelength_center_um",
            "values": [band.wavelength_center_um for band in band_specs],
            "mode": "zip",
        },
        {
            "param": "grid.wavelength_bandwidth_um",
            "values": [band.wavelength_bandwidth_um for band in band_specs],
            "mode": "zip",
        },
    ]
    for param_path in stage_band_params:
        sweeps.append(
            {
                "param": param_path,
                "values": [band.name for band in band_specs],
                "mode": "zip",
            }
        )
    return tuple(sweeps)


__all__ = ["SCIENCE_BANDS", "ScienceBand", "band_grid_sweep_specs", "science_band"]
