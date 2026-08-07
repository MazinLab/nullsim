"""Spectral template data layer for high dispersion coronagraphy (HDC).

Primitives for the HDC cross-correlation module described in
``docs/superpowers/specs/2026-08-07-hdc-ccf-design.md``: a validated template
spectrum container, a CSV loader with provenance, constant-resolving-power
grids, flux-conserving resampling, radial-velocity shifting, and Gaussian
line-spread-function degradation between two resolving powers.

The physical framing follows Wang, Mawet, Ruane, Hu & Benneke 2017, AJ 153,
183 ("Observing Exoplanets with High Dispersion Coronagraphy. I."): the planet
signal lives in a forest of narrow lines while the leaked starlight systematic
is spectrally smooth, so the analysis has to be done on a grid that resolves
the lines (a constant-R log grid) and then degraded to the instrument
resolving power.

Pure numpy/scipy: no TOML, ``SimulationState``, or stage imports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import gaussian_filter1d

C_KMS: float = 299792.458
"""Speed of light in km/s."""

_KINDS = frozenset({"albedo", "transmission", "stellar_flux"})

_TRANSMISSION_TOL = 1e-9

# Gaussian kernel half-width in sigma. Wider than the scipy default of 4 so
# that successive degradations compose to machine precision: truncating at 4
# sigma leaves ~1e-4 relative tail error, which shows up when comparing a
# single R_grid -> R_out hop against R_grid -> R_mid -> R_out.
_KERNEL_TRUNCATE = 8.0


@dataclass(frozen=True, eq=False)
class TemplateSpectrum:
    """A one-dimensional template spectrum on an ascending wavelength grid.

    Parameters
    ----------
    wavelength_um:
        Strictly ascending, positive, finite wavelengths in microns.
    values:
        Spectrum values on ``wavelength_um``. Interpretation is set by ``kind``.
    kind:
        One of ``"albedo"``, ``"transmission"``, ``"stellar_flux"``.
        ``"transmission"`` values are constrained to ``[0, 1]``; the others
        must be non-negative.
    provenance:
        Free-form origin string (model, snapshot date, citation).
    """

    wavelength_um: NDArray[np.float64]
    values: NDArray[np.float64]
    kind: str
    provenance: str = ""

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(
                f"unknown spectrum kind {self.kind!r}; expected one of "
                f"{sorted(_KINDS)}"
            )
        wavelength = np.asarray(self.wavelength_um, dtype=np.float64)
        values = np.asarray(self.values, dtype=np.float64)
        if wavelength.ndim != 1 or values.ndim != 1:
            raise ValueError(
                "wavelength_um and values must be 1-D, got shapes "
                f"{wavelength.shape} and {values.shape}"
            )
        if wavelength.shape != values.shape:
            raise ValueError(
                "wavelength_um and values must have the same shape, got "
                f"{wavelength.shape} and {values.shape}"
            )
        if wavelength.size < 2:
            raise ValueError("a template spectrum needs at least 2 samples")
        if not np.all(np.isfinite(wavelength)) or not np.all(np.isfinite(values)):
            raise ValueError("wavelength_um and values must all be finite")
        if np.any(wavelength <= 0.0):
            raise ValueError("wavelength_um must be strictly positive")
        if not np.all(np.diff(wavelength) > 0.0):
            raise ValueError("wavelength_um must be strictly ascending")
        if self.kind == "transmission":
            if np.any(values < 0.0) or np.any(values > 1.0 + _TRANSMISSION_TOL):
                raise ValueError("transmission values must lie in [0, 1]")
        elif np.any(values < 0.0):
            raise ValueError(f"{self.kind} values must be non-negative")
        object.__setattr__(self, "wavelength_um", wavelength)
        object.__setattr__(self, "values", values)


_PACKAGED_SPECTRA: dict[str, str] = {
    "earth": "earth_albedo.csv",
    "venus": "venus_albedo.csv",
    "g": "stellar_g2v.csv",
    "k": "stellar_k2v.csv",
    "m": "stellar_m2v.csv",
    "maunakea": "telluric_maunakea.csv",
    "armazones": "telluric_armazones.csv",
}


def packaged_spectrum_path(name: str) -> Path:
    """Resolve a short template name to its packaged CSV under ``data/spectra``.

    Names are case-insensitive:

    - ``"earth"``, ``"venus"`` -- disk-integrated *apparent albedo* spectra at
      quadrature phase (spectral shape; the absolute level is approximate and
      is not the zero-phase geometric albedo).
    - ``"G"``, ``"K"``, ``"M"`` -- PHOENIX stellar photon-flux shapes with
      arbitrary normalization.
    - ``"maunakea"``, ``"armazones"`` -- telluric transmission snapshots.

    Every file carries a full provenance header; read it before using the
    values for anything but their spectral shape. Raises ``KeyError`` listing
    the valid names for anything else.

    The returned path is not checked for existence; `load_template_csv` reports
    a missing file.
    """
    key = str(name).strip().lower()
    try:
        filename = _PACKAGED_SPECTRA[key]
    except KeyError:
        raise KeyError(
            f"unknown packaged spectrum {name!r}; valid names are "
            f"{sorted(_PACKAGED_SPECTRA)}"
        ) from None
    return Path(__file__).resolve().parent.parent / "data" / "spectra" / filename


def load_template_csv(path: Path, kind: str | None = None) -> TemplateSpectrum:
    """Load a two-column ``wavelength_um,value`` CSV into a `TemplateSpectrum`.

    Leading ``# key: value`` comment lines are collected into ``provenance``
    (joined with ``"; "``). A ``# kind: ...`` comment supplies ``kind`` when the
    argument is omitted; supplying both is allowed only if they agree. An
    optional ``wavelength_um,value`` header row is skipped.
    """
    path = Path(path)
    provenance_parts: list[str] = []
    header_kind: str | None = None
    wavelengths: list[float] = []
    values: list[float] = []

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            comment = line.lstrip("#").strip()
            if not comment:
                continue
            provenance_parts.append(comment)
            key, sep, value = comment.partition(":")
            if sep and key.strip().lower() == "kind":
                header_kind = value.strip()
            continue
        fields = [f for f in line.replace(",", " ").split() if f]
        if len(fields) != 2:
            raise ValueError(
                f"{path}: expected two columns per data row, got {raw_line!r}"
            )
        try:
            wavelength, value = float(fields[0]), float(fields[1])
        except ValueError:
            if not wavelengths:
                continue  # column header row
            raise ValueError(f"{path}: unparsable data row {raw_line!r}") from None
        wavelengths.append(wavelength)
        values.append(value)

    if not wavelengths:
        raise ValueError(f"{path}: no data rows found")

    if kind is None:
        if header_kind is None:
            raise ValueError(
                f"{path}: no '# kind:' header and no kind argument supplied"
            )
        resolved_kind = header_kind
    elif header_kind is not None and header_kind != kind:
        raise ValueError(
            f"{path}: kind argument {kind!r} disagrees with file header "
            f"{header_kind!r}"
        )
    else:
        resolved_kind = kind

    return TemplateSpectrum(
        wavelength_um=np.asarray(wavelengths, dtype=np.float64),
        values=np.asarray(values, dtype=np.float64),
        kind=resolved_kind,
        provenance="; ".join(provenance_parts),
    )


def constant_r_grid(
    lambda_min_um: float, lambda_max_um: float, resolving_power: float
) -> NDArray[np.float64]:
    """Log-uniform wavelength grid with one sample per resolution element.

    ``n = ceil(ln(lambda_max / lambda_min) * R) + 1`` samples, spaced by
    ``d ln lambda = 1 / R``, starting exactly at ``lambda_min_um`` and covering
    at least up to ``lambda_max_um``.
    """
    if lambda_min_um <= 0.0:
        raise ValueError("lambda_min_um must be strictly positive")
    if lambda_max_um <= lambda_min_um:
        raise ValueError("lambda_max_um must exceed lambda_min_um")
    if resolving_power <= 0.0 or not math.isfinite(resolving_power):
        raise ValueError("resolving_power must be positive and finite")
    n = int(math.ceil(math.log(lambda_max_um / lambda_min_um) * resolving_power)) + 1
    return lambda_min_um * np.exp(np.arange(n, dtype=np.float64) / resolving_power)


def interp_to_grid(
    spectrum: TemplateSpectrum, wavelength_um: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Linearly interpolate a smooth spectrum onto ``wavelength_um``.

    Edge-clamped (``np.interp`` semantics): appropriate for smoothly varying
    shapes such as albedo, transmission, and throughput. Use
    `resample_flux_conserving` for anything whose band integral must survive.
    """
    target = np.asarray(wavelength_um, dtype=np.float64)
    if target.ndim != 1:
        raise ValueError(f"wavelength_um must be 1-D, got shape {target.shape}")
    return np.interp(target, spectrum.wavelength_um, spectrum.values)


def resample_flux_conserving(
    wavelength_um: NDArray[np.float64],
    values: NDArray[np.float64],
    target_edges_um: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Bin-average ``values`` into the bins defined by ``target_edges_um``.

    Integrates the source spectrum with a cumulative trapezoid, differences the
    cumulative integral at the bin edges, and divides by the bin widths. The
    returned bin means therefore reproduce the source integral over the covered
    band exactly (up to the trapezoid rule on the source grid).
    """
    wavelength = np.asarray(wavelength_um, dtype=np.float64)
    source = np.asarray(values, dtype=np.float64)
    edges = np.asarray(target_edges_um, dtype=np.float64)
    if wavelength.ndim != 1 or source.ndim != 1 or edges.ndim != 1:
        raise ValueError("wavelength_um, values, and target_edges_um must be 1-D")
    if wavelength.shape != source.shape:
        raise ValueError(
            "wavelength_um and values must have the same shape, got "
            f"{wavelength.shape} and {source.shape}"
        )
    if wavelength.size < 2:
        raise ValueError("source grid needs at least 2 samples")
    if edges.size < 2:
        raise ValueError("target_edges_um needs at least 2 edges")
    if not np.all(np.diff(wavelength) > 0.0):
        raise ValueError("wavelength_um must be strictly ascending")
    if not np.all(np.diff(edges) > 0.0):
        raise ValueError("target_edges_um must be strictly ascending")
    if edges[0] < wavelength[0] or edges[-1] > wavelength[-1]:
        raise ValueError(
            f"target edges [{edges[0]}, {edges[-1]}] extend beyond the source "
            f"grid [{wavelength[0]}, {wavelength[-1]}]"
        )
    cum = cumulative_trapezoid(source, wavelength, initial=0.0)
    integral_per_bin = np.diff(np.interp(edges, wavelength, cum))
    return integral_per_bin / np.diff(edges)


def doppler_shift(spectrum: TemplateSpectrum, rv_kms: float) -> TemplateSpectrum:
    """Shift a spectrum by a radial velocity (positive = receding = redshift).

    Non-relativistic: ``lambda -> lambda * (1 + rv / c)``. Values are unchanged,
    so this is the correct operation for dimensionless shapes (albedo,
    transmission) and for flux densities whose grid is carried along.
    """
    if not math.isfinite(rv_kms):
        raise ValueError("rv_kms must be finite")
    factor = 1.0 + rv_kms / C_KMS
    if factor <= 0.0:
        raise ValueError(f"rv_kms = {rv_kms} implies a non-physical shift factor")
    note = f"doppler_shift rv_kms={rv_kms:+.6g}"
    provenance = f"{spectrum.provenance}; {note}" if spectrum.provenance else note
    return replace(
        spectrum,
        wavelength_um=spectrum.wavelength_um * factor,
        provenance=provenance,
    )


def convolve_to_resolving_power(
    wavelength_um: NDArray[np.float64],
    values: NDArray[np.float64],
    r_out: float,
    r_in: float = math.inf,
) -> NDArray[np.float64]:
    """Degrade a log-uniform spectrum from resolving power ``r_in`` to ``r_out``.

    On a constant-R grid a Gaussian line-spread function has a width that is
    constant in samples, so the degradation is a single ``gaussian_filter1d``.
    The broadening actually applied is the quadrature difference
    ``1 / r_eff**2 = 1 / r_out**2 - 1 / r_in**2``, so chaining
    ``r_grid -> r_mid -> r_out`` matches the single hop ``r_grid -> r_out``.

    Raises ``ValueError`` if the grid is not log-uniform (``d ln lambda``
    constant to 1e-6 relative) or if ``r_in <= r_out``.
    """
    wavelength = np.asarray(wavelength_um, dtype=np.float64)
    source = np.asarray(values, dtype=np.float64)
    if wavelength.ndim != 1 or source.ndim != 1:
        raise ValueError("wavelength_um and values must be 1-D")
    if wavelength.shape != source.shape:
        raise ValueError(
            "wavelength_um and values must have the same shape, got "
            f"{wavelength.shape} and {source.shape}"
        )
    if wavelength.size < 2:
        raise ValueError("spectrum needs at least 2 samples")
    if np.any(wavelength <= 0.0):
        raise ValueError("wavelength_um must be strictly positive")
    if r_out <= 0.0 or not math.isfinite(r_out):
        raise ValueError("r_out must be positive and finite")
    if r_in <= r_out:
        raise ValueError(
            f"r_in ({r_in}) must exceed r_out ({r_out}); a spectrum cannot be "
            "sharpened by convolution"
        )

    dln = np.diff(np.log(wavelength))
    dln_mean = float(np.mean(dln))
    if dln_mean <= 0.0 or not np.allclose(dln, dln_mean, rtol=1e-6, atol=0.0):
        raise ValueError(
            "convolve_to_resolving_power requires a log-uniform (constant-R) "
            "wavelength grid"
        )
    r_grid = 1.0 / dln_mean
    if r_grid <= r_out:
        raise ValueError(
            f"grid resolving power ({r_grid:.4g}) does not resolve r_out "
            f"({r_out}); resample onto a finer constant-R grid first"
        )

    inv_r_eff_sq = 1.0 / r_out**2 - (0.0 if math.isinf(r_in) else 1.0 / r_in**2)
    r_eff = 1.0 / math.sqrt(inv_r_eff_sq)
    sigma_samples = r_grid / (r_eff * 2.0 * math.sqrt(2.0 * math.log(2.0)))
    return gaussian_filter1d(
        source, sigma_samples, mode="nearest", truncate=_KERNEL_TRUNCATE
    )
