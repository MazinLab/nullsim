"""Photometric zero-point primitives for photon-rate calculations.

SPEC §15.2 calls for pure physics modules guided by KILO/NAYRA behavior while
pinning their assumptions in tests. This module starts that framework with
2MASS near-IR Vega zero points from Cohen, Wheaton & Megeath 2003, AJ 126,
1090; the HWO optical proxy uses a Cousins I zero point from Bessell, Castelli
& Plez 1998, A&A 333, 231.

The public API uses SI flux-density units. Cohen+ list the 2MASS J value as
3.129e-13 W cm^-2 um^-1, which is 3.129e-9 W m^-2 um^-1 after unit conversion.

NaN handling: Non-finite magnitude inputs (NaN, ±inf) propagate through the
computation rather than raising. ``magnitude = NaN`` returns NaN;
``magnitude = +inf`` (infinitely faint) returns 0; ``magnitude = -inf``
(infinitely bright) returns +inf. This lets catalog rows with missing or
unobserved photometry (e.g. astropy/SIMBAD results with NaN bands) flow
through the photon-rate calculation unchanged. **Callers are responsible
for filtering catalog rows with missing photometry before interpreting
results.** Non-magnitude inputs (wavelength, bandwidth, zero-point) still
require finite positive values and raise ``ValueError`` otherwise.
"""

from __future__ import annotations

import math
from collections.abc import Callable

_VJ_COLOR: dict[str, float] = {
    "O": 0.0,
    "B": 0.0,
    "A": 0.1,
    "F": 0.7,
    "G": 1.1,
    "K": 1.8,
    "M": 3.5,
}

_RJ_COLOR: dict[str, float] = {
    "O": 0.1,
    "B": 0.0,
    "A": 0.05,
    "F": 0.4,
    "G": 0.65,
    "K": 1.15,
    "M": 2.0,
}

_YJ_COLOR: dict[str, float] = {
    "O": 0.0,
    "B": 0.0,
    "A": 0.05,
    "F": 0.15,
    "G": 0.25,
    "K": 0.35,
    "M": 0.5,
}

_IJ_COLOR: dict[str, float] = {
    "O": -0.05,
    "B": -0.05,
    "A": 0.0,
    "F": 0.25,
    "G": 0.45,
    "K": 0.75,
    "M": 1.4,
}

_JH_COLOR: dict[str, float] = {
    "O": -0.1,
    "B": -0.05,
    "A": 0.0,
    "F": 0.15,
    "G": 0.3,
    "K": 0.5,
    "M": 0.6,
}

_JK_COLOR: dict[str, float] = {
    "O": -0.2,
    "B": -0.1,
    "A": 0.0,
    "F": 0.2,
    "G": 0.4,
    "K": 0.7,
    "M": 0.85,
}


def j_band_zero_point_flux_density() -> float:
    """Return the 2MASS J-band Vega zero-point flux density.

    The value is from Cohen, Wheaton & Megeath 2003, AJ 126, 1090, Table 2 /
    the 2MASS Explanatory Supplement: 3.129e-13 W cm^-2 um^-1 at
    lambda = 1.235 um. This function returns the SI-converted value in
    W m^-2 um^-1.

    Returns:
        Vega zero-point spectral flux density in W m^-2 um^-1.
    """
    return 3.129e-9


def h_band_zero_point_flux_density() -> float:
    """Return the 2MASS H-band Vega zero-point flux density.

    From Cohen, Wheaton & Megeath 2003, AJ 126, 1090, Table 2 (the
    2MASS Explanatory Supplement value): 1.133e-13 W cm^-2 um^-1 at
    lambda = 1.662 um, returned here in SI W m^-2 um^-1.
    """
    return 1.133e-9


def k_band_zero_point_flux_density() -> float:
    """Return the 2MASS Ks-band Vega zero-point flux density.

    From Cohen, Wheaton & Megeath 2003, AJ 126, 1090, Table 2: 4.283e-14
    W cm^-2 um^-1 at lambda = 2.159 um, returned in SI W m^-2 um^-1.
    """
    return 4.283e-10


def y_band_zero_point_flux_density() -> float:
    """Return the UKIDSS / MKO Y-band Vega zero-point flux density.

    Hewett+ 2006, MNRAS 367, 454, Table 7: 5.913e-13 W cm^-2 um^-1 at
    lambda = 1.020 um (effective wavelength), returned in SI
    W m^-2 um^-1. KILO Keck Y-band runs key on this zeropoint
    (``kilo/photometry.py``).
    """
    return 5.913e-9


def i_band_zero_point_flux_density() -> float:
    """Return the Cousins I-band Vega zero-point flux density.

    Bessell, Castelli & Plez 1998, A&A 333, 231 list a Cousins I effective
    wavelength near 0.806 um and a Vega zero point of about 2416 Jy. Converting
    F_nu to F_lambda gives 1.12e-8 W m^-2 um^-1. This optical band is the
    closest broad-band proxy for HWO's 0.8 um first-pass grid.
    """
    return 1.12e-8


# Private dispatch table: normalized band name → zero-point callable.
# Adding a new band is a one-line change here; no signature change required.
_ZERO_POINT_FN: dict[str, Callable[[], float]] = {
    "I": i_band_zero_point_flux_density,
    "Y": y_band_zero_point_flux_density,
    "J": j_band_zero_point_flux_density,
    "H": h_band_zero_point_flux_density,
    "K": k_band_zero_point_flux_density,
}


def magnitude_to_flux_density(
    magnitude: float,
    zero_point_flux_density: float,
) -> float:
    """Convert a Vega-system magnitude into spectral flux density.

    Non-finite magnitude inputs (NaN, ±inf) propagate as NaN / 0 / +inf
    through the Pogson formula rather than raising; callers with catalog
    data containing missing photometry should filter rows before
    interpreting the result. See the module docstring for the policy.

    Args:
        magnitude: Vega-system apparent magnitude. Negative finite magnitudes
            are valid and represent sources brighter than Vega. NaN, +inf,
            and -inf are accepted and propagate naturally
            (NaN -> NaN, +inf -> 0, -inf -> +inf).
        zero_point_flux_density: Zero-magnitude flux density in W m^-2 um^-1.
            Must be finite and positive.

    Returns:
        Spectral flux density in W m^-2 um^-1.

    Raises:
        ValueError: If the zero point is not finite and positive.
    """
    magnitude_value = float(magnitude)
    zero_point_value = _require_positive_finite_float(
        zero_point_flux_density,
        "zero_point_flux_density",
    )
    return zero_point_value * 10.0 ** (-0.4 * magnitude_value)


def photon_rate_per_m2_per_bin(
    magnitude: float,
    wavelength_um: float,
    bandwidth_um: float,
    band: str = "J",
) -> float:
    """Convert a Vega magnitude into photons s^-1 m^-2 in one wavelength bin.

    This is a rectangular-bin approximation: the zero-point spectral flux
    density is treated as constant across ``bandwidth_um`` and divided by the
    photon energy at ``wavelength_um``.

    Args:
        magnitude: Vega-system apparent magnitude. Negative finite magnitudes
            are valid and represent sources brighter than Vega.
        wavelength_um: Photon wavelength in micrometers.
        bandwidth_um: Spectral-bin width in micrometers.
        band: Photometric band for the Vega zero point.

    Returns:
        Photon rate in photons s^-1 m^-2 for one rectangular spectral bin.
        Non-finite magnitude inputs propagate (NaN -> NaN, +inf -> 0,
        -inf -> +inf); see module docstring.

    Raises:
        ValueError: If band is unsupported, or wavelength/bandwidth are not
            finite and positive.
    """
    normalized_band = _normalize_band(band)
    if normalized_band not in _ZERO_POINT_FN:
        supported = ", ".join(sorted(_ZERO_POINT_FN))
        raise ValueError(
            f"Unsupported photometric band '{band}'. Supported bands: {supported}."
        )

    wavelength_value_um = _require_positive_finite_float(
        wavelength_um,
        "wavelength_um",
    )
    bandwidth_value_um = _require_positive_finite_float(
        bandwidth_um,
        "bandwidth_um",
    )
    flux_density_w_m2_um = magnitude_to_flux_density(
        magnitude,
        _ZERO_POINT_FN[normalized_band](),
    )
    photon_energy_j = _photon_energy_joule(wavelength_value_um)
    return flux_density_w_m2_um * bandwidth_value_um / photon_energy_j


def jmag_to_band_mag(
    jmag: float,
    band: str,
    spectral_type: str | None = None,
    *,
    default_spectral_type: str = "F",
) -> float:
    """Estimate a broad-band magnitude from J magnitude and stellar type.

    The color table follows the compact KILO/NAYRA convention derived from
    Pecaut & Mamajek dwarf colors. Only the spectral class letter is used
    (``"G2V"`` -> ``"G"``). Unknown or missing spectral types fall back to
    ``default_spectral_type``.
    """

    magnitude = float(jmag)
    normalized_band = _normalize_band(band)
    key = _spectral_key(spectral_type, default_spectral_type)
    if normalized_band == "J":
        return magnitude
    if normalized_band == "V":
        return magnitude + _VJ_COLOR[key]
    if normalized_band == "R":
        return magnitude + _RJ_COLOR[key]
    if normalized_band == "I":
        return magnitude + _IJ_COLOR[key]
    if normalized_band == "Y":
        return magnitude + _YJ_COLOR[key]
    if normalized_band == "H":
        return magnitude - _JH_COLOR[key]
    if normalized_band == "K":
        return magnitude - _JK_COLOR[key]
    raise ValueError("band must be one of V, R, I, Y, J, H, or K.")


def stellar_band_magnitude_from_jmag(
    jmag: float,
    band: str,
    spectral_type: str | None = None,
    *,
    derive_from_jmag: bool = False,
    default_spectral_type: str = "F",
) -> float:
    """Return the magnitude to use with a band-specific Vega zero point.

    Existing KILO/NAYRA configs pass catalog J magnitudes directly to the
    detector's selected zero-point band. That legacy behavior remains the
    default. New optical configs can opt in to a compact color conversion from
    scene J magnitude and spectral class.
    """

    if derive_from_jmag:
        return jmag_to_band_mag(
            jmag,
            band,
            spectral_type,
            default_spectral_type=default_spectral_type,
        )
    return float(jmag)


def estimate_r_mag_from_j(
    jmag: float,
    spectral_type: str | None = None,
    *,
    default_spectral_type: str = "F",
) -> float:
    """Estimate Cousins R magnitude from J magnitude and stellar type."""

    return jmag_to_band_mag(
        jmag,
        "R",
        spectral_type,
        default_spectral_type=default_spectral_type,
    )


def _normalize_band(band: str) -> str:
    try:
        normalized_band = band.upper()
    except AttributeError as exc:
        raise ValueError("band must be a string photometric band name.") from exc
    return normalized_band


def _spectral_key(spectral_type: str | None, default_spectral_type: str) -> str:
    candidate = spectral_type or default_spectral_type
    if not candidate:
        candidate = "F"
    key = str(candidate).strip()[:1].upper()
    if key in _VJ_COLOR:
        return key
    fallback = str(default_spectral_type or "F").strip()[:1].upper()
    if fallback in _VJ_COLOR:
        return fallback
    return "F"


def _require_finite_float(value: float, name: str) -> float:
    value_float = float(value)
    if not math.isfinite(value_float):
        raise ValueError(f"{name} must be finite.")
    return value_float


def _require_positive_finite_float(value: float, name: str) -> float:
    value_float = _require_finite_float(value, name)
    if value_float <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return value_float


def _photon_energy_joule(wavelength_um: float) -> float:
    wavelength_m = wavelength_um * 1.0e-6
    return _planck_constant_j_s() * _speed_of_light_m_per_s() / wavelength_m


def _planck_constant_j_s() -> float:
    return 6.62607015e-34


def _speed_of_light_m_per_s() -> float:
    return 299_792_458.0


__all__ = [
    "h_band_zero_point_flux_density",
    "i_band_zero_point_flux_density",
    "j_band_zero_point_flux_density",
    "k_band_zero_point_flux_density",
    "y_band_zero_point_flux_density",
    "jmag_to_band_mag",
    "magnitude_to_flux_density",
    "photon_rate_per_m2_per_bin",
    "stellar_band_magnitude_from_jmag",
]
