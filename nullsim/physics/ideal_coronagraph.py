"""Theoretically optimal coronagraph performance bound (Guyon et al. 2006).

Guyon, Pluzhnik, Kuchner, Collins & Ridgway (2006, ApJS 167, 81) define the
theoretically optimal coronagraph as the instrument that removes exactly the
on-axis stellar mode of the pupil and nothing else. For a clear circular
aperture of diameter D, the field overlap between two point sources separated
by an angle theta is the Airy amplitude 2 J1(x)/x with x = pi D theta / lambda,
so the bound has closed form:

- An off-axis planet loses only the part of its field that lies in the removed
  stellar mode: useful throughput T(theta) = 1 - |2 J1(x)/x|^2. This reaches
  50% near 0.52 lambda/D, the "IWA ~ 0.5 lambda/D" headline of the paper.
- A point star is removed perfectly; a finite stellar disk leaks the
  area-weighted average of the same function over the disk, ~ (pi D rho /
  lambda)^2 / 8 for a uniform disk of angular radius rho << lambda/D
  (a second-order null).

Guyon 2006 (sec. 4.2) also defines higher-order versions that additionally
remove the low-order derivatives of the stellar field, trading inner working
angle for resistance to finite stellar size. Order 4 removes the flat mode
plus the two linear (tip/tilt) pupil modes; the only nonzero extra overlap for
a source offset along x is with the x-tilt mode, |4 J2(x)/x|^2, giving

    T4(theta) = 1 - |2 J1(x)/x|^2 - |4 J2(x)/x|^2  ~ x^4 / 64 near axis,

with uniform-disk leakage ~ x_edge^4 / 192. The 50% throughput point moves
out to ~1 lambda/D.

The detection bound is then photon noise on the leaked starlight plus
backgrounds, against the planet rate through T(theta). These primitives stay
independent of TOML and SimulationState per the physics-layer contract.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import j1, jv

__all__ = [
    "planet_mode_throughput",
    "uniform_disk_leakage",
    "ideal_coronagraph_contrast_curve",
]

MAS_TO_RAD = np.pi / (180.0 * 3600.0 * 1000.0)


def _removed_mode_overlap_power(
    x: NDArray[np.float64], order: int
) -> NDArray[np.float64]:
    """Power overlap of an off-axis source with the removed stellar subspace.

    Order 2 removes the flat pupil mode: |2 J1(x)/x|^2 (x -> 0 limit 1).
    Order 4 additionally removes the tip/tilt modes; for a source offset
    along x the only extra nonzero overlap is the x-tilt term |4 J2(x)/x|^2.
    """

    if order not in (2, 4):
        raise ValueError(f"ideal coronagraph order must be 2 or 4 (got {order}).")
    x = np.asarray(x, dtype=np.float64)
    out = np.ones_like(x)
    nonzero = x != 0.0
    xs = x[nonzero]
    overlap = (2.0 * j1(xs) / xs) ** 2
    if order == 4:
        overlap = overlap + (4.0 * jv(2, xs) / xs) ** 2
    out[nonzero] = overlap
    return out


def planet_mode_throughput(
    separation_rad: NDArray[np.float64] | float,
    wavelength_m: float,
    diameter_m: float,
    *,
    order: int = 2,
) -> NDArray[np.float64]:
    """Planet throughput of the optimal coronagraph of the given order.

    Order 2: 1 - |2 J1(x)/x|^2. Order 4 additionally loses the tip/tilt-mode
    projection (see module docstring).

    Parameters
    ----------
    separation_rad:
        Source offset(s) from the optical axis in radians.
    wavelength_m:
        Observing wavelength in meters.
    diameter_m:
        Clear circular aperture diameter in meters.
    order:
        Null order of the removed stellar subspace (2 or 4).
    """

    if wavelength_m <= 0.0 or diameter_m <= 0.0:
        raise ValueError(
            "planet_mode_throughput requires positive wavelength_m and "
            f"diameter_m (got {wavelength_m}, {diameter_m})."
        )
    separation = np.atleast_1d(np.asarray(separation_rad, dtype=np.float64))
    if np.any(separation < 0.0):
        raise ValueError("planet_mode_throughput separations must be >= 0.")
    x = np.pi * diameter_m * separation / wavelength_m
    return 1.0 - _removed_mode_overlap_power(x, order)


def uniform_disk_leakage(
    stellar_radius_rad: float,
    wavelength_m: float,
    diameter_m: float,
    *,
    order: int = 2,
    n_radial: int = 128,
) -> float:
    """Stellar leakage of the optimal coronagraph for a uniform disk.

    Area-weighted average of ``planet_mode_throughput`` over a uniform disk of
    angular radius ``stellar_radius_rad``:

        L = (2 / rho^2) * Integral_0^rho r * T(r) dr

    evaluated with trapezoidal quadrature on ``n_radial`` samples. Small-disk
    limits: L -> (pi D rho / lambda)^2 / 8 at order 2 and
    (pi D rho / lambda)^4 / 192 at order 4.
    """

    if stellar_radius_rad < 0.0:
        raise ValueError("uniform_disk_leakage requires stellar_radius_rad >= 0.")
    if stellar_radius_rad == 0.0:
        return 0.0
    radii = np.linspace(0.0, stellar_radius_rad, int(n_radial))
    throughput = planet_mode_throughput(radii, wavelength_m, diameter_m, order=order)
    integral = np.trapezoid(radii * throughput, radii)
    return float(2.0 * integral / stellar_radius_rad**2)


def ideal_coronagraph_contrast_curve(
    separations_mas: NDArray[np.float64],
    *,
    wavelengths_m: NDArray[np.float64],
    diameter_m: float,
    stellar_radius_mas: float,
    stellar_rate_hz: float,
    background_rate_hz: float,
    integration_time_s: float,
    snr_target: float = 5.0,
    order: int = 2,
) -> NDArray[np.float64]:
    """Photon-limited detectable contrast for the optimal coronagraph.

    For each separation the detectable planet/star contrast at ``snr_target``
    is

        c(theta) = snr * sqrt((L * R_star + R_bg) * t) / (T(theta) * R_star * t)

    where T and the uniform-disk stellar leakage L are band averaged with
    equal weight over ``wavelengths_m``. ``stellar_rate_hz`` is the stellar
    photon rate through the comparison instrument's unavoidable losses
    (collecting area, mirror train, detector QE) with NO coronagraph mask;
    ``background_rate_hz`` is the total background (e.g. local + exozodi)
    photon rate in the search aperture. Separations where the band-averaged
    throughput is zero return inf.

    Parameters
    ----------
    separations_mas:
        Sky separations in milliarcseconds.
    wavelengths_m:
        Wavelength grid in meters; the bound is averaged across it.
    diameter_m:
        Aperture diameter in meters.
    stellar_radius_mas:
        Uniform-disk stellar angular radius in milliarcseconds.
    stellar_rate_hz:
        Unocculted stellar photon detection rate in counts/s.
    background_rate_hz:
        Background photon detection rate in counts/s.
    integration_time_s:
        Integration time in seconds.
    snr_target:
        Required detection significance.
    order:
        Null order of the removed stellar subspace (2 or 4).
    """

    separations = np.asarray(separations_mas, dtype=np.float64)
    wavelengths = np.atleast_1d(np.asarray(wavelengths_m, dtype=np.float64))
    if wavelengths.size == 0 or np.any(wavelengths <= 0.0):
        raise ValueError("wavelengths_m must be non-empty and positive.")
    if stellar_rate_hz <= 0.0 or integration_time_s <= 0.0:
        raise ValueError(
            "ideal_coronagraph_contrast_curve requires positive "
            "stellar_rate_hz and integration_time_s."
        )
    if background_rate_hz < 0.0:
        raise ValueError("background_rate_hz must be >= 0.")
    if snr_target <= 0.0:
        raise ValueError("snr_target must be > 0.")

    separations_rad = separations * MAS_TO_RAD
    stellar_radius_rad = float(stellar_radius_mas) * MAS_TO_RAD

    throughput = np.zeros_like(separations_rad)
    leakage = 0.0
    for wavelength_m in wavelengths:
        throughput += planet_mode_throughput(
            separations_rad, float(wavelength_m), diameter_m, order=order
        )
        leakage += uniform_disk_leakage(
            stellar_radius_rad, float(wavelength_m), diameter_m, order=order
        )
    throughput /= float(wavelengths.size)
    leakage /= float(wavelengths.size)

    noise_counts = (leakage * stellar_rate_hz + background_rate_hz) * float(
        integration_time_s
    )
    signal_per_contrast = throughput * stellar_rate_hz * float(integration_time_s)
    with np.errstate(divide="ignore"):
        curve = float(snr_target) * np.sqrt(noise_counts) / signal_per_contrast
    curve[signal_per_contrast <= 0.0] = np.inf
    return curve
