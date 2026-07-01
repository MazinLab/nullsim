"""Planet throughput vs angular separation through a kernel-nulled chip.

For a planet at angular offset ``(alpha, beta) = sep * (sin PA, cos PA)``
from the on-axis star, each sub-aperture sees a baseline-dependent
fringe phase ``phi_k = 2*pi/lambda * (x_k * alpha + y_k * beta)`` plus
an angular-coupling rolloff into the SMF that scales the amplitude by
``sqrt(eta_ang(theta, D, lambda))``. The complex per-mode field is

    E_k = sqrt(eta_k * A_k * eta_ang_k(theta)) * exp(i phi_k)

and the chip unitary ``U`` routes it to dark ports. The throughput is

    T(sep, PA) = sum_{k in dark} |U @ E|_k^2 / sum_k |E_star_k|^2

position-angle-averaged. The denominator uses the on-axis stellar
``|E_star_k|^2 = eta_k * A_k`` so the returned throughput captures both
the angular coupling rolloff and the MZI routing efficiency.

Band-averaging — pass ``unitaries_by_wavelength=[(U_wl, lambda_wl), ...]``
to mean across multiple wavelengths. Long baselines wrap ~25 fringe
cycles at J band over the keck pair, so the band-averaged throughput
can differ from the monochromatic value by O(50%) at the most chromatic
separations (KILO performance.py docstring lines 1035-1046).

Port of KILO ``kilo/performance.py`` ``compute_planet_throughput_vs_separation``
(lines 1010-1128) and ``kilo/injection.py`` ``planet_input_field``
(lines 459-525). The angular-coupling rolloff uses
``nullsim.physics.ruilier_cassaing.angular_coupling_factor`` so the
on-axis SMF convention stays consistent across the package.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from nullsim.physics.imaging import compute_beam_patterns
from nullsim.physics.ruilier_cassaing import angular_coupling_factor

__all__ = [
    "stellar_input_field",
    "planet_input_field",
    "compute_planet_throughput_vs_separation",
    "dark_port_transmission_map",
]


def stellar_input_field(
    coupling_per_sub: NDArray[np.float64],
    sub_areas_m2: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Return the on-axis complex stellar field per sub-aperture.

    Args:
        coupling_per_sub: SMF coupling efficiency per sub-aperture, shape
            ``(M,)`` or ``(n_wavelength, M)``.
        sub_areas_m2: Collecting area per sub-aperture in m^2, shape ``(M,)``.

    Returns:
        ``E_k = sqrt(eta_k * A_k)`` cast to complex. Shape matches
        ``coupling_per_sub``.
    """

    coupling = np.asarray(coupling_per_sub, dtype=np.float64)
    areas = np.asarray(sub_areas_m2, dtype=np.float64)
    return np.sqrt(coupling * areas).astype(np.complex128)


def planet_input_field(
    positions_m: NDArray[np.float64],
    coupling_per_sub: NDArray[np.float64],
    sub_areas_m2: NDArray[np.float64],
    sub_diameters_m: NDArray[np.float64] | None,
    wavelength_m: float,
    separation_rad: float,
    position_angle_rad: float,
) -> NDArray[np.complex128]:
    """Off-axis planet field per sub-aperture for a single wavelength.

    Args:
        positions_m: Sub-aperture ENU positions, shape ``(M, 2)`` with
            columns ``[East, North]`` in meters.
        coupling_per_sub: On-axis SMF coupling per sub-aperture, ``(M,)``.
        sub_areas_m2: Collecting area per sub-aperture in m^2, ``(M,)``.
        sub_diameters_m: Sub-aperture diameter per mode in m, ``(M,)`` or
            ``None``. When supplied, the angular-coupling rolloff
            ``exp(-(pi w theta / lambda)^2)`` is applied per mode.
        wavelength_m: Wavelength in meters.
        separation_rad: Planet angular separation in radians.
        position_angle_rad: Planet position angle (rad, N-through-E sky
            convention matching KILO ``planet_input_field``).

    Returns:
        Complex field vector ``E_k`` of shape ``(M,)``.
    """

    pos = np.asarray(positions_m, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError(
            "planet_input_field: positions_m must have shape (M, 2); "
            f"got {pos.shape}."
        )
    coupling = np.asarray(coupling_per_sub, dtype=np.float64)
    areas = np.asarray(sub_areas_m2, dtype=np.float64)

    alpha = separation_rad * np.sin(position_angle_rad)
    beta = separation_rad * np.cos(position_angle_rad)
    opd_m = pos[:, 0] * alpha + pos[:, 1] * beta
    phase = (2.0 * np.pi / float(wavelength_m)) * opd_m

    amplitude = np.sqrt(coupling * areas)
    if sub_diameters_m is not None:
        ang_power = angular_coupling_factor(
            separation_rad,
            np.asarray(sub_diameters_m, dtype=np.float64),
            float(wavelength_m),
        )
        amplitude = amplitude * np.sqrt(np.asarray(ang_power, dtype=np.float64))
    return (amplitude * np.exp(1j * phase)).astype(np.complex128)


def compute_planet_throughput_vs_separation(
    *,
    u_stack: NDArray[np.complex128],
    wavelengths_m: Sequence[float],
    positions_m: NDArray[np.float64],
    dark_indices: NDArray[np.intp] | Sequence[int],
    coupling_per_sub: NDArray[np.float64],
    sub_areas_m2: NDArray[np.float64],
    sub_diameters_m: NDArray[np.float64] | None,
    separations_rad: NDArray[np.float64] | Sequence[float],
    n_pa_samples: int = 36,
    input_transport_scales: NDArray[np.complex128] | None = None,
) -> NDArray[np.float64]:
    """Band- and PA-averaged planet throughput vs separation.

    The on-axis stellar power ``P_star = sum_k coupling_k * A_k`` is the
    normalization, so the returned ``T(sep)`` represents the fraction of
    coupled stellar power reaching the dark ports for a planet at the
    given separation (averaged over position angle, then over
    wavelength bins).

    Args:
        u_stack: Chip transfer stack of shape ``(n_ports, n_modes, n_wl)``.
        wavelengths_m: One wavelength per bin (length ``n_wl``).
        positions_m: ENU sub-aperture positions in m, shape ``(M, 2)``.
        dark_indices: Dark output-port indices on the chip.
        coupling_per_sub: SMF coupling per mode, shape ``(M,)``.
        sub_areas_m2: Collecting area per mode in m^2, shape ``(M,)``.
        sub_diameters_m: Sub-aperture diameter per mode in m, ``(M,)`` or
            ``None`` (skip the angular-coupling rolloff).
        separations_rad: Separations to evaluate, shape ``(n_sep,)``.
        n_pa_samples: Number of uniformly-spaced position angles in
            ``[0, 2*pi)``.
        input_transport_scales: Optional complex amplitude scale at the chip
            input, shape ``(n_transport_states, n_wl, M)``. This folds
            pointing-dependent delay-line loss and chromatic phase into the
            planet leg before the chip unitary. Each transport state is
            normalized by its own transported on-axis stellar power so common
            delay-line attenuation remains in the detector photon rate, while
            differential loss and phase still affect dark-port routing.

    Returns:
        Throughput array of shape ``(n_sep,)``.
    """

    u_array = np.asarray(u_stack)
    if u_array.ndim != 3:
        raise ValueError(
            "compute_planet_throughput_vs_separation: u_stack must have "
            f"shape (n_ports, n_modes, n_wl); got {u_array.shape}."
        )
    n_ports, n_modes, n_wl = u_array.shape
    wavelengths_arr = np.asarray(wavelengths_m, dtype=np.float64)
    if wavelengths_arr.shape != (n_wl,):
        raise ValueError(
            "compute_planet_throughput_vs_separation: wavelengths_m must "
            f"have length {n_wl} matching u_stack; got {wavelengths_arr.shape}."
        )

    coupling = np.asarray(coupling_per_sub, dtype=np.float64)
    areas = np.asarray(sub_areas_m2, dtype=np.float64)
    if coupling.shape != areas.shape:
        raise ValueError(
            "compute_planet_throughput_vs_separation: coupling and areas "
            f"shape mismatch (coupling {coupling.shape}, areas {areas.shape})."
        )
    base_amplitudes = np.sqrt(coupling * areas)
    base_power = base_amplitudes * base_amplitudes
    star_total_power = float(np.sum(base_power))
    if star_total_power <= 0.0:
        raise ValueError(
            "compute_planet_throughput_vs_separation: star_total_power "
            "is zero; check coupling_per_sub and sub_areas_m2."
        )

    dark_idx = np.asarray(dark_indices, dtype=np.intp)
    if np.any((dark_idx < 0) | (dark_idx >= n_ports)):
        raise ValueError(
            "compute_planet_throughput_vs_separation: dark_indices out of range."
        )
    seps = np.asarray(separations_rad, dtype=np.float64)
    pa_angles = np.linspace(0.0, 2.0 * np.pi, n_pa_samples, endpoint=False)
    transport = _input_transport_scales(
        input_transport_scales,
        n_wl=n_wl,
        n_modes=n_modes,
    )

    throughputs = np.zeros(seps.size, dtype=np.float64)
    sub_diameters = (
        np.asarray(sub_diameters_m, dtype=np.float64)
        if sub_diameters_m is not None
        else None
    )

    # Vectorized over position angle and transport state. The previous
    # quadruple Python loop ran ~300k single matvecs per kilo_keck_6 sweep
    # cell (20 wl x 70 sep x 6 states x 36 PA); batching the PA/state axes
    # into one einsum per (wl, sep) is identical math up to summation
    # order (review 2026-06-10 Issue 31a).
    sin_pa = np.sin(pa_angles)
    cos_pa = np.cos(pa_angles)
    for w_idx in range(n_wl):
        u_dark = u_array[dark_idx, :, w_idx]  # (n_dark, M)
        wavelength = float(wavelengths_arr[w_idx])
        transport_wl = transport[:, w_idx, :]  # (n_states, M)
        transported_star_power = np.sum(
            base_power[np.newaxis, :] * np.abs(transport_wl) ** 2,
            axis=1,
        )
        if np.any(transported_star_power <= 0.0):
            raise ValueError(
                "compute_planet_throughput_vs_separation: transported "
                "star_total_power is zero; check input_transport_scales."
            )
        for i, sep in enumerate(seps):
            if sub_diameters is not None:
                ang_power = np.asarray(
                    angular_coupling_factor(sep, sub_diameters, wavelength),
                    dtype=np.float64,
                )
                amplitudes = base_amplitudes * np.sqrt(ang_power)
            else:
                amplitudes = base_amplitudes

            # (n_pa, M) planet phase factors for this (wl, sep).
            opd = sep * (
                np.outer(sin_pa, positions_m[:, 0])
                + np.outer(cos_pa, positions_m[:, 1])
            )
            planet_phase = np.exp(1j * (2.0 * np.pi / wavelength) * opd)
            planet_fields = (
                amplitudes[np.newaxis, np.newaxis, :]
                * planet_phase[np.newaxis, :, :]
                * transport_wl[:, np.newaxis, :]
            )  # (n_states, n_pa, M)
            outputs = np.einsum("dm,spm->spd", u_dark, planet_fields)
            dark_power = np.sum(np.abs(outputs) ** 2, axis=2)  # (n_states, n_pa)
            per_transport = np.mean(dark_power, axis=1) / transported_star_power
            throughputs[i] += float(np.mean(per_transport))

    throughputs /= float(n_wl)
    return throughputs


def dark_port_transmission_map(
    *,
    u_stack: NDArray[np.complex128],
    wavelengths_m: Sequence[float],
    positions_m: NDArray[np.float64],
    dark_indices: NDArray[np.intp] | Sequence[int],
    coupling_per_sub: NDArray[np.float64],
    sub_areas_m2: NDArray[np.float64],
    sub_diameters_m: NDArray[np.float64] | None,
    l_coords_rad: NDArray[np.float64],
    m_coords_rad: NDArray[np.float64],
    wavelength_indices: Sequence[int],
) -> NDArray[np.float64]:
    """Return a 2-D summed dark-port transmission map.

    The returned quantity is the static optical dark-port fraction over sky
    coordinates ``(l, m)``. It uses the same on-axis stellar-power
    normalization as ``compute_planet_throughput_vs_separation`` but does not
    include pointing-jitter convolution or delay-line transport scales.

    Args:
        u_stack: Output matrix stack, shape ``(n_ports, n_modes, n_wl)``.
            The first axis may be a full square chip unitary or an effective
            dark-row matrix, for example the arm-weighted rows used for
            ``deployment_mode='per_pol_arm'`` diagnostics.
        wavelengths_m: One wavelength per bin, length ``n_wl``.
        positions_m: Pupil-plane mode positions, shape ``(n_modes, 2)`` with
            columns ``[East, North]`` in meters.
        dark_indices: Output-row indices to sum.
        coupling_per_sub: On-axis SMF coupling per mode, shape ``(n_modes,)``.
        sub_areas_m2: Collecting area per mode in m^2, shape ``(n_modes,)``.
        sub_diameters_m: Mode diameter per mode in m, shape ``(n_modes,)``, or
            ``None`` to skip the angular-coupling envelope.
        l_coords_rad: East-coordinate grid in radians, shape ``(n_l,)``.
        m_coords_rad: North-coordinate grid in radians, shape ``(n_m,)``.
        wavelength_indices: Wavelength bins to average.

    Returns:
        Transmission map with shape ``(n_m, n_l)``.
    """

    u_array = np.asarray(u_stack, dtype=np.complex128)
    if u_array.ndim != 3:
        raise ValueError(
            "dark_port_transmission_map: u_stack must have shape "
            f"(n_ports, n_modes, n_wavelength); got {u_array.shape}."
        )
    n_ports, n_modes, n_wl = u_array.shape
    wavelengths_arr = np.asarray(wavelengths_m, dtype=np.float64)
    if wavelengths_arr.shape != (n_wl,):
        raise ValueError(
            "dark_port_transmission_map: wavelengths_m must have length "
            f"{n_wl} matching u_stack; got {wavelengths_arr.shape}."
        )
    if not np.all(np.isfinite(wavelengths_arr)) or np.any(wavelengths_arr <= 0.0):
        raise ValueError("dark_port_transmission_map: wavelengths_m must be positive.")

    positions = np.asarray(positions_m, dtype=np.float64)
    if positions.shape != (n_modes, 2):
        raise ValueError(
            "dark_port_transmission_map: positions_m must have shape "
            f"({n_modes}, 2); got {positions.shape}."
        )
    if not np.all(np.isfinite(positions)):
        raise ValueError("dark_port_transmission_map: positions_m must be finite.")

    dark_idx = np.asarray(dark_indices, dtype=np.intp)
    if dark_idx.ndim != 1 or dark_idx.size == 0:
        raise ValueError("dark_port_transmission_map: dark_indices must be non-empty.")
    if np.any((dark_idx < 0) | (dark_idx >= n_ports)):
        raise ValueError("dark_port_transmission_map: dark_indices out of range.")

    coupling = np.asarray(coupling_per_sub, dtype=np.float64)
    areas = np.asarray(sub_areas_m2, dtype=np.float64)
    if coupling.shape != (n_modes,) or areas.shape != (n_modes,):
        raise ValueError(
            "dark_port_transmission_map: coupling_per_sub and sub_areas_m2 "
            f"must both have shape ({n_modes},)."
        )
    if (
        not np.all(np.isfinite(coupling))
        or not np.all(np.isfinite(areas))
        or np.any(coupling < 0.0)
        or np.any(areas < 0.0)
    ):
        raise ValueError(
            "dark_port_transmission_map: coupling_per_sub and sub_areas_m2 "
            "must be finite and non-negative."
        )
    base_power = coupling * areas
    star_total_power = float(np.sum(base_power))
    if star_total_power <= 0.0:
        raise ValueError(
            "dark_port_transmission_map: star_total_power is zero; check "
            "coupling_per_sub and sub_areas_m2."
        )

    sub_diameters = None
    if sub_diameters_m is not None:
        sub_diameters = np.asarray(sub_diameters_m, dtype=np.float64)
        if sub_diameters.shape != (n_modes,):
            raise ValueError(
                "dark_port_transmission_map: sub_diameters_m must have shape "
                f"({n_modes},)."
            )
        if not np.all(np.isfinite(sub_diameters)) or np.any(sub_diameters <= 0.0):
            raise ValueError(
                "dark_port_transmission_map: sub_diameters_m must be positive."
            )

    l_coords = _finite_1d_coords(l_coords_rad, "l_coords_rad")
    m_coords = _finite_1d_coords(m_coords_rad, "m_coords_rad")
    indices = np.asarray(tuple(wavelength_indices), dtype=np.intp)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError(
            "dark_port_transmission_map: wavelength_indices must be non-empty."
        )
    if np.any((indices < 0) | (indices >= n_wl)):
        raise ValueError(
            "dark_port_transmission_map: wavelength_indices out of range."
        )

    amplitudes = np.sqrt(base_power).astype(np.complex128)
    transmission = np.zeros((m_coords.size, l_coords.size), dtype=np.float64)
    for wavelength_index in indices:
        wavelength = float(wavelengths_arr[int(wavelength_index)])
        beams = compute_beam_patterns(
            u_sub=positions[:, 0] / wavelength,
            v_sub=positions[:, 1] / wavelength,
            amplitudes=amplitudes,
            unitary=u_array[:, :, int(wavelength_index)],
            port_indices=dark_idx,
            l_coords=l_coords,
            m_coords=m_coords,
            sub_diameters_m=sub_diameters,
            wavelength_m=wavelength if sub_diameters is not None else None,
        )
        transmission += np.sum(beams, axis=0) / star_total_power

    transmission /= float(indices.size)
    return np.maximum(transmission, 0.0)


def _input_transport_scales(
    input_transport_scales: NDArray[np.complex128] | None,
    *,
    n_wl: int,
    n_modes: int,
) -> NDArray[np.complex128]:
    """Return transport scales as ``(n_transport_states, n_wl, n_modes)``."""

    if input_transport_scales is None:
        return np.ones((1, n_wl, n_modes), dtype=np.complex128)

    transport = np.asarray(input_transport_scales, dtype=np.complex128)
    expected_tail = (n_wl, n_modes)
    if transport.ndim != 3 or transport.shape[1:] != expected_tail:
        raise ValueError(
            "compute_planet_throughput_vs_separation: input_transport_scales "
            "must have shape (n_transport_states, n_wl, n_modes); got "
            f"{transport.shape}, expected tail {expected_tail}."
        )
    if transport.shape[0] <= 0:
        raise ValueError(
            "compute_planet_throughput_vs_separation: input_transport_scales "
            "must contain at least one transport state."
        )
    return transport


def _finite_1d_coords(
    coords: NDArray[np.float64] | Sequence[float],
    name: str,
) -> NDArray[np.float64]:
    """Validate a non-empty finite coordinate vector."""

    values = np.asarray(coords, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"dark_port_transmission_map: {name} must be 1-D.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"dark_port_transmission_map: {name} must be finite.")
    return values
