"""Aperture-synthesis image reconstruction for photonic nuller outputs.

The functions here model dark-port beam patterns, simulate photon-count
measurements for point-source scenes, and reconstruct matched-filter SNR maps
from those measurements. They are intentionally independent of nullsim config,
pipeline state, and output writers so the same math can be tested directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import lsq_linear, nnls

from nullsim.physics.ruilier_cassaing import angular_coupling_factor

MAS_PER_RAD = 206_264_806.247
MAUNAKEA_LAT_RAD = np.radians(19.8283)


@dataclass(frozen=True)
class SkyScene:
    """Synthetic sky brightness map in planet/star contrast units."""

    image: NDArray[np.float64]
    fov_rad: float

    @property
    def n_pix(self) -> int:
        return int(self.image.shape[0])

    @property
    def fov_mas(self) -> float:
        return float(self.fov_rad * MAS_PER_RAD)

    @property
    def l_coords(self) -> NDArray[np.float64]:
        return np.linspace(-self.fov_rad, self.fov_rad, self.n_pix)

    @property
    def m_coords(self) -> NDArray[np.float64]:
        return self.l_coords


@dataclass(frozen=True)
class SubAperture:
    """One chip-input mode with an ENU position and equivalent diameter."""

    position_enu_m: NDArray[np.float64]
    diameter_m: float


@dataclass(frozen=True)
class ObservationData:
    """Simulated dark-port count measurements over an observation."""

    measurements: NDArray[np.float64]
    hour_angles_rad: NDArray[np.float64]
    dark_port_indices: NDArray[np.intp]
    stellar_leakage_per_port: NDArray[np.float64]
    dark_counts_per_step: float = 0.0
    sky_counts_per_step: float = 0.0
    response_matrix: NDArray[np.float64] | None = None
    response_scale_per_row: NDArray[np.float64] | None = None
    baseline_counts: NDArray[np.float64] | None = None
    variance_counts: NDArray[np.float64] | None = None
    wavelengths_m: NDArray[np.float64] | None = None
    # Band weights (stellar photon fraction per wavelength, summing to one)
    # used to build the band-averaged response rows. None means uniform.
    wavelength_weights: NDArray[np.float64] | None = None


@dataclass(frozen=True)
class ReconstructedImage:
    """Image reconstruction product."""

    image: NDArray[np.float64]
    residuals: NDArray[np.float64]
    clean_components: tuple[tuple[int, int, float], ...]
    synthesized_beam: NDArray[np.float64]
    method: str
    n_iterations: int | None
    converged: bool = True
    solver_status: int | None = None
    solver_message: str = ""
    residual_norm: float | None = None


def create_scene(
    *,
    n_pix: int,
    fov_mas: float,
    planets: Sequence[tuple[float, float, float]] = (),
) -> SkyScene:
    """Create a point-source planet scene.

    Args:
        n_pix: Number of pixels per image side.
        fov_mas: Half-width of the image in milliarcseconds.
        planets: ``(separation_mas, position_angle_deg, contrast)`` entries.

    Returns:
        A contrast image with each planet placed at the nearest pixel center.
    """

    if n_pix <= 1:
        raise ValueError("n_pix must be greater than 1.")
    if fov_mas <= 0.0:
        raise ValueError("fov_mas must be positive.")

    fov_rad = float(fov_mas) / MAS_PER_RAD
    coords = np.linspace(-fov_rad, fov_rad, int(n_pix))
    image = np.zeros((int(n_pix), int(n_pix)), dtype=np.float64)

    for sep_mas, pa_deg, contrast in planets:
        if contrast < 0.0:
            raise ValueError("planet contrast values must be non-negative.")
        sep_rad = float(sep_mas) / MAS_PER_RAD
        pa_rad = np.radians(float(pa_deg))
        l_offset = sep_rad * np.sin(pa_rad)
        m_offset = sep_rad * np.cos(pa_rad)
        col = int(np.argmin(np.abs(coords - l_offset)))
        row = int(np.argmin(np.abs(coords - m_offset)))
        image[row, col] += float(contrast)

    return SkyScene(image=image, fov_rad=fov_rad)


def subapertures_to_uv(
    subapertures: Sequence[SubAperture],
    hour_angle_rad: float | NDArray[np.floating],
    declination_rad: float,
    wavelength_m: float,
    latitude_rad: float = MAUNAKEA_LAT_RAD,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Project sub-aperture ENU positions to sky-plane ``(u, v)`` coordinates."""

    if wavelength_m <= 0.0:
        raise ValueError("wavelength_m must be positive.")
    if not subapertures:
        raise ValueError("at least one sub-aperture is required.")

    ha = np.atleast_1d(np.asarray(hour_angle_rad, dtype=np.float64))
    positions = np.asarray(
        [np.asarray(sub.position_enu_m, dtype=np.float64) for sub in subapertures],
        dtype=np.float64,
    )
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("sub-aperture positions must have shape (M, 3).")

    sin_ha, cos_ha = np.sin(ha), np.cos(ha)
    sin_dec, cos_dec = np.sin(float(declination_rad)), np.cos(float(declination_rad))
    sin_lat, cos_lat = np.sin(float(latitude_rad)), np.cos(float(latitude_rad))

    pe, pn, pu = positions[:, 0], positions[:, 1], positions[:, 2]
    px = -sin_lat * pn + cos_lat * pu
    py = pe
    pz = cos_lat * pn + sin_lat * pu

    u_arr = sin_ha[:, None] * px[None, :] + cos_ha[:, None] * py[None, :]
    v_arr = (
        -sin_dec * cos_ha[:, None] * px[None, :]
        + sin_dec * sin_ha[:, None] * py[None, :]
        + cos_dec * pz[None, :]
    )

    u_arr = u_arr / float(wavelength_m)
    v_arr = v_arr / float(wavelength_m)

    if u_arr.shape[0] == 1 and np.ndim(hour_angle_rad) == 0:
        return u_arr.squeeze(axis=0), v_arr.squeeze(axis=0)
    return u_arr, v_arr


def compute_beam_patterns(
    *,
    u_sub: NDArray[np.float64],
    v_sub: NDArray[np.float64],
    amplitudes: NDArray[np.complex128],
    unitary: NDArray[np.complex128],
    port_indices: NDArray[np.intp],
    l_coords: NDArray[np.float64],
    m_coords: NDArray[np.float64],
    sub_diameters_m: NDArray[np.float64] | None = None,
    wavelength_m: float | None = None,
) -> NDArray[np.float64]:
    """Compute dark-port beam patterns ``|sum_k U_pk a_k exp(i phase)|^2``."""

    amps = np.asarray(amplitudes, dtype=np.complex128)
    u = np.asarray(unitary, dtype=np.complex128)
    ports = np.asarray(port_indices, dtype=np.intp)
    u_pos = np.asarray(u_sub, dtype=np.float64)
    v_pos = np.asarray(v_sub, dtype=np.float64)
    if u.ndim != 2 or u.shape[1] != amps.size:
        raise ValueError(
            "unitary must have shape (N_ports, M) matching amplitudes; "
            f"got {u.shape} and M={amps.size}."
        )
    if np.any((ports < 0) | (ports >= u.shape[0])):
        raise ValueError("port_indices must be valid row indices for unitary.")
    if u_pos.shape != (amps.size,) or v_pos.shape != (amps.size,):
        raise ValueError("u_sub and v_sub must have shape (M,).")

    l_2d, m_2d = np.meshgrid(l_coords, m_coords, indexing="xy")
    phase = 2.0 * np.pi * (
        u_pos[:, None, None] * l_2d[None, :, :]
        + v_pos[:, None, None] * m_2d[None, :, :]
    )
    weighted = amps[:, None, None] * np.exp(1j * phase)

    if sub_diameters_m is not None and wavelength_m is not None:
        diameters = np.asarray(sub_diameters_m, dtype=np.float64)
        if diameters.shape != (amps.size,):
            raise ValueError("sub_diameters_m must have shape (M,).")
        theta = np.sqrt(l_2d * l_2d + m_2d * m_2d)
        coupling = angular_coupling_factor(
            theta[None, :, :],
            diameters[:, None, None],
            float(wavelength_m),
        )
        weighted = weighted * np.sqrt(np.asarray(coupling, dtype=np.float64))

    fields = u[ports, :] @ weighted.reshape(amps.size, -1)
    return (np.abs(fields) ** 2).reshape(ports.size, len(m_coords), len(l_coords))


def build_response_matrix(
    *,
    subapertures: Sequence[SubAperture],
    amplitudes: NDArray[np.complex128],
    unitary: NDArray[np.complex128],
    dark_port_indices: NDArray[np.intp],
    hour_angles_rad: NDArray[np.float64],
    declination_rad: float,
    wavelength_m: float,
    l_coords: NDArray[np.float64],
    m_coords: NDArray[np.float64],
    latitude_rad: float = MAUNAKEA_LAT_RAD,
    unitaries_by_wavelength: Sequence[tuple[NDArray[np.complex128], float]] = (),
    roll_angles_rad: NDArray[np.float64] | None = None,
    wavelength_weights: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Build the linear response matrix mapping contrast pixels to dark ports.

    Each row is the band average of the per-wavelength beams, weighted by
    ``wavelength_weights`` (the star's photon fraction in each bin, summing to
    one; uniform when None). A planet of flat contrast ``c`` contributes
    ``c * F_lambda`` photons in each bin, so its band count is
    ``c * F_total * sum_lambda w_lambda B_lambda``.
    """

    wl_pairs = _unitary_wavelength_pairs(unitary, wavelength_m, unitaries_by_wavelength)
    n_time = int(len(hour_angles_rad))
    roll_angles = _validated_roll_angles(roll_angles_rad, n_time)
    n_dark = int(len(dark_port_indices))
    n_pix_sq = int(len(l_coords) * len(m_coords))
    response = np.zeros((n_time * n_dark, n_pix_sq), dtype=np.float64)

    for t_index, hour_angle in enumerate(hour_angles_rad):
        response[t_index * n_dark : (t_index + 1) * n_dark, :] = (
            _band_averaged_beams_flat(
                subapertures=subapertures,
                amplitudes=amplitudes,
                dark_port_indices=dark_port_indices,
                hour_angle_rad=float(hour_angle),
                declination_rad=declination_rad,
                roll_angle_rad=None if roll_angles is None else float(roll_angles[t_index]),
                l_coords=l_coords,
                m_coords=m_coords,
                latitude_rad=latitude_rad,
                unitary_wavelength_pairs=wl_pairs,
                wavelength_weights=wavelength_weights,
            )
        )
    return response


def build_wavelength_resolved_response_matrix(
    *,
    subapertures: Sequence[SubAperture],
    amplitudes: NDArray[np.complex128],
    unitary: NDArray[np.complex128],
    dark_port_indices: NDArray[np.intp],
    hour_angles_rad: NDArray[np.float64],
    declination_rad: float,
    wavelength_m: float,
    l_coords: NDArray[np.float64],
    m_coords: NDArray[np.float64],
    latitude_rad: float = MAUNAKEA_LAT_RAD,
    unitaries_by_wavelength: Sequence[tuple[NDArray[np.complex128], float]] = (),
    roll_angles_rad: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Build response rows for each time, wavelength, and dark port."""

    wl_pairs = _unitary_wavelength_pairs(unitary, wavelength_m, unitaries_by_wavelength)
    n_time = int(len(hour_angles_rad))
    roll_angles = _validated_roll_angles(roll_angles_rad, n_time)
    n_wl = int(len(wl_pairs))
    n_dark = int(len(dark_port_indices))
    n_pix_sq = int(len(l_coords) * len(m_coords))
    response = np.zeros((n_time * n_wl * n_dark, n_pix_sq), dtype=np.float64)

    for t_index, hour_angle in enumerate(hour_angles_rad):
        start = t_index * n_wl * n_dark
        response[start : start + n_wl * n_dark, :] = (
            _wavelength_resolved_beams_flat(
                subapertures=subapertures,
                amplitudes=amplitudes,
                dark_port_indices=dark_port_indices,
                hour_angle_rad=float(hour_angle),
                declination_rad=declination_rad,
                roll_angle_rad=None if roll_angles is None else float(roll_angles[t_index]),
                l_coords=l_coords,
                m_coords=m_coords,
                latitude_rad=latitude_rad,
                unitary_wavelength_pairs=wl_pairs,
            )
        )
    return response


def phase_chop_unitary_pair(
    unitary: NDArray[np.complex128],
    dark_port_indices: NDArray[np.intp],
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.intp]]:
    """Return conjugate pairwise phase-chop output matrices.

    Adjacent dark outputs are recombined as ``(a + i b) / sqrt(2)`` and
    ``(a - i b) / sqrt(2)``. The signed difference between those two chip
    states is the quadrature cross term that breaks centro-symmetric intensity
    degeneracies.
    """

    u = np.asarray(unitary, dtype=np.complex128)
    dark = np.asarray(dark_port_indices, dtype=np.intp)
    if u.ndim != 2:
        raise ValueError("unitary must be a 2D matrix.")
    if dark.ndim != 1 or dark.size < 2:
        raise ValueError("phase chopping requires at least two dark ports.")
    if dark.size % 2 != 0:
        raise ValueError("phase chopping requires an even number of dark ports.")
    if np.any((dark < 0) | (dark >= u.shape[0])):
        raise ValueError("dark_port_indices must be valid row indices for unitary.")

    first = u[dark[0::2], :]
    second = u[dark[1::2], :]
    scale = 1.0 / np.sqrt(2.0)
    plus = (first + 1j * second) * scale
    minus = (first - 1j * second) * scale
    chop_indices = np.arange(plus.shape[0], dtype=np.intp)
    return plus, minus, chop_indices


def simulate_phase_chopped_observation(
    *,
    scene: SkyScene,
    subapertures: Sequence[SubAperture],
    amplitudes: NDArray[np.complex128],
    unitary: NDArray[np.complex128],
    dark_port_indices: NDArray[np.intp],
    hour_angles_rad: NDArray[np.float64],
    declination_rad: float,
    wavelength_m: float,
    stellar_photon_rate_hz: float,
    null_depth: float,
    integration_time_per_step_s: float,
    dark_count_rate_hz_per_port: float = 0.0,
    sky_rate_hz_per_port: float = 0.0,
    seed: int | None = None,
    latitude_rad: float = MAUNAKEA_LAT_RAD,
    keep_response_matrix: bool = False,
    null_instability: float = 0.0,
    ft_servo_bandwidth_hz: float = 0.0,
    mkid_max_count_rate_hz: float | None = None,
    unitaries_by_wavelength: Sequence[tuple[NDArray[np.complex128], float]] = (),
    roll_angles_rad: NDArray[np.float64] | None = None,
    poisson: bool = True,
    wavelength_resolved: bool = False,
    stellar_photon_rate_hz_by_wavelength: NDArray[np.float64] | None = None,
    dark_count_rate_hz_per_port_by_wavelength: NDArray[np.float64] | None = None,
    sky_rate_hz_per_port_by_wavelength: NDArray[np.float64] | None = None,
) -> ObservationData:
    """Simulate a signed phase-chopped dark-port observation.

    The plus and minus rows are a unitary recombination of the dark pairs, so
    together they carry the dark power once: each state's rows receive their
    share of the realized dark leakage, the same share they take of a
    planet's light.
    """

    if stellar_photon_rate_hz < 0.0:
        raise ValueError("stellar_photon_rate_hz must be non-negative.")
    if null_depth < 0.0:
        raise ValueError("null_depth must be non-negative.")
    if integration_time_per_step_s <= 0.0:
        raise ValueError("integration_time_per_step_s must be positive.")

    rng = np.random.default_rng(seed)
    n_time = int(len(hour_angles_rad))
    total_area = _positive_total_amplitude_power(amplitudes)
    dt_state = 0.5 * float(integration_time_per_step_s)
    plus_pairs, minus_pairs, chop_indices = _phase_chop_wavelength_pairs(
        unitary=unitary,
        wavelength_m=wavelength_m,
        unitaries_by_wavelength=unitaries_by_wavelength,
        dark_port_indices=dark_port_indices,
    )
    n_chop = int(chop_indices.size)
    wavelengths = np.asarray([wavelength for _unitary, wavelength in plus_pairs])
    n_wl = int(len(plus_pairs))
    stellar_rates_by_wl = _rates_by_wavelength(
        total_rate_hz=float(stellar_photon_rate_hz),
        rates_by_wavelength=stellar_photon_rate_hz_by_wavelength,
        n_wavelengths=n_wl,
        name="stellar_photon_rate_hz_by_wavelength",
    )
    weights = _band_weights(stellar_rates_by_wl)
    leakage_plus, leakage_minus = _chopped_leakage_counts(
        amplitudes=amplitudes,
        plus_pairs=plus_pairs,
        minus_pairs=minus_pairs,
        null_depth=float(null_depth),
        stellar_photon_rate_hz_by_wavelength=stellar_rates_by_wl,
        integration_time_per_step_s=dt_state,
    )

    if wavelength_resolved:
        dark_rates_by_wl = _rates_by_wavelength(
            total_rate_hz=float(dark_count_rate_hz_per_port),
            rates_by_wavelength=dark_count_rate_hz_per_port_by_wavelength,
            n_wavelengths=n_wl,
            name="dark_count_rate_hz_per_port_by_wavelength",
        )
        sky_rates_by_wl = _rates_by_wavelength(
            total_rate_hz=float(sky_rate_hz_per_port),
            rates_by_wavelength=sky_rate_hz_per_port_by_wavelength,
            n_wavelengths=n_wl,
            name="sky_rate_hz_per_port_by_wavelength",
        )
        response_plus = build_wavelength_resolved_response_matrix(
            subapertures=subapertures,
            amplitudes=amplitudes,
            unitary=plus_pairs[0][0],
            dark_port_indices=chop_indices,
            hour_angles_rad=hour_angles_rad,
            declination_rad=declination_rad,
            wavelength_m=wavelength_m,
            l_coords=scene.l_coords,
            m_coords=scene.m_coords,
            latitude_rad=latitude_rad,
            unitaries_by_wavelength=plus_pairs,
            roll_angles_rad=roll_angles_rad,
        )
        response_minus = build_wavelength_resolved_response_matrix(
            subapertures=subapertures,
            amplitudes=amplitudes,
            unitary=minus_pairs[0][0],
            dark_port_indices=chop_indices,
            hour_angles_rad=hour_angles_rad,
            declination_rad=declination_rad,
            wavelength_m=wavelength_m,
            l_coords=scene.l_coords,
            m_coords=scene.m_coords,
            latitude_rad=latitude_rad,
            unitaries_by_wavelength=minus_pairs,
            roll_angles_rad=roll_angles_rad,
        )
        response_scale = (
            np.tile(np.repeat(stellar_rates_by_wl, n_chop), n_time)
            * dt_state
            / total_area
        )
        response_diff = response_plus - response_minus
        planet_plus = response_scale * (response_plus @ scene.image.ravel())
        planet_minus = response_scale * (response_minus @ scene.image.ravel())
        leakage_scales = _leakage_time_scales(
            n_time=n_time,
            null_instability=float(null_instability),
            ft_servo_bandwidth_hz=float(ft_servo_bandwidth_hz),
            integration_time_per_step_s=dt_state,
            rng=rng,
        )
        leakage_plus_tiled = _time_tiled_leakage(
            leakage_per_port=leakage_plus.ravel(),
            n_time=n_time,
            null_instability=float(null_instability),
            ft_servo_bandwidth_hz=float(ft_servo_bandwidth_hz),
            integration_time_per_step_s=dt_state,
            rng=rng,
            leakage_scales=leakage_scales,
        )
        leakage_minus_tiled = _time_tiled_leakage(
            leakage_per_port=leakage_minus.ravel(),
            n_time=n_time,
            null_instability=float(null_instability),
            ft_servo_bandwidth_hz=float(ft_servo_bandwidth_hz),
            integration_time_per_step_s=dt_state,
            rng=rng,
            leakage_scales=leakage_scales,
        )
        detector_template = np.repeat(dark_rates_by_wl + sky_rates_by_wl, n_chop)
        detector_counts = np.tile(detector_template * dt_state, n_time)
        baseline_plus = leakage_plus_tiled + detector_counts
        baseline_minus = leakage_minus_tiled + detector_counts
        expected_plus = np.maximum(planet_plus + baseline_plus, 0.0)
        expected_minus = np.maximum(planet_minus + baseline_minus, 0.0)

        if mkid_max_count_rate_hz is not None and mkid_max_count_rate_hz > 0.0:
            cap = float(mkid_max_count_rate_hz) * dt_state
            expected_plus = np.minimum(expected_plus, cap)
            expected_minus = np.minimum(expected_minus, cap)

        measured_plus = (
            rng.poisson(expected_plus).astype(np.float64)
            if poisson
            else expected_plus.astype(np.float64, copy=True)
        )
        measured_minus = (
            rng.poisson(expected_minus).astype(np.float64)
            if poisson
            else expected_minus.astype(np.float64, copy=True)
        )
        measurement_shape = (n_time, n_wl, n_chop)
        return ObservationData(
            measurements=(measured_plus - measured_minus).reshape(measurement_shape),
            hour_angles_rad=np.asarray(hour_angles_rad, dtype=np.float64),
            dark_port_indices=chop_indices,
            stellar_leakage_per_port=np.sum(leakage_plus + leakage_minus, axis=0),
            dark_counts_per_step=float(dark_count_rate_hz_per_port) * dt_state,
            sky_counts_per_step=float(sky_rate_hz_per_port) * dt_state,
            response_matrix=response_diff if keep_response_matrix else None,
            response_scale_per_row=response_scale,
            baseline_counts=(baseline_plus - baseline_minus).reshape(measurement_shape),
            variance_counts=(expected_plus + expected_minus).reshape(measurement_shape),
            wavelengths_m=wavelengths.astype(np.float64, copy=True),
            wavelength_weights=weights,
        )

    response_plus = build_response_matrix(
        subapertures=subapertures,
        amplitudes=amplitudes,
        unitary=plus_pairs[0][0],
        dark_port_indices=chop_indices,
        hour_angles_rad=hour_angles_rad,
        declination_rad=declination_rad,
        wavelength_m=wavelength_m,
        l_coords=scene.l_coords,
        m_coords=scene.m_coords,
        latitude_rad=latitude_rad,
        unitaries_by_wavelength=plus_pairs,
        roll_angles_rad=roll_angles_rad,
        wavelength_weights=weights,
    )
    response_minus = build_response_matrix(
        subapertures=subapertures,
        amplitudes=amplitudes,
        unitary=minus_pairs[0][0],
        dark_port_indices=chop_indices,
        hour_angles_rad=hour_angles_rad,
        declination_rad=declination_rad,
        wavelength_m=wavelength_m,
        l_coords=scene.l_coords,
        m_coords=scene.m_coords,
        latitude_rad=latitude_rad,
        unitaries_by_wavelength=minus_pairs,
        roll_angles_rad=roll_angles_rad,
        wavelength_weights=weights,
    )
    response_scale = np.full(
        n_time * n_chop,
        float(stellar_photon_rate_hz) * dt_state / total_area,
        dtype=np.float64,
    )
    response_diff = response_plus - response_minus
    planet_plus = response_scale * (response_plus @ scene.image.ravel())
    planet_minus = response_scale * (response_minus @ scene.image.ravel())
    leakage_plus = leakage_plus.sum(axis=0)
    leakage_minus = leakage_minus.sum(axis=0)
    leakage_scales = _leakage_time_scales(
        n_time=n_time,
        null_instability=float(null_instability),
        ft_servo_bandwidth_hz=float(ft_servo_bandwidth_hz),
        integration_time_per_step_s=dt_state,
        rng=rng,
    )
    leakage_plus_tiled = _time_tiled_leakage(
        leakage_per_port=leakage_plus,
        n_time=n_time,
        null_instability=float(null_instability),
        ft_servo_bandwidth_hz=float(ft_servo_bandwidth_hz),
        integration_time_per_step_s=dt_state,
        rng=rng,
        leakage_scales=leakage_scales,
    )
    leakage_minus_tiled = _time_tiled_leakage(
        leakage_per_port=leakage_minus,
        n_time=n_time,
        null_instability=float(null_instability),
        ft_servo_bandwidth_hz=float(ft_servo_bandwidth_hz),
        integration_time_per_step_s=dt_state,
        rng=rng,
        leakage_scales=leakage_scales,
    )
    detector_counts = (
        float(dark_count_rate_hz_per_port) + float(sky_rate_hz_per_port)
    ) * dt_state
    baseline_plus = leakage_plus_tiled + detector_counts
    baseline_minus = leakage_minus_tiled + detector_counts
    expected_plus = np.maximum(planet_plus + baseline_plus, 0.0)
    expected_minus = np.maximum(planet_minus + baseline_minus, 0.0)

    if mkid_max_count_rate_hz is not None and mkid_max_count_rate_hz > 0.0:
        cap = float(mkid_max_count_rate_hz) * dt_state
        expected_plus = np.minimum(expected_plus, cap)
        expected_minus = np.minimum(expected_minus, cap)

    measured_plus = (
        rng.poisson(expected_plus).astype(np.float64)
        if poisson
        else expected_plus.astype(np.float64, copy=True)
    )
    measured_minus = (
        rng.poisson(expected_minus).astype(np.float64)
        if poisson
        else expected_minus.astype(np.float64, copy=True)
    )
    return ObservationData(
        measurements=(measured_plus - measured_minus).reshape(n_time, n_chop),
        hour_angles_rad=np.asarray(hour_angles_rad, dtype=np.float64),
        dark_port_indices=chop_indices,
        stellar_leakage_per_port=leakage_plus + leakage_minus,
        dark_counts_per_step=float(dark_count_rate_hz_per_port) * dt_state,
        sky_counts_per_step=float(sky_rate_hz_per_port) * dt_state,
        response_matrix=response_diff if keep_response_matrix else None,
        response_scale_per_row=response_scale,
        baseline_counts=(baseline_plus - baseline_minus).reshape(n_time, n_chop),
        variance_counts=(expected_plus + expected_minus).reshape(n_time, n_chop),
        wavelengths_m=wavelengths.astype(np.float64, copy=True),
        wavelength_weights=weights,
    )


def _chopped_leakage_counts(
    *,
    amplitudes: NDArray[np.complex128],
    plus_pairs: Sequence[tuple[NDArray[np.complex128], float]],
    minus_pairs: Sequence[tuple[NDArray[np.complex128], float]],
    null_depth: float,
    stellar_photon_rate_hz_by_wavelength: NDArray[np.float64],
    integration_time_per_step_s: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Split one realized dark leakage over the plus and minus rows jointly."""

    joint = _stellar_leakage_counts_per_wavelength(
        amplitudes=amplitudes,
        rows_by_wavelength=[
            np.vstack((plus, minus))
            for (plus, _wl), (minus, _wl2) in zip(plus_pairs, minus_pairs)
        ],
        null_depth=null_depth,
        stellar_photon_rate_hz_by_wavelength=stellar_photon_rate_hz_by_wavelength,
        integration_time_per_step_s=integration_time_per_step_s,
    )
    n_chop = int(plus_pairs[0][0].shape[0])
    return joint[:, :n_chop], joint[:, n_chop:]


def simulate_observation(
    *,
    scene: SkyScene,
    subapertures: Sequence[SubAperture],
    amplitudes: NDArray[np.complex128],
    unitary: NDArray[np.complex128],
    dark_port_indices: NDArray[np.intp],
    hour_angles_rad: NDArray[np.float64],
    declination_rad: float,
    wavelength_m: float,
    stellar_photon_rate_hz: float,
    null_depth: float,
    integration_time_per_step_s: float,
    dark_count_rate_hz_per_port: float = 0.0,
    sky_rate_hz_per_port: float = 0.0,
    seed: int | None = None,
    latitude_rad: float = MAUNAKEA_LAT_RAD,
    keep_response_matrix: bool = False,
    null_instability: float = 0.0,
    ft_servo_bandwidth_hz: float = 0.0,
    mkid_max_count_rate_hz: float | None = None,
    unitaries_by_wavelength: Sequence[tuple[NDArray[np.complex128], float]] = (),
    roll_angles_rad: NDArray[np.float64] | None = None,
    poisson: bool = True,
    wavelength_resolved: bool = False,
    stellar_photon_rate_hz_by_wavelength: NDArray[np.float64] | None = None,
    dark_count_rate_hz_per_port_by_wavelength: NDArray[np.float64] | None = None,
    sky_rate_hz_per_port_by_wavelength: NDArray[np.float64] | None = None,
) -> ObservationData:
    """Simulate dark-port photon counts for a point-source scene."""

    if stellar_photon_rate_hz < 0.0:
        raise ValueError("stellar_photon_rate_hz must be non-negative.")
    if null_depth < 0.0:
        raise ValueError("null_depth must be non-negative.")
    if integration_time_per_step_s <= 0.0:
        raise ValueError("integration_time_per_step_s must be positive.")

    rng = np.random.default_rng(seed)
    n_time = int(len(hour_angles_rad))
    n_dark = int(len(dark_port_indices))
    wl_pairs = _unitary_wavelength_pairs(unitary, wavelength_m, unitaries_by_wavelength)
    n_wl = int(len(wl_pairs))
    wavelengths = np.asarray([wavelength for _unitary, wavelength in wl_pairs])
    total_area = _positive_total_amplitude_power(amplitudes)
    dt = float(integration_time_per_step_s)
    stellar_rates_by_wl = _rates_by_wavelength(
        total_rate_hz=float(stellar_photon_rate_hz),
        rates_by_wavelength=stellar_photon_rate_hz_by_wavelength,
        n_wavelengths=n_wl,
        name="stellar_photon_rate_hz_by_wavelength",
    )
    weights = _band_weights(stellar_rates_by_wl)
    dark = np.asarray(dark_port_indices, dtype=np.intp)
    leakage_by_wl_port = _stellar_leakage_counts_per_wavelength(
        amplitudes=amplitudes,
        rows_by_wavelength=[u_wl[dark, :] for u_wl, _wl in wl_pairs],
        null_depth=float(null_depth),
        stellar_photon_rate_hz_by_wavelength=stellar_rates_by_wl,
        integration_time_per_step_s=dt,
    )

    if wavelength_resolved:
        dark_rates_by_wl = _rates_by_wavelength(
            total_rate_hz=float(dark_count_rate_hz_per_port),
            rates_by_wavelength=dark_count_rate_hz_per_port_by_wavelength,
            n_wavelengths=n_wl,
            name="dark_count_rate_hz_per_port_by_wavelength",
        )
        sky_rates_by_wl = _rates_by_wavelength(
            total_rate_hz=float(sky_rate_hz_per_port),
            rates_by_wavelength=sky_rate_hz_per_port_by_wavelength,
            n_wavelengths=n_wl,
            name="sky_rate_hz_per_port_by_wavelength",
        )
        response = build_wavelength_resolved_response_matrix(
            subapertures=subapertures,
            amplitudes=amplitudes,
            unitary=unitary,
            dark_port_indices=dark_port_indices,
            hour_angles_rad=hour_angles_rad,
            declination_rad=declination_rad,
            wavelength_m=wavelength_m,
            l_coords=scene.l_coords,
            m_coords=scene.m_coords,
            latitude_rad=latitude_rad,
            unitaries_by_wavelength=wl_pairs,
            roll_angles_rad=roll_angles_rad,
        )
        response_scale = (
            np.tile(np.repeat(stellar_rates_by_wl, n_dark), n_time)
            * dt
            / total_area
        )
        planet_counts = response_scale * (response @ scene.image.ravel())
        leakage_tiled = _time_tiled_leakage(
            leakage_per_port=leakage_by_wl_port.ravel(),
            n_time=n_time,
            null_instability=float(null_instability),
            ft_servo_bandwidth_hz=float(ft_servo_bandwidth_hz),
            integration_time_per_step_s=dt,
            rng=rng,
        )
        detector_counts_template = np.repeat(dark_rates_by_wl + sky_rates_by_wl, n_dark)
        baseline = leakage_tiled + np.tile(detector_counts_template * dt, n_time)
        expected = np.maximum(planet_counts + baseline, 0.0)

        if mkid_max_count_rate_hz is not None and mkid_max_count_rate_hz > 0.0:
            expected = np.minimum(expected, float(mkid_max_count_rate_hz) * dt)

        measured = (
            rng.poisson(expected).astype(np.float64)
            if poisson
            else expected.astype(np.float64, copy=True)
        )
        measurement_shape = (n_time, n_wl, n_dark)
        return ObservationData(
            measurements=measured.reshape(measurement_shape),
            hour_angles_rad=np.asarray(hour_angles_rad, dtype=np.float64),
            dark_port_indices=np.asarray(dark_port_indices, dtype=np.intp),
            stellar_leakage_per_port=np.sum(leakage_by_wl_port, axis=0),
            dark_counts_per_step=float(dark_count_rate_hz_per_port) * dt,
            sky_counts_per_step=float(sky_rate_hz_per_port) * dt,
            response_matrix=response if keep_response_matrix else None,
            response_scale_per_row=response_scale,
            baseline_counts=baseline.reshape(measurement_shape),
            variance_counts=expected.reshape(measurement_shape),
            wavelengths_m=wavelengths.astype(np.float64, copy=True),
            wavelength_weights=weights,
        )

    response = build_response_matrix(
        subapertures=subapertures,
        amplitudes=amplitudes,
        unitary=unitary,
        dark_port_indices=dark_port_indices,
        hour_angles_rad=hour_angles_rad,
        declination_rad=declination_rad,
        wavelength_m=wavelength_m,
        l_coords=scene.l_coords,
        m_coords=scene.m_coords,
        latitude_rad=latitude_rad,
        unitaries_by_wavelength=unitaries_by_wavelength,
        roll_angles_rad=roll_angles_rad,
        wavelength_weights=weights,
    )

    response_scale = np.full(n_time * n_dark, float(stellar_photon_rate_hz) * dt / total_area)
    planet_counts = response_scale * (response @ scene.image.ravel())
    leakage_per_port = leakage_by_wl_port.sum(axis=0)
    leakage_tiled = _time_tiled_leakage(
        leakage_per_port=leakage_per_port,
        n_time=n_time,
        null_instability=float(null_instability),
        ft_servo_bandwidth_hz=float(ft_servo_bandwidth_hz),
        integration_time_per_step_s=dt,
        rng=rng,
    )

    baseline = (
        leakage_tiled
        + float(dark_count_rate_hz_per_port) * dt
        + float(sky_rate_hz_per_port) * dt
    )
    expected = planet_counts + baseline
    expected = np.maximum(expected, 0.0)

    if mkid_max_count_rate_hz is not None and mkid_max_count_rate_hz > 0.0:
        expected = np.minimum(expected, float(mkid_max_count_rate_hz) * dt)

    measured = (
        rng.poisson(expected).astype(np.float64)
        if poisson
        else expected.astype(np.float64, copy=True)
    )
    return ObservationData(
        measurements=measured.reshape(n_time, n_dark),
        hour_angles_rad=np.asarray(hour_angles_rad, dtype=np.float64),
        dark_port_indices=np.asarray(dark_port_indices, dtype=np.intp),
        stellar_leakage_per_port=leakage_per_port,
        dark_counts_per_step=float(dark_count_rate_hz_per_port) * dt,
        sky_counts_per_step=float(sky_rate_hz_per_port) * dt,
        response_matrix=response if keep_response_matrix else None,
        response_scale_per_row=response_scale,
        baseline_counts=baseline.reshape(n_time, n_dark),
        variance_counts=expected.reshape(n_time, n_dark),
        wavelengths_m=wavelengths.astype(np.float64, copy=True),
        wavelength_weights=weights,
    )


def compute_synthesized_beam(
    *,
    subapertures: Sequence[SubAperture],
    amplitudes: NDArray[np.complex128],
    unitary: NDArray[np.complex128],
    dark_port_indices: NDArray[np.intp],
    hour_angles_rad: NDArray[np.float64],
    declination_rad: float,
    wavelength_m: float,
    l_coords: NDArray[np.float64],
    m_coords: NDArray[np.float64],
    latitude_rad: float = MAUNAKEA_LAT_RAD,
    unitaries_by_wavelength: Sequence[tuple[NDArray[np.complex128], float]] = (),
    roll_angles_rad: NDArray[np.float64] | None = None,
    wavelength_weights: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Compute the normalized synthesized beam for the dark-port set."""

    synth_flat = np.zeros(len(l_coords) * len(m_coords), dtype=np.float64)
    wl_pairs = _unitary_wavelength_pairs(unitary, wavelength_m, unitaries_by_wavelength)
    roll_angles = _validated_roll_angles(roll_angles_rad, len(hour_angles_rad))
    for t_index, hour_angle in enumerate(hour_angles_rad):
        beams_flat = _band_averaged_beams_flat(
            subapertures=subapertures,
            amplitudes=amplitudes,
            dark_port_indices=dark_port_indices,
            hour_angle_rad=float(hour_angle),
            declination_rad=declination_rad,
            roll_angle_rad=None if roll_angles is None else float(roll_angles[t_index]),
            l_coords=l_coords,
            m_coords=m_coords,
            latitude_rad=latitude_rad,
            unitary_wavelength_pairs=wl_pairs,
            wavelength_weights=wavelength_weights,
        )
        synth_flat += beams_flat.sum(axis=0)

    beam = synth_flat.reshape(len(m_coords), len(l_coords))
    peak = float(np.max(beam)) if beam.size else 0.0
    if peak > 0.0:
        beam = beam / peak
    return beam


def reconstruct_snr_map(
    *,
    data: ObservationData,
    subapertures: Sequence[SubAperture],
    amplitudes: NDArray[np.complex128],
    unitary: NDArray[np.complex128],
    declination_rad: float,
    wavelength_m: float,
    l_coords: NDArray[np.float64],
    m_coords: NDArray[np.float64],
    latitude_rad: float = MAUNAKEA_LAT_RAD,
    null_instability: float = 0.0,
    ft_servo_bandwidth_hz: float = 0.0,
    integration_time_per_step_s: float = 0.0,
    unitaries_by_wavelength: Sequence[tuple[NDArray[np.complex128], float]] = (),
    roll_angles_rad: NDArray[np.float64] | None = None,
) -> ReconstructedImage:
    """Return the matched-filter SNR map for a dark-port observation."""

    n_dark = int(len(data.dark_port_indices))
    n_time = int(len(data.hour_angles_rad))
    n_rows = int(data.measurements.size)
    expected_rows = n_time * n_dark
    if n_rows != expected_rows:
        raise ValueError(
            "reconstruct_snr_map expects band-averaged measurements with shape "
            f"(n_time, n_dark); got {data.measurements.shape} for "
            f"n_time={n_time}, n_dark={n_dark}."
        )
    baseline_flat = _baseline_counts_flat(data, n_time=n_time)
    d = data.measurements.ravel() - baseline_flat
    variance_flat = _variance_counts_flat(data, n_time=n_time)
    inv_variance_flat = np.divide(
        1.0,
        variance_flat,
        out=np.zeros_like(variance_flat, dtype=np.float64),
        where=variance_flat > 0.0,
    )
    response = _response_matrix_for_reconstruction(
        data=data,
        subapertures=subapertures,
        amplitudes=amplitudes,
        unitary=unitary,
        declination_rad=declination_rad,
        wavelength_m=wavelength_m,
        l_coords=l_coords,
        m_coords=m_coords,
        latitude_rad=latitude_rad,
        unitaries_by_wavelength=unitaries_by_wavelength,
        roll_angles_rad=roll_angles_rad,
    )
    response_scale = _response_scale_flat(data, n_rows=n_rows)
    template = response * response_scale[:, None]
    sqrt_inv_variance = np.sqrt(inv_variance_flat)
    design = template * sqrt_inv_variance[:, None]
    rhs = d * sqrt_inv_variance
    design, rhs = _project_common_response_mode(design, rhs)

    if null_instability > 0.0:
        if ft_servo_bandwidth_hz > 0.0 and integration_time_per_step_s > 0.0:
            syst_var_factor = (
                float(null_instability) ** 2
                / (float(integration_time_per_step_s) * 2.0 * float(ft_servo_bandwidth_hz))
            )
        else:
            syst_var_factor = float(null_instability) ** 2
        leakage_counts = np.tile(
            np.asarray(data.stellar_leakage_per_port, dtype=np.float64),
            n_time,
        )
    else:
        syst_var_factor = 0.0
        leakage_counts = None

    dirty_flat = rhs @ design
    variance_flat = np.sum(design * design, axis=0)
    if syst_var_factor > 0.0 and leakage_counts is not None:
        syst_dirty = (leakage_counts * sqrt_inv_variance) @ design
        variance_flat += syst_var_factor * (syst_dirty * syst_dirty)

    denominator = np.sqrt(variance_flat)
    snr_flat = np.divide(
        dirty_flat,
        denominator,
        out=np.zeros_like(dirty_flat),
        where=denominator > 0.0,
    )
    snr = snr_flat.reshape(len(m_coords), len(l_coords))
    synth = _matched_filter_beam_rms(response, len(m_coords), len(l_coords))

    return ReconstructedImage(
        image=snr,
        residuals=np.zeros_like(snr),
        clean_components=(),
        synthesized_beam=synth,
        method="snr_map",
        n_iterations=0,
    )


def _baseline_counts_flat(
    data: ObservationData,
    *,
    n_time: int,
) -> NDArray[np.float64]:
    n_rows = int(data.measurements.size)
    if data.baseline_counts is not None and data.baseline_counts.size == n_rows:
        return np.asarray(data.baseline_counts, dtype=np.float64).ravel()
    mu0_per_port = (
        data.stellar_leakage_per_port
        + data.dark_counts_per_step
        + data.sky_counts_per_step
    )
    return np.tile(np.asarray(mu0_per_port, dtype=np.float64), int(n_time))


def _variance_counts_flat(
    data: ObservationData,
    *,
    n_time: int,
) -> NDArray[np.float64]:
    n_rows = int(data.measurements.size)
    if data.variance_counts is not None and data.variance_counts.size == n_rows:
        return np.asarray(data.variance_counts, dtype=np.float64).ravel()
    return _baseline_counts_flat(data, n_time=n_time)


def _matched_filter_beam_rms(
    response: NDArray[np.float64],
    n_m: int,
    n_l: int,
) -> NDArray[np.float64]:
    beam = np.sqrt(np.sum(np.asarray(response, dtype=np.float64) ** 2, axis=0))
    image = beam.reshape(int(n_m), int(n_l))
    peak = float(np.max(image)) if image.size else 0.0
    if peak > 0.0:
        image = image / peak
    return image


def _response_matrix_for_reconstruction(
    *,
    data: ObservationData,
    subapertures: Sequence[SubAperture],
    amplitudes: NDArray[np.complex128],
    unitary: NDArray[np.complex128],
    declination_rad: float,
    wavelength_m: float,
    l_coords: NDArray[np.float64],
    m_coords: NDArray[np.float64],
    latitude_rad: float,
    unitaries_by_wavelength: Sequence[tuple[NDArray[np.complex128], float]],
    roll_angles_rad: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    n_rows = int(data.measurements.size)
    n_pix_sq = int(len(l_coords) * len(m_coords))
    if data.response_matrix is not None and data.response_matrix.shape == (
        n_rows,
        n_pix_sq,
    ):
        return np.asarray(data.response_matrix, dtype=np.float64)
    return build_response_matrix(
        subapertures=subapertures,
        amplitudes=amplitudes,
        unitary=unitary,
        dark_port_indices=data.dark_port_indices,
        hour_angles_rad=data.hour_angles_rad,
        declination_rad=declination_rad,
        wavelength_m=wavelength_m,
        l_coords=l_coords,
        m_coords=m_coords,
        latitude_rad=latitude_rad,
        unitaries_by_wavelength=unitaries_by_wavelength,
        roll_angles_rad=roll_angles_rad,
        wavelength_weights=data.wavelength_weights,
    )


def _response_scale_flat(
    data: ObservationData,
    *,
    n_rows: int,
) -> NDArray[np.float64]:
    if data.response_scale_per_row is not None and data.response_scale_per_row.size == n_rows:
        return np.asarray(data.response_scale_per_row, dtype=np.float64).ravel()
    return np.ones(int(n_rows), dtype=np.float64)


def _project_common_response_mode(
    design_matrix: NDArray[np.float64],
    rhs: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    common = _common_response_mode(design_matrix)
    if common is None:
        return design_matrix, rhs

    projected_design = _project_against_common_mode(design_matrix, common)
    common_norm_sq = float(common @ common)
    rhs_coeff = float(common @ rhs) / common_norm_sq
    projected_rhs = rhs - common * rhs_coeff
    return projected_design, projected_rhs


def _common_response_mode(
    design_matrix: NDArray[np.float64],
) -> NDArray[np.float64] | None:
    common = np.mean(design_matrix, axis=1)
    common_norm_sq = float(common @ common)
    if common_norm_sq <= 0.0:
        return None
    return common


def _project_against_common_mode(
    design_matrix: NDArray[np.float64],
    common: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    if common is None:
        return design_matrix
    common_norm_sq = float(common @ common)
    col_coeff = (common @ design_matrix) / common_norm_sq
    return design_matrix - common[:, None] * col_coeff[None, :]


def reconstruct_nnls(
    *,
    data: ObservationData,
    subapertures: Sequence[SubAperture],
    amplitudes: NDArray[np.complex128],
    unitary: NDArray[np.complex128],
    declination_rad: float,
    wavelength_m: float,
    l_coords: NDArray[np.float64],
    m_coords: NDArray[np.float64],
    latitude_rad: float = MAUNAKEA_LAT_RAD,
    stellar_photon_rate_hz: float | None = None,
    integration_time_per_step_s: float | None = None,
    unitaries_by_wavelength: Sequence[tuple[NDArray[np.complex128], float]] = (),
    roll_angles_rad: NDArray[np.float64] | None = None,
    candidate_indices: NDArray[np.intp] | None = None,
    regularization_fraction: float = 0.0,
) -> ReconstructedImage:
    """Solve a non-negative least-squares reconstruction in contrast units."""

    n_rows = int(data.measurements.size)
    if data.response_matrix is not None and data.response_matrix.shape[0] == n_rows:
        response = np.asarray(data.response_matrix, dtype=np.float64)
    else:
        response = build_response_matrix(
            subapertures=subapertures,
            amplitudes=amplitudes,
            unitary=unitary,
            dark_port_indices=data.dark_port_indices,
            hour_angles_rad=data.hour_angles_rad,
            declination_rad=declination_rad,
            wavelength_m=wavelength_m,
            l_coords=l_coords,
            m_coords=m_coords,
            latitude_rad=latitude_rad,
            unitaries_by_wavelength=unitaries_by_wavelength,
            roll_angles_rad=roll_angles_rad,
            wavelength_weights=data.wavelength_weights,
        )
    if data.baseline_counts is not None and data.baseline_counts.size == n_rows:
        rhs = data.measurements.ravel() - np.asarray(
            data.baseline_counts,
            dtype=np.float64,
        ).ravel()
    else:
        mu0_per_port = (
            data.stellar_leakage_per_port
            + data.dark_counts_per_step
            + data.sky_counts_per_step
        )
        rhs = data.measurements.ravel() - np.tile(
            mu0_per_port,
            len(data.hour_angles_rad),
        )

    response_scale = data.response_scale_per_row
    has_row_scale = response_scale is not None and response_scale.size == response.shape[0]
    full_n_pix = int(response.shape[1])
    if candidate_indices is None:
        solve_response = response
        selected_indices = None
    else:
        selected_indices = _validated_candidate_indices(candidate_indices, full_n_pix)
        solve_response = response[:, selected_indices]
    if has_row_scale:
        design_matrix = solve_response * np.asarray(response_scale, dtype=np.float64)[:, None]
    else:
        design_matrix = solve_response
    if regularization_fraction > 0.0:
        design_matrix, rhs = _regularized_design(
            design_matrix,
            rhs,
            regularization_fraction=float(regularization_fraction),
        )
    solve = _solve_nnls(design_matrix, rhs)
    if selected_indices is None:
        image_flat = solve.x
    else:
        image_flat = np.zeros(full_n_pix, dtype=np.float64)
        image_flat[selected_indices] = solve.x
    if (
        not has_row_scale
        and stellar_photon_rate_hz is not None
        and integration_time_per_step_s is not None
    ):
        cal = (
            float(stellar_photon_rate_hz)
            * float(integration_time_per_step_s)
            / _positive_total_amplitude_power(amplitudes)
        )
        if cal > 0.0:
            image_flat = image_flat / cal

    image = image_flat.reshape(len(m_coords), len(l_coords))
    return ReconstructedImage(
        image=image,
        residuals=np.zeros_like(image),
        clean_components=(),
        synthesized_beam=compute_synthesized_beam(
            subapertures=subapertures,
            amplitudes=amplitudes,
            unitary=unitary,
            dark_port_indices=data.dark_port_indices,
            hour_angles_rad=data.hour_angles_rad,
            declination_rad=declination_rad,
            wavelength_m=wavelength_m,
            l_coords=l_coords,
            m_coords=m_coords,
            latitude_rad=latitude_rad,
            unitaries_by_wavelength=unitaries_by_wavelength,
            roll_angles_rad=roll_angles_rad,
            wavelength_weights=data.wavelength_weights,
        ),
        method="nnls",
        n_iterations=solve.n_iterations,
        converged=solve.converged,
        solver_status=solve.status,
        solver_message=solve.message,
        residual_norm=solve.residual_norm,
    )


@dataclass(frozen=True)
class _NNLSSolution:
    x: NDArray[np.float64]
    residual_norm: float
    n_iterations: int | None
    converged: bool
    status: int | None
    message: str


_LSQ_MAX_ITER = 200


def _solve_nnls(
    design_matrix: NDArray[np.float64],
    rhs: NDArray[np.float64],
) -> _NNLSSolution:
    """Non-negative least squares, never returning a failed solve silently.

    Unsigned designs use scipy's active-set ``nnls``, which raises at its own
    iteration limit and reports no iteration count. Signed (phase-chopped)
    designs use ``lsq_linear``, which always returns an iterate: a status of
    0 (iteration limit) or -1 (no progress) is retried on the
    column-normalized design with ``max(200, 20 * n_columns)`` iterations.
    A solve that still fails is returned with ``converged=False`` and a
    warning.
    """

    if not np.any(design_matrix < 0.0):
        x, rnorm = nnls(design_matrix, rhs)
        return _NNLSSolution(x, float(rnorm), None, True, None, "")

    result = _lsq_nonnegative(design_matrix, rhs, max_iter=_LSQ_MAX_ITER)
    n_iterations = int(result.nit)
    x = np.asarray(result.x, dtype=np.float64)
    if result.status <= 0:
        column_norms = np.linalg.norm(design_matrix, axis=0)
        scale = np.where(column_norms > 0.0, column_norms, 1.0)
        result = _lsq_nonnegative(
            design_matrix / scale,
            rhs,
            max_iter=max(_LSQ_MAX_ITER, 20 * design_matrix.shape[1]),
        )
        n_iterations += int(result.nit)
        x = np.where(column_norms > 0.0, result.x / scale, 0.0)
    converged = bool(result.status > 0)
    if not converged:
        warnings.warn(
            "bounded NNLS solve did not converge after column normalization "
            f"(status {result.status}: {result.message}); the reconstruction "
            "is a non-converged iterate.",
            RuntimeWarning,
            stacklevel=3,
        )
    return _NNLSSolution(
        x=x,
        residual_norm=float(np.linalg.norm(design_matrix @ x - rhs)),
        n_iterations=n_iterations,
        converged=converged,
        status=int(result.status),
        message=str(result.message),
    )


def _lsq_nonnegative(
    design_matrix: NDArray[np.float64],
    rhs: NDArray[np.float64],
    *,
    max_iter: int,
):
    return lsq_linear(
        design_matrix,
        rhs,
        bounds=(0.0, np.inf),
        method="trf",
        tol=1.0e-8,
        lsmr_tol="auto",
        max_iter=int(max_iter),
    )


def _regularized_design(
    design_matrix: NDArray[np.float64],
    rhs: NDArray[np.float64],
    *,
    regularization_fraction: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if regularization_fraction <= 0.0:
        return design_matrix, rhs
    column_norms = np.linalg.norm(design_matrix, axis=0)
    positive = column_norms[column_norms > 0.0]
    if positive.size == 0:
        return design_matrix, rhs
    regularization = float(regularization_fraction) * float(np.median(positive))
    penalty = regularization * np.eye(design_matrix.shape[1], dtype=np.float64)
    return (
        np.vstack((design_matrix, penalty)),
        np.concatenate((rhs, np.zeros(design_matrix.shape[1], dtype=np.float64))),
    )


def _validated_candidate_indices(
    candidate_indices: NDArray[np.intp],
    n_columns: int,
) -> NDArray[np.intp]:
    values = np.asarray(candidate_indices, dtype=np.intp).ravel()
    if values.size == 0:
        raise ValueError("candidate_indices must not be empty.")
    if np.any((values < 0) | (values >= int(n_columns))):
        raise ValueError("candidate_indices contains out-of-range columns.")
    return np.unique(values)


def _unitary_wavelength_pairs(
    unitary: NDArray[np.complex128],
    wavelength_m: float,
    unitaries_by_wavelength: Sequence[tuple[NDArray[np.complex128], float]],
) -> tuple[tuple[NDArray[np.complex128], float], ...]:
    if unitaries_by_wavelength:
        return tuple(
            (np.asarray(u, dtype=np.complex128), float(wl))
            for u, wl in unitaries_by_wavelength
        )
    return ((np.asarray(unitary, dtype=np.complex128), float(wavelength_m)),)


def _phase_chop_wavelength_pairs(
    *,
    unitary: NDArray[np.complex128],
    wavelength_m: float,
    unitaries_by_wavelength: Sequence[tuple[NDArray[np.complex128], float]],
    dark_port_indices: NDArray[np.intp],
) -> tuple[
    tuple[tuple[NDArray[np.complex128], float], ...],
    tuple[tuple[NDArray[np.complex128], float], ...],
    NDArray[np.intp],
]:
    plus_pairs: list[tuple[NDArray[np.complex128], float]] = []
    minus_pairs: list[tuple[NDArray[np.complex128], float]] = []
    chop_indices: NDArray[np.intp] | None = None
    for u_wl, wavelength in _unitary_wavelength_pairs(
        unitary,
        wavelength_m,
        unitaries_by_wavelength,
    ):
        plus, minus, indices = phase_chop_unitary_pair(u_wl, dark_port_indices)
        if chop_indices is None:
            chop_indices = indices
        elif indices.shape != chop_indices.shape:
            raise ValueError("phase-chop dark-port pairing changed with wavelength.")
        plus_pairs.append((plus, float(wavelength)))
        minus_pairs.append((minus, float(wavelength)))
    if chop_indices is None:
        raise ValueError("phase chopping requires at least one wavelength.")
    return tuple(plus_pairs), tuple(minus_pairs), chop_indices


def _band_averaged_beams_flat(
    *,
    subapertures: Sequence[SubAperture],
    amplitudes: NDArray[np.complex128],
    dark_port_indices: NDArray[np.intp],
    hour_angle_rad: float,
    declination_rad: float,
    roll_angle_rad: float | None,
    l_coords: NDArray[np.float64],
    m_coords: NDArray[np.float64],
    latitude_rad: float,
    unitary_wavelength_pairs: Sequence[tuple[NDArray[np.complex128], float]],
    wavelength_weights: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    n_dark = int(len(dark_port_indices))
    n_pix_sq = int(len(l_coords) * len(m_coords))
    beams_flat = np.zeros((n_dark, n_pix_sq), dtype=np.float64)
    weights = _validated_wavelength_weights(
        wavelength_weights, len(unitary_wavelength_pairs)
    )
    sub_diameters = np.asarray(
        [sub.diameter_m for sub in subapertures],
        dtype=np.float64,
    )

    for weight, (unitary, wavelength) in zip(weights, unitary_wavelength_pairs):
        if roll_angle_rad is None:
            u_sub, v_sub = subapertures_to_uv(
                subapertures,
                hour_angle_rad,
                declination_rad,
                wavelength,
                latitude_rad,
            )
        else:
            u_sub, v_sub = _rolled_subapertures_to_uv(
                subapertures,
                roll_angle_rad,
                wavelength,
            )
        beams = compute_beam_patterns(
            u_sub=u_sub,
            v_sub=v_sub,
            amplitudes=amplitudes,
            unitary=unitary,
            port_indices=dark_port_indices,
            l_coords=l_coords,
            m_coords=m_coords,
            sub_diameters_m=sub_diameters,
            wavelength_m=wavelength,
        )
        beams_flat += weight * beams.reshape(n_dark, n_pix_sq)

    return beams_flat


def _validated_wavelength_weights(
    wavelength_weights: NDArray[np.float64] | None,
    n_wavelengths: int,
) -> NDArray[np.float64]:
    if wavelength_weights is None:
        return np.full(int(n_wavelengths), 1.0 / float(n_wavelengths))
    weights = np.asarray(wavelength_weights, dtype=np.float64)
    if weights.shape != (int(n_wavelengths),):
        raise ValueError(
            "wavelength_weights must have one value per wavelength "
            f"({weights.shape} != ({int(n_wavelengths)},))."
        )
    return weights


def _band_weights(rates_by_wavelength: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize per-bin stellar rates to band weights; uniform for a dark star."""

    rates = np.asarray(rates_by_wavelength, dtype=np.float64)
    total = float(np.sum(rates))
    if total > 0.0:
        return rates / total
    return np.full(rates.size, 1.0 / float(rates.size))


def _wavelength_resolved_beams_flat(
    *,
    subapertures: Sequence[SubAperture],
    amplitudes: NDArray[np.complex128],
    dark_port_indices: NDArray[np.intp],
    hour_angle_rad: float,
    declination_rad: float,
    roll_angle_rad: float | None,
    l_coords: NDArray[np.float64],
    m_coords: NDArray[np.float64],
    latitude_rad: float,
    unitary_wavelength_pairs: Sequence[tuple[NDArray[np.complex128], float]],
) -> NDArray[np.float64]:
    n_dark = int(len(dark_port_indices))
    n_pix_sq = int(len(l_coords) * len(m_coords))
    beams_flat = np.zeros(
        (len(unitary_wavelength_pairs) * n_dark, n_pix_sq),
        dtype=np.float64,
    )
    sub_diameters = np.asarray(
        [sub.diameter_m for sub in subapertures],
        dtype=np.float64,
    )

    for wl_index, (unitary, wavelength) in enumerate(unitary_wavelength_pairs):
        if roll_angle_rad is None:
            u_sub, v_sub = subapertures_to_uv(
                subapertures,
                hour_angle_rad,
                declination_rad,
                wavelength,
                latitude_rad,
            )
        else:
            u_sub, v_sub = _rolled_subapertures_to_uv(
                subapertures,
                roll_angle_rad,
                wavelength,
            )
        beams = compute_beam_patterns(
            u_sub=u_sub,
            v_sub=v_sub,
            amplitudes=amplitudes,
            unitary=unitary,
            port_indices=dark_port_indices,
            l_coords=l_coords,
            m_coords=m_coords,
            sub_diameters_m=sub_diameters,
            wavelength_m=wavelength,
        )
        start = wl_index * n_dark
        beams_flat[start : start + n_dark, :] = beams.reshape(n_dark, n_pix_sq)

    return beams_flat


def _rolled_subapertures_to_uv(
    subapertures: Sequence[SubAperture],
    roll_angle_rad: float,
    wavelength_m: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Project pupil coordinates after spacecraft line-of-sight roll.

    Positive roll is an active counter-clockwise rotation of the pupil-frame
    ``(x, y)`` coordinates into sky east/north coordinates.
    """

    positions = np.asarray(
        [np.asarray(sub.position_enu_m, dtype=np.float64) for sub in subapertures],
        dtype=np.float64,
    )
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("sub-aperture positions must have shape (M, 3).")
    c = np.cos(float(roll_angle_rad))
    s = np.sin(float(roll_angle_rad))
    x = positions[:, 0]
    y = positions[:, 1]
    u_sub = (c * x - s * y) / float(wavelength_m)
    v_sub = (s * x + c * y) / float(wavelength_m)
    return u_sub.astype(np.float64), v_sub.astype(np.float64)


def _validated_roll_angles(
    roll_angles_rad: NDArray[np.float64] | None,
    n_time: int,
) -> NDArray[np.float64] | None:
    if roll_angles_rad is None:
        return None
    values = np.asarray(roll_angles_rad, dtype=np.float64)
    if values.shape != (int(n_time),):
        raise ValueError(
            "roll_angles_rad must have one value per time sample "
            f"({values.shape} != ({int(n_time)},))."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("roll_angles_rad must contain only finite values.")
    return values


def _stellar_leakage_counts_per_wavelength(
    *,
    amplitudes: NDArray[np.complex128],
    rows_by_wavelength: Sequence[NDArray[np.complex128]],
    null_depth: float,
    stellar_photon_rate_hz_by_wavelength: NDArray[np.float64],
    integration_time_per_step_s: float,
) -> NDArray[np.float64]:
    """Distribute the realized null over the dark rows, per wavelength.

    ``null_depth`` is the band dark-power fraction ``L / F_total`` the
    upstream budget realized (static chip residual plus piston, amplitude, and
    other random errors). It is split into two parts:

    * the nominal chip's static coherent residual ``|sum_m U_dm a_m|^2 / A``,
      which lands in the ports it lands in;
    * the excess ``null - static``, distributed in proportion to
      ``sum_m |U_dm|^2 |a_m|^2``, the IID-piston leakage shape.

    For independent, identically distributed input phase errors with
    characteristic function ``chi = <exp(i phi)>``, the cross terms of
    ``<|sum_m U_dm a_m exp(i phi_m)|^2>`` average to ``|chi|^2``, so the mean
    port power is ``|chi|^2 |sum_m U_dm a_m|^2 + (1 - |chi|^2) sum_m
    |U_dm|^2 |a_m|^2``. The split here is that result with ``|chi|^2 -> 1``
    on the static term, which is exact to first order in the phase variance.
    The shape no longer depends on the nominal chip's coherent residual,
    which a deep null makes arbitrary (a 1e-10 rad rotation can move all of
    it into one port).

    Both parts are weighted by the per-bin stellar rate, so the band total is
    ``null * sum(F_lambda)``. If the static residual alone exceeds the
    realized null (a calibrated null below the chip floor), the static pattern
    is scaled down to the realized total.

    ``rows_by_wavelength`` holds one ``(n_rows, M)`` output-row matrix per
    wavelength; the result has shape ``(n_wavelength, n_rows)`` in counts.
    """

    a = np.asarray(amplitudes, dtype=np.complex128)
    area = _positive_total_amplitude_power(a)
    rows = np.stack([np.asarray(r, dtype=np.complex128) for r in rows_by_wavelength])
    rates = np.asarray(stellar_photon_rate_hz_by_wavelength, dtype=np.float64)
    if rates.shape != (rows.shape[0],):
        raise ValueError(
            "stellar_photon_rate_hz_by_wavelength must have one value per "
            f"wavelength ({rates.shape} != ({rows.shape[0]},))."
        )
    static = rates[:, None] * np.abs(rows @ a) ** 2 / area
    incoherent = rates[:, None] * ((np.abs(rows) ** 2) @ (np.abs(a) ** 2)) / area
    target = float(null_depth) * float(np.sum(rates))
    static_total = float(np.sum(static))
    if static_total >= target:
        leakage = static * (target / static_total) if static_total > 0.0 else static
    else:
        if float(np.sum(incoherent)) <= 0.0:
            # Rows uncoupled to every input: spread the excess uniformly.
            incoherent = np.repeat(rates[:, None], rows.shape[1], axis=1)
        leakage = static + (target - static_total) * incoherent / np.sum(incoherent)
    return leakage * float(integration_time_per_step_s)


def _time_tiled_leakage(
    *,
    leakage_per_port: NDArray[np.float64],
    n_time: int,
    null_instability: float,
    ft_servo_bandwidth_hz: float,
    integration_time_per_step_s: float,
    rng: np.random.Generator,
    leakage_scales: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    if leakage_scales is None:
        scales = _leakage_time_scales(
            n_time=n_time,
            null_instability=null_instability,
            ft_servo_bandwidth_hz=ft_servo_bandwidth_hz,
            integration_time_per_step_s=integration_time_per_step_s,
            rng=rng,
        )
    else:
        scales = np.asarray(leakage_scales, dtype=np.float64)
        if scales.shape != (int(n_time),):
            raise ValueError("leakage_scales must have one value per time step.")
    return np.repeat(scales, len(leakage_per_port)) * np.tile(
        leakage_per_port,
        n_time,
    )


def _leakage_time_scales(
    *,
    n_time: int,
    null_instability: float,
    ft_servo_bandwidth_hz: float,
    integration_time_per_step_s: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    if null_instability <= 0.0:
        return np.ones(int(n_time), dtype=np.float64)
    if ft_servo_bandwidth_hz > 0.0 and integration_time_per_step_s > 0.0:
        rms = null_instability / np.sqrt(
            2.0 * ft_servo_bandwidth_hz * integration_time_per_step_s
        )
    else:
        rms = null_instability
    return np.maximum(1.0 + rng.normal(0.0, rms, size=int(n_time)), 0.0)


def _positive_total_amplitude_power(amplitudes: NDArray[np.complex128]) -> float:
    total = float(np.sum(np.abs(np.asarray(amplitudes, dtype=np.complex128)) ** 2))
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("amplitudes must have positive finite total power.")
    return total


def _rates_by_wavelength(
    *,
    total_rate_hz: float,
    rates_by_wavelength: NDArray[np.float64] | None,
    n_wavelengths: int,
    name: str,
) -> NDArray[np.float64]:
    if total_rate_hz < 0.0 or not np.isfinite(total_rate_hz):
        raise ValueError("total_rate_hz must be finite and non-negative.")
    if rates_by_wavelength is None:
        return np.full(
            int(n_wavelengths),
            float(total_rate_hz) / float(n_wavelengths),
            dtype=np.float64,
        )
    values = np.asarray(rates_by_wavelength, dtype=np.float64)
    if values.shape != (int(n_wavelengths),):
        raise ValueError(
            f"{name} must have one value per wavelength "
            f"({values.shape} != ({int(n_wavelengths)},))."
        )
    if np.any(values < 0.0) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} entries must be finite and non-negative.")
    if total_rate_hz == 0.0:
        return np.zeros(int(n_wavelengths), dtype=np.float64)
    total = float(np.sum(values))
    if total_rate_hz > 0.0 and total > 0.0:
        return values * (float(total_rate_hz) / total)
    return values


__all__ = [
    "MAS_PER_RAD",
    "MAUNAKEA_LAT_RAD",
    "ObservationData",
    "ReconstructedImage",
    "SkyScene",
    "SubAperture",
    "build_response_matrix",
    "build_wavelength_resolved_response_matrix",
    "compute_beam_patterns",
    "compute_synthesized_beam",
    "create_scene",
    "phase_chop_unitary_pair",
    "reconstruct_nnls",
    "reconstruct_snr_map",
    "simulate_observation",
    "simulate_phase_chopped_observation",
    "subapertures_to_uv",
]
