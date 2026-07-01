"""Fringe-tracker piston observability primitives.

The functions here test whether a deployed photonic-chip unitary maps
differential aperture pistons into independent bright-port intensity
measurements. They deliberately operate on arrays only; pipeline stages own
TOML parsing, detector labels, and result payloads.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

_TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class IntensityResponse:
    """Linear bright-port intensity response to per-mode piston offsets."""

    response_power_per_m: NDArray[np.float64]
    reference_power: NDArray[np.float64]
    row_port_indices: NDArray[np.intp]
    row_wavelength_indices: NDArray[np.intp]


@dataclass(frozen=True)
class SpectralChannelCountResponse:
    """Count response for wavelength bins summed into tracker channels."""

    response_counts_per_m: NDArray[np.float64]
    reference_power: NDArray[np.float64]
    expected_counts: NDArray[np.float64]
    row_port_indices: NDArray[np.intp]
    row_channel_indices: NDArray[np.intp]
    row_wavelength_indices: NDArray[np.intp]


@dataclass(frozen=True)
class FractionalIntensityResponse:
    """Linear bright-port response to per-mode fractional intensity changes."""

    response_power_per_fractional_intensity: NDArray[np.float64]
    reference_power: NDArray[np.float64]
    row_port_indices: NDArray[np.intp]
    row_wavelength_indices: NDArray[np.intp]


@dataclass(frozen=True)
class SpectralChannelFractionalIntensityResponse:
    """Count response to fractional intensity changes in tracker channels."""

    response_counts_per_fractional_intensity: NDArray[np.float64]
    reference_power: NDArray[np.float64]
    expected_counts: NDArray[np.float64]
    row_port_indices: NDArray[np.intp]
    row_channel_indices: NDArray[np.intp]
    row_wavelength_indices: NDArray[np.intp]


@dataclass(frozen=True)
class ObservabilitySVD:
    """SVD diagnostics for a whitened piston response matrix."""

    singular_values: NDArray[np.float64]
    right_singular_vectors: NDArray[np.float64]
    tolerance: float
    rank: int
    full_rank: bool
    observable_fraction: float
    condition_number: float
    truncated_condition_number: float


@dataclass(frozen=True)
class CramerRaoBound:
    """CRLB implied by a whitened piston response matrix."""

    singular_mode_std_m: NDArray[np.float64]
    piston_rms_m: float
    coordinate_rms_m: float
    truncated_piston_rms_m: float
    truncated_coordinate_rms_m: float
    finite: bool


@dataclass(frozen=True)
class PistonMeasurementCovariance:
    """Full-rank piston covariance implied by a whitened response matrix."""

    coordinate_covariance_m2: NDArray[np.float64]
    piston_covariance_m2: NDArray[np.float64]
    piston_rms_m: float
    coordinate_rms_m: float
    finite: bool


@dataclass(frozen=True)
class ReconstructionMonteCarlo:
    """Monte Carlo reconstruction diagnostics for the linear estimator."""

    input_rms_m: NDArray[np.float64]
    recovered_rms_m: NDArray[np.float64]
    residual_rms_m: NDArray[np.float64]
    median_residual_rms_m: float
    p95_residual_rms_m: float
    median_recovered_rms_m: float


@dataclass(frozen=True)
class AmplitudePostprocessingMonteCarlo:
    """Monte Carlo diagnostics for differential-amplitude post-processing."""

    input_rms_fractional_intensity: NDArray[np.float64]
    bright_linear_residual_rms_fractional_intensity: NDArray[np.float64]
    bright_wiener_residual_rms_fractional_intensity: NDArray[np.float64]
    ideal_tap_residual_rms_fractional_intensity: NDArray[np.float64]
    median_input_rms_fractional_intensity: float
    median_bright_linear_residual_rms_fractional_intensity: float
    p95_bright_linear_residual_rms_fractional_intensity: float
    median_bright_wiener_residual_rms_fractional_intensity: float
    p95_bright_wiener_residual_rms_fractional_intensity: float
    median_ideal_tap_residual_rms_fractional_intensity: float
    p95_ideal_tap_residual_rms_fractional_intensity: float
    ideal_tap_counts_per_mode: float


@dataclass(frozen=True)
class GaussNewtonMonteCarlo:
    """Monte Carlo reconstruction diagnostics for a nonlinear estimator."""

    input_rms_m: NDArray[np.float64]
    recovered_rms_m: NDArray[np.float64]
    residual_rms_m: NDArray[np.float64]
    median_residual_rms_m: float
    p95_residual_rms_m: float
    median_recovered_rms_m: float
    converged: NDArray[np.bool_]
    iterations: NDArray[np.int64]
    final_weighted_cost: NDArray[np.float64]
    convergence_fraction: float
    max_iter_fraction: float
    worse_than_input_fraction: float


@dataclass(frozen=True)
class ActiveAcquisitionMonteCarlo:
    """Monte Carlo diagnostics for an active coordinate-scan lock acquisition."""

    input_rms_m: NDArray[np.float64]
    recovered_rms_m: NDArray[np.float64]
    residual_rms_m: NDArray[np.float64]
    median_residual_rms_m: float
    p95_residual_rms_m: float
    median_recovered_rms_m: float
    locked: NDArray[np.bool_]
    evaluations: NDArray[np.int64]
    final_weighted_cost: NDArray[np.float64]
    lock_fraction: float
    worse_than_input_fraction: float


@dataclass(frozen=True)
class CophasingHandoffMonteCarlo:
    """Monte Carlo diagnostics for cophasing-unitary acquisition and handoff."""

    input_rms_m: NDArray[np.float64]
    acquisition_residual_rms_m: NDArray[np.float64]
    science_input_rms_m: NDArray[np.float64]
    science_residual_rms_m: NDArray[np.float64]
    median_acquisition_residual_rms_m: float
    p95_acquisition_residual_rms_m: float
    median_science_residual_rms_m: float
    p95_science_residual_rms_m: float
    median_science_input_rms_m: float
    p95_science_input_rms_m: float
    acquisition_locked: NDArray[np.bool_]
    science_locked: NDArray[np.bool_]
    acquisition_lock_fraction: float
    science_lock_fraction: float
    edges: NDArray[np.int64]
    edge_wrap_fraction: float
    max_edge_opd_over_lambda: float


def linearized_reconstruction_monte_carlo(
    *,
    whitened_response_per_m: NDArray[np.floating],
    piston_basis: NDArray[np.floating],
    piston_rms_m: float,
    n_trials: int,
    rng: np.random.Generator,
    relative_threshold: float = 1.0e-6,
) -> ReconstructionMonteCarlo:
    """Reconstruct random pistons from the linearized Poisson model.

    The measurement model is ``delta = A @ coords + N(0, I)`` where ``A`` is
    the whitened count response. This reports the small-signal Fisher floor
    implied by the response matrix and should track the CRLB. Nonlinear
    capture-range effects are intentionally excluded.
    """

    if n_trials < 0:
        raise ValueError("n_trials must be non-negative.")
    if piston_rms_m < 0.0:
        raise ValueError("piston_rms_m must be non-negative.")
    if n_trials == 0:
        empty = np.zeros(0, dtype=np.float64)
        return ReconstructionMonteCarlo(
            empty, empty, empty, float("nan"), float("nan"), float("nan")
        )

    basis = np.asarray(piston_basis, dtype=np.float64)
    whitened = np.asarray(whitened_response_per_m, dtype=np.float64)
    if basis.ndim != 2:
        raise ValueError("piston_basis must be a 2-D matrix.")
    if whitened.ndim != 2 or whitened.shape[1] != basis.shape[1]:
        raise ValueError("whitened_response_per_m shape is inconsistent.")

    estimator = np.linalg.pinv(whitened, rcond=float(relative_threshold))
    n_modes = int(basis.shape[0])
    n_dof = int(basis.shape[1])
    coord_sigma = (
        float(piston_rms_m) * np.sqrt(float(n_modes) / float(n_dof))
        if n_dof > 0
        else 0.0
    )

    input_rms = np.zeros(n_trials, dtype=np.float64)
    recovered_rms = np.zeros(n_trials, dtype=np.float64)
    residual_rms = np.zeros(n_trials, dtype=np.float64)
    for i in range(n_trials):
        coords_true = rng.normal(0.0, coord_sigma, size=n_dof)
        pistons_true = basis @ coords_true
        unit_noise = rng.normal(0.0, 1.0, size=whitened.shape[0])
        coords_hat = estimator @ (whitened @ coords_true + unit_noise)
        pistons_hat = basis @ coords_hat
        residual = pistons_true - pistons_hat
        input_rms[i] = _rms(pistons_true)
        recovered_rms[i] = _rms(pistons_hat)
        residual_rms[i] = _rms(residual)

    return ReconstructionMonteCarlo(
        input_rms_m=input_rms,
        recovered_rms_m=recovered_rms,
        residual_rms_m=residual_rms,
        median_residual_rms_m=float(np.median(residual_rms)),
        p95_residual_rms_m=float(np.percentile(residual_rms, 95.0)),
        median_recovered_rms_m=float(np.median(recovered_rms)),
    )


def amplitude_postprocessing_monte_carlo(
    *,
    whitened_response_per_fractional_intensity: NDArray[np.floating],
    amplitude_basis: NDArray[np.floating],
    scintillation_sigma_i: float,
    ideal_tap_counts_per_mode: float,
    n_trials: int,
    rng: np.random.Generator,
    relative_threshold: float = 1.0e-6,
) -> AmplitudePostprocessingMonteCarlo:
    """Compare amplitude post-processing with current and ideal measurements.

    ``scintillation_sigma_i`` is the per-physical-aperture RMS fractional
    intensity jitter injected by the realized-null model. The current-unitary
    estimators use the supplied bright-port response: an unconstrained
    least-squares inverse and a Gaussian-prior Wiener estimator. The ideal
    photometric-tap reference assumes one direct intensity measurement per
    physical aperture with ``ideal_tap_counts_per_mode`` photons per sample.
    """

    if n_trials < 0:
        raise ValueError("n_trials must be non-negative.")
    if scintillation_sigma_i < 0.0:
        raise ValueError("scintillation_sigma_i must be non-negative.")
    if ideal_tap_counts_per_mode < 0.0:
        raise ValueError("ideal_tap_counts_per_mode must be non-negative.")
    if relative_threshold <= 0.0:
        raise ValueError("relative_threshold must be positive.")

    basis = np.asarray(amplitude_basis, dtype=np.float64)
    whitened = np.asarray(
        whitened_response_per_fractional_intensity,
        dtype=np.float64,
    )
    if basis.ndim != 2:
        raise ValueError("amplitude_basis must be a 2-D matrix.")
    if whitened.ndim != 2 or whitened.shape[1] != basis.shape[1]:
        raise ValueError(
            "whitened_response_per_fractional_intensity shape is inconsistent."
        )
    if n_trials == 0:
        empty = np.zeros(0, dtype=np.float64)
        return AmplitudePostprocessingMonteCarlo(
            empty,
            empty,
            empty,
            empty,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float(ideal_tap_counts_per_mode),
        )

    n_modes = int(basis.shape[0])
    n_dof = int(basis.shape[1])
    sigma = float(scintillation_sigma_i)
    linear_estimator = np.linalg.pinv(whitened, rcond=float(relative_threshold))
    if sigma > 0.0:
        prior_precision = 1.0 / (sigma * sigma)
        lhs = whitened.T @ whitened + prior_precision * np.eye(n_dof)
        wiener_estimator = np.linalg.solve(lhs, whitened.T)
    else:
        wiener_estimator = np.zeros((n_dof, whitened.shape[0]), dtype=np.float64)

    tap_noise_sigma = (
        1.0 / np.sqrt(float(ideal_tap_counts_per_mode))
        if ideal_tap_counts_per_mode > 0.0
        else float("inf")
    )
    input_rms = np.zeros(n_trials, dtype=np.float64)
    bright_linear_residual = np.zeros(n_trials, dtype=np.float64)
    bright_wiener_residual = np.zeros(n_trials, dtype=np.float64)
    ideal_tap_residual = np.zeros(n_trials, dtype=np.float64)

    for i in range(n_trials):
        physical_true = rng.normal(0.0, sigma, size=n_modes)
        coords_true = basis.T @ physical_true
        differential_true = basis @ coords_true
        unit_noise = rng.normal(0.0, 1.0, size=whitened.shape[0])
        whitened_measurement = whitened @ coords_true + unit_noise

        coords_linear = linear_estimator @ whitened_measurement
        coords_wiener = wiener_estimator @ whitened_measurement
        if np.isfinite(tap_noise_sigma):
            tap_observed = physical_true + rng.normal(
                0.0,
                tap_noise_sigma,
                size=n_modes,
            )
            coords_tap = basis.T @ tap_observed
        else:
            coords_tap = np.zeros(n_dof, dtype=np.float64)

        input_rms[i] = _rms(differential_true)
        bright_linear_residual[i] = _rms(differential_true - basis @ coords_linear)
        bright_wiener_residual[i] = _rms(differential_true - basis @ coords_wiener)
        ideal_tap_residual[i] = _rms(differential_true - basis @ coords_tap)

    return AmplitudePostprocessingMonteCarlo(
        input_rms_fractional_intensity=input_rms,
        bright_linear_residual_rms_fractional_intensity=bright_linear_residual,
        bright_wiener_residual_rms_fractional_intensity=bright_wiener_residual,
        ideal_tap_residual_rms_fractional_intensity=ideal_tap_residual,
        median_input_rms_fractional_intensity=float(np.median(input_rms)),
        median_bright_linear_residual_rms_fractional_intensity=float(
            np.median(bright_linear_residual)
        ),
        p95_bright_linear_residual_rms_fractional_intensity=float(
            np.percentile(bright_linear_residual, 95.0)
        ),
        median_bright_wiener_residual_rms_fractional_intensity=float(
            np.median(bright_wiener_residual)
        ),
        p95_bright_wiener_residual_rms_fractional_intensity=float(
            np.percentile(bright_wiener_residual, 95.0)
        ),
        median_ideal_tap_residual_rms_fractional_intensity=float(
            np.median(ideal_tap_residual)
        ),
        p95_ideal_tap_residual_rms_fractional_intensity=float(
            np.percentile(ideal_tap_residual, 95.0)
        ),
        ideal_tap_counts_per_mode=float(ideal_tap_counts_per_mode),
    )


def differential_piston_basis(n_modes: int) -> NDArray[np.float64]:
    """Return an orthonormal basis for the sum-zero piston subspace.

    The returned matrix has shape ``(n_modes, n_modes - 1)``. Multiplying it by
    coordinate vectors produces per-mode pistons with the unobservable common
    piston removed.
    """

    if n_modes < 2:
        raise ValueError("differential_piston_basis requires n_modes >= 2.")
    basis = np.zeros((n_modes, n_modes - 1), dtype=np.float64)
    for j in range(n_modes - 1):
        scale = np.sqrt(float((j + 1) * (j + 2)))
        basis[: j + 1, j] = 1.0 / scale
        basis[j + 1, j] = -float(j + 1) / scale
    return basis


def intensity_response_matrix(
    *,
    u_stack: NDArray[np.complexfloating],
    input_fields: NDArray[np.complexfloating],
    wavelengths_m: NDArray[np.floating],
    port_indices: NDArray[np.integer],
    wavelength_indices: NDArray[np.integer] | None = None,
) -> IntensityResponse:
    """Return ``d |U field|^2 / d piston`` for selected ports and wavelengths.

    Args:
        u_stack: Complex chip stack with shape ``(n_ports, n_modes, n_wl)``.
        input_fields: Chip-input stellar field with shape
            ``(n_modes, n_wl, n_pol)``. A 2-D ``(n_modes, n_wl)`` array is
            accepted and treated as one polarization channel.
        wavelengths_m: Wavelength grid in meters, length ``n_wl``.
        port_indices: Output-port indices to use as measurements.
        wavelength_indices: Optional wavelength-bin subset. Rows are ordered by
            wavelength, then port.

    Returns:
        Linear intensity response and the corresponding reference powers.
    """

    u_arr = _as_unitary_stack(u_stack)
    fields = _as_input_fields(input_fields, u_arr.shape[1], u_arr.shape[2])
    wavelengths = _as_wavelengths(wavelengths_m, u_arr.shape[2])
    ports = _as_indices(port_indices, upper=u_arr.shape[0], name="port_indices")
    if wavelength_indices is None:
        wave_idx = np.arange(u_arr.shape[2], dtype=np.intp)
    else:
        wave_idx = _as_indices(
            wavelength_indices, upper=u_arr.shape[2], name="wavelength_indices"
        )
    if ports.size == 0:
        raise ValueError("intensity_response_matrix requires at least one port.")
    if wave_idx.size == 0:
        raise ValueError(
            "intensity_response_matrix requires at least one wavelength bin."
        )

    n_rows = int(ports.size * wave_idx.size)
    response = np.zeros((n_rows, u_arr.shape[1]), dtype=np.float64)
    reference_power = np.zeros(n_rows, dtype=np.float64)
    row_ports = np.zeros(n_rows, dtype=np.intp)
    row_waves = np.zeros(n_rows, dtype=np.intp)

    row0 = 0
    for wave in wave_idx:
        u_ports = u_arr[ports, :, wave]
        field_w = fields[:, wave, :]
        electric = u_ports @ field_w
        d_electric_dp = (
            u_ports[:, :, np.newaxis]
            * field_w[np.newaxis, :, :]
            * (1j * _TWO_PI / wavelengths[wave])
        )
        derivative = 2.0 * np.real(
            np.sum(np.conj(electric)[:, np.newaxis, :] * d_electric_dp, axis=2)
        )
        rows = slice(row0, row0 + ports.size)
        response[rows, :] = derivative
        reference_power[rows] = np.sum(np.abs(electric) ** 2, axis=1)
        row_ports[rows] = ports
        row_waves[rows] = wave
        row0 += int(ports.size)

    return IntensityResponse(
        response_power_per_m=response,
        reference_power=reference_power,
        row_port_indices=row_ports,
        row_wavelength_indices=row_waves,
    )


def spectral_channel_count_response_matrix(
    *,
    u_stack: NDArray[np.complexfloating],
    input_fields: NDArray[np.complexfloating],
    wavelengths_m: NDArray[np.floating],
    port_indices: NDArray[np.integer],
    wavelength_groups: tuple[NDArray[np.integer], ...],
    expected_counts_by_wavelength: NDArray[np.floating],
) -> SpectralChannelCountResponse:
    """Return count response for incoherently summed spectral count channels.

    Each row corresponds to one ``(channel, port)`` measurement. Within a
    channel, wavelength-bin powers and derivatives are converted to detector
    counts using the per-wavelength count budget and then summed. This models
    finite tracker count channels rather than treating the center wavelength as
    a proxy for the whole channel.
    """

    u_arr = _as_unitary_stack(u_stack)
    fields = _as_input_fields(input_fields, u_arr.shape[1], u_arr.shape[2])
    wavelengths = _as_wavelengths(wavelengths_m, u_arr.shape[2])
    ports = _as_indices(port_indices, upper=u_arr.shape[0], name="port_indices")
    groups = _as_wavelength_groups(wavelength_groups, n_wavelengths=u_arr.shape[2])
    counts_by_wave = _as_counts_by_wavelength(
        expected_counts_by_wavelength,
        n_ports=ports.size,
        n_wavelengths=u_arr.shape[2],
    )
    counts_per_power = _counts_per_power_by_wavelength(
        u_stack=u_arr,
        input_fields=fields,
        wavelengths_m=wavelengths,
        port_indices=ports,
        expected_counts_by_wavelength=counts_by_wave,
    )

    n_rows = int(ports.size * len(groups))
    response_counts = np.zeros((n_rows, u_arr.shape[1]), dtype=np.float64)
    reference_power = np.zeros(n_rows, dtype=np.float64)
    expected_counts = np.zeros(n_rows, dtype=np.float64)
    row_ports = np.zeros(n_rows, dtype=np.intp)
    row_channels = np.zeros(n_rows, dtype=np.intp)
    row_waves = np.zeros(n_rows, dtype=np.intp)

    row0 = 0
    for channel_i, group in enumerate(groups):
        response = intensity_response_matrix(
            u_stack=u_arr,
            input_fields=fields,
            wavelengths_m=wavelengths,
            port_indices=ports,
            wavelength_indices=group,
        )
        rows = slice(row0, row0 + ports.size)
        for local_wave_i, wave in enumerate(group):
            wave_rows = slice(
                local_wave_i * ports.size,
                (local_wave_i + 1) * ports.size,
            )
            scale = counts_per_power[:, int(wave)]
            response_counts[rows] += (
                response.response_power_per_m[wave_rows] * scale[:, np.newaxis]
            )
            reference_power[rows] += response.reference_power[wave_rows]
            expected_counts[rows] += counts_by_wave[:, int(wave)]
        representative = int(group[(group.size - 1) // 2])
        row_ports[rows] = ports
        row_channels[rows] = int(channel_i)
        row_waves[rows] = representative
        row0 += int(ports.size)

    return SpectralChannelCountResponse(
        response_counts_per_m=response_counts,
        reference_power=reference_power,
        expected_counts=expected_counts,
        row_port_indices=row_ports,
        row_channel_indices=row_channels,
        row_wavelength_indices=row_waves,
    )


def fractional_intensity_response_matrix(
    *,
    u_stack: NDArray[np.complexfloating],
    input_fields: NDArray[np.complexfloating],
    port_indices: NDArray[np.integer],
    wavelength_indices: NDArray[np.integer] | None = None,
) -> FractionalIntensityResponse:
    """Return ``d |U field|^2 / d fractional input intensity``.

    The perturbation coordinate is the small fractional intensity change
    ``delta_I`` for each input mode. The corresponding complex field factor is
    ``sqrt(1 + delta_I) ~= 1 + 0.5 * delta_I``. A common fractional-intensity
    perturbation therefore changes every output power by its reference power,
    while differential perturbations diagnose scintillation modes that leak
    through a fixed null.
    """

    u_arr = _as_unitary_stack(u_stack)
    fields = _as_input_fields(input_fields, u_arr.shape[1], u_arr.shape[2])
    ports = _as_indices(port_indices, upper=u_arr.shape[0], name="port_indices")
    if wavelength_indices is None:
        wave_idx = np.arange(u_arr.shape[2], dtype=np.intp)
    else:
        wave_idx = _as_indices(
            wavelength_indices, upper=u_arr.shape[2], name="wavelength_indices"
        )
    if ports.size == 0:
        raise ValueError(
            "fractional_intensity_response_matrix requires at least one port."
        )
    if wave_idx.size == 0:
        raise ValueError(
            "fractional_intensity_response_matrix requires at least one wavelength bin."
        )

    n_rows = int(ports.size * wave_idx.size)
    response = np.zeros((n_rows, u_arr.shape[1]), dtype=np.float64)
    reference_power = np.zeros(n_rows, dtype=np.float64)
    row_ports = np.zeros(n_rows, dtype=np.intp)
    row_waves = np.zeros(n_rows, dtype=np.intp)

    row0 = 0
    for wave in wave_idx:
        u_ports = u_arr[ports, :, wave]
        field_w = fields[:, wave, :]
        electric = u_ports @ field_w
        d_electric_di = 0.5 * u_ports[:, :, np.newaxis] * field_w[np.newaxis, :, :]
        derivative = 2.0 * np.real(
            np.sum(np.conj(electric)[:, np.newaxis, :] * d_electric_di, axis=2)
        )
        rows = slice(row0, row0 + ports.size)
        response[rows, :] = derivative
        reference_power[rows] = np.sum(np.abs(electric) ** 2, axis=1)
        row_ports[rows] = ports
        row_waves[rows] = wave
        row0 += int(ports.size)

    return FractionalIntensityResponse(
        response_power_per_fractional_intensity=response,
        reference_power=reference_power,
        row_port_indices=row_ports,
        row_wavelength_indices=row_waves,
    )


def spectral_channel_fractional_intensity_response_matrix(
    *,
    u_stack: NDArray[np.complexfloating],
    input_fields: NDArray[np.complexfloating],
    wavelengths_m: NDArray[np.floating],
    port_indices: NDArray[np.integer],
    wavelength_groups: tuple[NDArray[np.integer], ...],
    expected_counts_by_wavelength: NDArray[np.floating],
) -> SpectralChannelFractionalIntensityResponse:
    """Return count response to fractional intensity perturbations.

    Each row corresponds to one ``(channel, port)`` measurement. Within a
    channel, wavelength-bin powers and derivatives are converted to detector
    counts using the per-wavelength count budget and then summed.
    """

    u_arr = _as_unitary_stack(u_stack)
    fields = _as_input_fields(input_fields, u_arr.shape[1], u_arr.shape[2])
    wavelengths = _as_wavelengths(wavelengths_m, u_arr.shape[2])
    ports = _as_indices(port_indices, upper=u_arr.shape[0], name="port_indices")
    groups = _as_wavelength_groups(wavelength_groups, n_wavelengths=u_arr.shape[2])
    counts_by_wave = _as_counts_by_wavelength(
        expected_counts_by_wavelength,
        n_ports=ports.size,
        n_wavelengths=u_arr.shape[2],
    )
    counts_per_power = _counts_per_power_by_wavelength(
        u_stack=u_arr,
        input_fields=fields,
        wavelengths_m=wavelengths,
        port_indices=ports,
        expected_counts_by_wavelength=counts_by_wave,
    )

    n_rows = int(ports.size * len(groups))
    response_counts = np.zeros((n_rows, u_arr.shape[1]), dtype=np.float64)
    reference_power = np.zeros(n_rows, dtype=np.float64)
    expected_counts = np.zeros(n_rows, dtype=np.float64)
    row_ports = np.zeros(n_rows, dtype=np.intp)
    row_channels = np.zeros(n_rows, dtype=np.intp)
    row_waves = np.zeros(n_rows, dtype=np.intp)

    row0 = 0
    for channel_i, group in enumerate(groups):
        response = fractional_intensity_response_matrix(
            u_stack=u_arr,
            input_fields=fields,
            port_indices=ports,
            wavelength_indices=group,
        )
        rows = slice(row0, row0 + ports.size)
        for local_wave_i, wave in enumerate(group):
            wave_rows = slice(
                local_wave_i * ports.size,
                (local_wave_i + 1) * ports.size,
            )
            scale = counts_per_power[:, int(wave)]
            response_counts[rows] += (
                response.response_power_per_fractional_intensity[wave_rows]
                * scale[:, np.newaxis]
            )
            reference_power[rows] += response.reference_power[wave_rows]
            expected_counts[rows] += counts_by_wave[:, int(wave)]
        representative = int(group[(group.size - 1) // 2])
        row_ports[rows] = ports
        row_channels[rows] = int(channel_i)
        row_waves[rows] = representative
        row0 += int(ports.size)

    return SpectralChannelFractionalIntensityResponse(
        response_counts_per_fractional_intensity=response_counts,
        reference_power=reference_power,
        expected_counts=expected_counts,
        row_port_indices=row_ports,
        row_channel_indices=row_channels,
        row_wavelength_indices=row_waves,
    )


def port_powers_for_pistons(
    *,
    u_stack: NDArray[np.complexfloating],
    input_fields: NDArray[np.complexfloating],
    wavelengths_m: NDArray[np.floating],
    port_indices: NDArray[np.integer],
    pistons_m: NDArray[np.floating],
    wavelength_indices: NDArray[np.integer] | None = None,
) -> NDArray[np.float64]:
    """Evaluate selected bright-port powers for arbitrary piston offsets."""

    u_arr = _as_unitary_stack(u_stack)
    fields = _as_input_fields(input_fields, u_arr.shape[1], u_arr.shape[2])
    wavelengths = _as_wavelengths(wavelengths_m, u_arr.shape[2])
    ports = _as_indices(port_indices, upper=u_arr.shape[0], name="port_indices")
    pistons = np.asarray(pistons_m, dtype=np.float64)
    if pistons.shape != (u_arr.shape[1],):
        raise ValueError(
            f"pistons_m must have shape ({u_arr.shape[1]},), got {pistons.shape}."
        )
    if wavelength_indices is None:
        wave_idx = np.arange(u_arr.shape[2], dtype=np.intp)
    else:
        wave_idx = _as_indices(
            wavelength_indices, upper=u_arr.shape[2], name="wavelength_indices"
        )

    powers = np.zeros(ports.size * wave_idx.size, dtype=np.float64)
    row0 = 0
    for wave in wave_idx:
        phase = np.exp(1j * _TWO_PI * pistons / wavelengths[wave])
        field_w = fields[:, wave, :] * phase[:, np.newaxis]
        electric = u_arr[ports, :, wave] @ field_w
        rows = slice(row0, row0 + ports.size)
        powers[rows] = np.sum(np.abs(electric) ** 2, axis=1)
        row0 += int(ports.size)
    return powers


def spectral_channel_counts_for_pistons(
    *,
    u_stack: NDArray[np.complexfloating],
    input_fields: NDArray[np.complexfloating],
    wavelengths_m: NDArray[np.floating],
    port_indices: NDArray[np.integer],
    pistons_m: NDArray[np.floating],
    wavelength_groups: tuple[NDArray[np.integer], ...],
    expected_counts_by_wavelength: NDArray[np.floating],
    background_counts_by_wavelength: NDArray[np.floating] | None = None,
) -> NDArray[np.float64]:
    """Evaluate total counts for wavelength bins summed into count channels."""

    u_arr = _as_unitary_stack(u_stack)
    fields = _as_input_fields(input_fields, u_arr.shape[1], u_arr.shape[2])
    wavelengths = _as_wavelengths(wavelengths_m, u_arr.shape[2])
    ports = _as_indices(port_indices, upper=u_arr.shape[0], name="port_indices")
    pistons = np.asarray(pistons_m, dtype=np.float64)
    if pistons.shape != (u_arr.shape[1],):
        raise ValueError(
            f"pistons_m must have shape ({u_arr.shape[1]},), got {pistons.shape}."
        )
    groups = _as_wavelength_groups(wavelength_groups, n_wavelengths=u_arr.shape[2])
    counts_by_wave = _as_counts_by_wavelength(
        expected_counts_by_wavelength,
        n_ports=ports.size,
        n_wavelengths=u_arr.shape[2],
    )
    if background_counts_by_wavelength is None:
        background_by_wave = np.zeros_like(counts_by_wave)
    else:
        background_by_wave = _as_counts_by_wavelength(
            background_counts_by_wavelength,
            n_ports=ports.size,
            n_wavelengths=u_arr.shape[2],
        )
    counts_per_power = _counts_per_power_by_wavelength(
        u_stack=u_arr,
        input_fields=fields,
        wavelengths_m=wavelengths,
        port_indices=ports,
        expected_counts_by_wavelength=counts_by_wave,
    )
    powers = port_powers_for_pistons(
        u_stack=u_arr,
        input_fields=fields,
        wavelengths_m=wavelengths,
        port_indices=ports,
        pistons_m=pistons,
        wavelength_indices=np.arange(u_arr.shape[2], dtype=np.intp),
    ).reshape(u_arr.shape[2], ports.size).T
    return _sum_spectral_channel_counts(
        signal_by_wavelength=np.maximum(powers * counts_per_power, 0.0),
        background_by_wavelength=background_by_wave,
        wavelength_groups=groups,
    )


def intensity_response_matrix_at_pistons(
    *,
    u_stack: NDArray[np.complexfloating],
    input_fields: NDArray[np.complexfloating],
    wavelengths_m: NDArray[np.floating],
    port_indices: NDArray[np.integer],
    pistons_m: NDArray[np.floating],
    wavelength_indices: NDArray[np.integer] | None = None,
) -> IntensityResponse:
    """Return the local intensity response at nonzero chip-input pistons."""

    u_arr = _as_unitary_stack(u_stack)
    fields = _as_input_fields(input_fields, u_arr.shape[1], u_arr.shape[2])
    wavelengths = _as_wavelengths(wavelengths_m, u_arr.shape[2])
    ports = _as_indices(port_indices, upper=u_arr.shape[0], name="port_indices")
    pistons = np.asarray(pistons_m, dtype=np.float64)
    if pistons.shape != (u_arr.shape[1],):
        raise ValueError(
            f"pistons_m must have shape ({u_arr.shape[1]},), got {pistons.shape}."
        )
    if wavelength_indices is None:
        wave_idx = np.arange(u_arr.shape[2], dtype=np.intp)
    else:
        wave_idx = _as_indices(
            wavelength_indices, upper=u_arr.shape[2], name="wavelength_indices"
        )
    if ports.size == 0:
        raise ValueError(
            "intensity_response_matrix_at_pistons requires at least one port."
        )
    if wave_idx.size == 0:
        raise ValueError(
            "intensity_response_matrix_at_pistons requires at least one wavelength bin."
        )

    n_rows = int(ports.size * wave_idx.size)
    response = np.zeros((n_rows, u_arr.shape[1]), dtype=np.float64)
    reference_power = np.zeros(n_rows, dtype=np.float64)
    row_ports = np.zeros(n_rows, dtype=np.intp)
    row_waves = np.zeros(n_rows, dtype=np.intp)

    row0 = 0
    for wave in wave_idx:
        phase = np.exp(1j * _TWO_PI * pistons / wavelengths[wave])
        u_ports = u_arr[ports, :, wave]
        field_w = fields[:, wave, :] * phase[:, np.newaxis]
        electric = u_ports @ field_w
        d_electric_dp = (
            u_ports[:, :, np.newaxis]
            * field_w[np.newaxis, :, :]
            * (1j * _TWO_PI / wavelengths[wave])
        )
        derivative = 2.0 * np.real(
            np.sum(np.conj(electric)[:, np.newaxis, :] * d_electric_dp, axis=2)
        )
        rows = slice(row0, row0 + ports.size)
        response[rows, :] = derivative
        reference_power[rows] = np.sum(np.abs(electric) ** 2, axis=1)
        row_ports[rows] = ports
        row_waves[rows] = wave
        row0 += int(ports.size)

    return IntensityResponse(
        response_power_per_m=response,
        reference_power=reference_power,
        row_port_indices=row_ports,
        row_wavelength_indices=row_waves,
    )


def scale_response_to_counts(
    *,
    response_power_per_m: NDArray[np.floating],
    reference_power: NDArray[np.floating],
    expected_counts: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Scale a power response into detector-count response units."""

    response = np.asarray(response_power_per_m, dtype=np.float64)
    reference = np.asarray(reference_power, dtype=np.float64)
    counts = np.asarray(expected_counts, dtype=np.float64)
    if response.ndim != 2:
        raise ValueError("response_power_per_m must be a 2-D matrix.")
    if reference.shape != (response.shape[0],):
        raise ValueError("reference_power length must match response rows.")
    if counts.shape != (response.shape[0],):
        raise ValueError("expected_counts length must match response rows.")
    if not (
        np.all(np.isfinite(response))
        and np.all(np.isfinite(reference))
        and np.all(np.isfinite(counts))
    ):
        raise ValueError(
            "response, reference_power, and expected_counts must be finite."
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(reference > 0.0, counts / reference, 0.0)
    scale = np.where(np.isfinite(scale), scale, 0.0)
    return response * scale[:, np.newaxis]


def whiten_response_matrix(
    *,
    response_counts_per_m: NDArray[np.floating],
    expected_counts: NDArray[np.floating] | None = None,
    variance_counts: NDArray[np.floating] | None = None,
    noise_floor_counts: float = 1.0,
) -> NDArray[np.float64]:
    """Whiten count-response rows by Poisson shot noise.

    ``noise_floor_counts`` is a numerical regularizer for zero-signal rows, not
    a physical dark/background model. Pass ``variance_counts`` when dark or
    background counts should contribute to the row variance.
    """

    response = np.asarray(response_counts_per_m, dtype=np.float64)
    if variance_counts is None:
        if expected_counts is None:
            raise ValueError("expected_counts or variance_counts is required.")
        counts = np.asarray(expected_counts, dtype=np.float64)
    else:
        counts = np.asarray(variance_counts, dtype=np.float64)
    if response.ndim != 2:
        raise ValueError("response_counts_per_m must be a 2-D matrix.")
    if counts.shape != (response.shape[0],):
        raise ValueError("variance/count length must match response rows.")
    if noise_floor_counts < 0.0:
        raise ValueError("noise_floor_counts must be non-negative.")
    variance = np.maximum(counts, float(noise_floor_counts))
    denom = np.sqrt(variance)
    return response / denom[:, np.newaxis]


def svd_observability(
    response_matrix: NDArray[np.floating],
    *,
    relative_threshold: float = 1.0e-6,
    absolute_threshold: float = 0.0,
) -> ObservabilitySVD:
    """Return rank and conditioning diagnostics for a response matrix."""

    matrix = np.asarray(response_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("response_matrix must be a 2-D matrix.")
    if matrix.shape[1] == 0:
        raise ValueError("response_matrix must have at least one piston column.")
    if relative_threshold <= 0.0:
        raise ValueError("relative_threshold must be positive.")
    if absolute_threshold < 0.0:
        raise ValueError("absolute_threshold must be non-negative.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("response_matrix must contain only finite values.")

    _u, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)
    if singular_values.size == 0:
        tolerance = float(absolute_threshold)
        rank = 0
    else:
        tolerance = max(
            float(absolute_threshold),
            float(relative_threshold) * float(singular_values[0]),
        )
        rank = int(np.sum(singular_values > tolerance))

    n_dof = int(matrix.shape[1])
    full_rank = rank >= n_dof
    min_full_sv = (
        float(singular_values[n_dof - 1]) if singular_values.size >= n_dof else 0.0
    )
    condition = (
        float(singular_values[0] / min_full_sv)
        if singular_values.size > 0 and min_full_sv > 0.0 and full_rank
        else float("inf")
    )
    truncated = (
        float(singular_values[0] / singular_values[rank - 1])
        if singular_values.size > 0 and rank > 0 and singular_values[rank - 1] > 0.0
        else float("inf")
    )
    return ObservabilitySVD(
        singular_values=singular_values.astype(np.float64, copy=True),
        right_singular_vectors=vh.astype(np.float64, copy=True),
        tolerance=float(tolerance),
        rank=rank,
        full_rank=bool(full_rank),
        observable_fraction=float(rank) / float(n_dof),
        condition_number=condition,
        truncated_condition_number=truncated,
    )


def cramer_rao_bound_from_svd(
    svd: ObservabilitySVD,
    *,
    n_piston_modes: int,
) -> CramerRaoBound:
    """Return OPD CRLB metrics from an observability SVD.

    The response columns are orthonormal differential-piston coordinates with
    units of meters. ``piston_rms_m`` is the average per-aperture differential
    piston RMS floor, while ``coordinate_rms_m`` averages over the
    ``M - 1`` basis coordinates. If the response is rank deficient, the full
    CRLB is infinite and the ``truncated_*`` fields report the finite floor
    over the observable singular subspace only.
    """

    if n_piston_modes < 2:
        raise ValueError("n_piston_modes must be at least 2.")
    n_dof = int(svd.right_singular_vectors.shape[1])
    singular = np.asarray(svd.singular_values, dtype=np.float64)
    singular_mode_std = np.full(n_dof, float("inf"), dtype=np.float64)
    finite_count = min(int(svd.rank), n_dof, singular.size)
    if finite_count > 0:
        singular_mode_std[:finite_count] = 1.0 / singular[:finite_count]

    if finite_count > 0:
        truncated_var_sum = float(np.sum(singular_mode_std[:finite_count] ** 2))
        truncated_piston = np.sqrt(truncated_var_sum / float(n_piston_modes))
        truncated_coord = np.sqrt(truncated_var_sum / float(n_dof))
    else:
        truncated_piston = float("inf")
        truncated_coord = float("inf")

    if svd.full_rank:
        full_var_sum = float(np.sum(singular_mode_std[:n_dof] ** 2))
        piston = float(np.sqrt(full_var_sum / float(n_piston_modes)))
        coord = float(np.sqrt(full_var_sum / float(n_dof)))
        finite = True
    else:
        piston = float("inf")
        coord = float("inf")
        finite = False

    return CramerRaoBound(
        singular_mode_std_m=singular_mode_std,
        piston_rms_m=piston,
        coordinate_rms_m=coord,
        truncated_piston_rms_m=float(truncated_piston),
        truncated_coordinate_rms_m=float(truncated_coord),
        finite=finite,
    )


def piston_measurement_covariance_from_svd(
    svd: ObservabilitySVD,
    *,
    piston_basis: NDArray[np.floating],
) -> PistonMeasurementCovariance:
    """Return the full-rank physical piston covariance from an SVD response.

    The input ``svd`` must be computed from the whitened count-response matrix
    ``A`` in orthonormal differential-piston coordinates. When ``A`` is full
    column rank, the coordinate covariance is ``(A.T @ A)^-1`` and the physical
    covariance is ``B @ cov @ B.T`` for piston basis ``B``. If the response is
    rank deficient, the finite observable-subspace pseudo-covariance would hide
    unobservable modes, so the returned covariance matrices are filled with
    ``nan`` and ``finite`` is false.
    """

    basis = np.asarray(piston_basis, dtype=np.float64)
    if basis.ndim != 2:
        raise ValueError("piston_basis must be a 2-D matrix.")
    n_modes, n_dof = basis.shape
    vh = np.asarray(svd.right_singular_vectors, dtype=np.float64)
    singular = np.asarray(svd.singular_values, dtype=np.float64)
    if vh.ndim != 2 or vh.shape[1] != n_dof:
        raise ValueError("right_singular_vectors width must match piston_basis.")
    if n_modes < 2:
        raise ValueError("piston_basis must describe at least two piston modes.")
    if n_dof == 0:
        raise ValueError("piston_basis must contain at least one differential mode.")

    if not svd.full_rank or singular.size < n_dof or np.any(singular[:n_dof] <= 0.0):
        coord_cov = np.full((n_dof, n_dof), np.nan, dtype=np.float64)
        piston_cov = np.full((n_modes, n_modes), np.nan, dtype=np.float64)
        return PistonMeasurementCovariance(
            coordinate_covariance_m2=coord_cov,
            piston_covariance_m2=piston_cov,
            piston_rms_m=float("inf"),
            coordinate_rms_m=float("inf"),
            finite=False,
        )

    v = vh[:n_dof, :].T
    inv_var = 1.0 / (singular[:n_dof] ** 2)
    coord_cov = (v * inv_var[np.newaxis, :]) @ v.T
    coord_cov = 0.5 * (coord_cov + coord_cov.T)
    piston_cov = basis @ coord_cov @ basis.T
    piston_cov = 0.5 * (piston_cov + piston_cov.T)
    coord_trace = float(np.trace(coord_cov))
    piston_trace = float(np.trace(piston_cov))
    return PistonMeasurementCovariance(
        coordinate_covariance_m2=coord_cov.astype(np.float64, copy=True),
        piston_covariance_m2=piston_cov.astype(np.float64, copy=True),
        piston_rms_m=float(np.sqrt(max(piston_trace, 0.0) / float(n_modes))),
        coordinate_rms_m=float(np.sqrt(max(coord_trace, 0.0) / float(n_dof))),
        finite=True,
    )


def weakest_piston_modes(
    *,
    right_singular_vectors: NDArray[np.floating],
    piston_basis: NDArray[np.floating],
    n_weak_modes: int,
) -> NDArray[np.float64]:
    """Map the weakest right-singular vectors into per-aperture piston modes."""

    vh = np.asarray(right_singular_vectors, dtype=np.float64)
    basis = np.asarray(piston_basis, dtype=np.float64)
    if vh.ndim != 2:
        raise ValueError("right_singular_vectors must be a 2-D matrix.")
    if basis.shape[1] != vh.shape[1]:
        raise ValueError("piston_basis columns must match singular-vector width.")
    count = min(int(n_weak_modes), vh.shape[0])
    if count <= 0:
        return np.zeros((0, basis.shape[0]), dtype=np.float64)
    modes = np.zeros((count, basis.shape[0]), dtype=np.float64)
    for out_i, coeff in enumerate(vh[-count:]):
        piston = basis @ coeff
        rms = float(np.sqrt(np.mean(piston * piston)))
        if rms > 0.0:
            piston = piston / rms
        modes[out_i, :] = piston
    return modes


def reconstruction_monte_carlo(
    *,
    u_stack: NDArray[np.complexfloating],
    input_fields: NDArray[np.complexfloating],
    wavelengths_m: NDArray[np.floating],
    port_indices: NDArray[np.integer],
    wavelength_indices: NDArray[np.integer],
    piston_basis: NDArray[np.floating],
    reference_power: NDArray[np.floating],
    expected_counts: NDArray[np.floating],
    whitened_response_per_m: NDArray[np.floating],
    piston_rms_m: float,
    n_trials: int,
    rng: np.random.Generator,
    field_piston_basis: NDArray[np.floating] | None = None,
    variance_counts: NDArray[np.floating] | None = None,
    background_counts: NDArray[np.floating] | None = None,
    wavelength_groups: tuple[NDArray[np.integer], ...] | None = None,
    expected_counts_by_wavelength: NDArray[np.floating] | None = None,
    background_counts_by_wavelength: NDArray[np.floating] | None = None,
    relative_threshold: float = 1.0e-6,
    noise_floor_counts: float = 1.0,
    shot_noise: bool = True,
) -> ReconstructionMonteCarlo:
    """Apply one linearized inverse to nonlinear count measurements.

    This is a capture-range diagnostic. The residual includes model curvature
    from evaluating the true nonlinear chip intensity at finite piston offset,
    so it is not expected to match the small-signal CRLB.
    """

    if n_trials < 0:
        raise ValueError("n_trials must be non-negative.")
    if piston_rms_m < 0.0:
        raise ValueError("piston_rms_m must be non-negative.")
    if noise_floor_counts < 0.0:
        raise ValueError("noise_floor_counts must be non-negative.")
    if n_trials == 0:
        empty = np.zeros(0, dtype=np.float64)
        return ReconstructionMonteCarlo(
            empty, empty, empty, float("nan"), float("nan"), float("nan")
        )

    basis = np.asarray(piston_basis, dtype=np.float64)
    field_basis = (
        basis
        if field_piston_basis is None
        else np.asarray(field_piston_basis, dtype=np.float64)
    )
    expected = np.asarray(expected_counts, dtype=np.float64)
    variance = (
        np.asarray(variance_counts, dtype=np.float64)
        if variance_counts is not None
        else expected
    )
    background = (
        np.asarray(background_counts, dtype=np.float64)
        if background_counts is not None
        else np.zeros_like(expected)
    )
    reference = np.asarray(reference_power, dtype=np.float64)
    whitened = np.asarray(whitened_response_per_m, dtype=np.float64)
    if expected.shape != reference.shape:
        raise ValueError(
            "expected_counts and reference_power must have the same shape."
        )
    if variance.shape != expected.shape or background.shape != expected.shape:
        raise ValueError(
            "variance_counts and background_counts must match expected_counts."
        )
    if whitened.shape[0] != expected.size or whitened.shape[1] != basis.shape[1]:
        raise ValueError("whitened_response_per_m shape is inconsistent.")
    if field_basis.ndim != 2 or field_basis.shape[1] != basis.shape[1]:
        raise ValueError("field_piston_basis shape is inconsistent.")

    grouped_model = wavelength_groups is not None
    if grouped_model and expected_counts_by_wavelength is None:
        raise ValueError(
            "expected_counts_by_wavelength is required with wavelength_groups."
        )
    if not grouped_model:
        with np.errstate(divide="ignore", invalid="ignore"):
            counts_per_power = np.where(reference > 0.0, expected / reference, 0.0)
        counts_per_power = np.where(
            np.isfinite(counts_per_power),
            counts_per_power,
            0.0,
        )
    noise = np.sqrt(np.maximum(variance, float(noise_floor_counts)))
    estimator = np.linalg.pinv(whitened, rcond=float(relative_threshold))
    n_modes = int(basis.shape[0])
    n_dof = int(basis.shape[1])
    coord_sigma = (
        float(piston_rms_m) * np.sqrt(float(n_modes) / float(n_dof))
        if n_dof > 0
        else 0.0
    )

    input_rms = np.zeros(n_trials, dtype=np.float64)
    recovered_rms = np.zeros(n_trials, dtype=np.float64)
    residual_rms = np.zeros(n_trials, dtype=np.float64)
    for i in range(n_trials):
        coords_true = rng.normal(0.0, coord_sigma, size=n_dof)
        pistons_true = basis @ coords_true
        field_pistons_true = field_basis @ coords_true
        if grouped_model:
            assert wavelength_groups is not None
            assert expected_counts_by_wavelength is not None
            mu_total = spectral_channel_counts_for_pistons(
                u_stack=u_stack,
                input_fields=input_fields,
                wavelengths_m=wavelengths_m,
                port_indices=port_indices,
                pistons_m=field_pistons_true,
                wavelength_groups=wavelength_groups,
                expected_counts_by_wavelength=expected_counts_by_wavelength,
                background_counts_by_wavelength=background_counts_by_wavelength,
            )
        else:
            powers = port_powers_for_pistons(
                u_stack=u_stack,
                input_fields=input_fields,
                wavelengths_m=wavelengths_m,
                port_indices=port_indices,
                pistons_m=field_pistons_true,
                wavelength_indices=wavelength_indices,
            )
            mu_signal = np.maximum(powers * counts_per_power, 0.0)
            mu_total = mu_signal + background
        expected_total = expected + background
        observed = rng.poisson(mu_total) if shot_noise else mu_total
        whitened_delta = (observed - expected_total) / noise
        coords_hat = estimator @ whitened_delta
        pistons_hat = basis @ coords_hat
        residual = pistons_true - pistons_hat
        input_rms[i] = _rms(pistons_true)
        recovered_rms[i] = _rms(pistons_hat)
        residual_rms[i] = _rms(residual)

    return ReconstructionMonteCarlo(
        input_rms_m=input_rms,
        recovered_rms_m=recovered_rms,
        residual_rms_m=residual_rms,
        median_residual_rms_m=float(np.median(residual_rms)),
        p95_residual_rms_m=float(np.percentile(residual_rms, 95.0)),
        median_recovered_rms_m=float(np.median(recovered_rms)),
    )


def gauss_newton_reconstruction_monte_carlo(
    *,
    u_stack: NDArray[np.complexfloating],
    input_fields: NDArray[np.complexfloating],
    wavelengths_m: NDArray[np.floating],
    port_indices: NDArray[np.integer],
    wavelength_indices: NDArray[np.integer],
    piston_basis: NDArray[np.floating],
    reference_power: NDArray[np.floating],
    expected_counts: NDArray[np.floating],
    piston_rms_m: float,
    n_trials: int,
    rng: np.random.Generator,
    field_piston_basis: NDArray[np.floating] | None = None,
    variance_counts: NDArray[np.floating] | None = None,
    background_counts: NDArray[np.floating] | None = None,
    whitened_response_per_m: NDArray[np.floating] | None = None,
    wavelength_groups: tuple[NDArray[np.integer], ...] | None = None,
    expected_counts_by_wavelength: NDArray[np.floating] | None = None,
    background_counts_by_wavelength: NDArray[np.floating] | None = None,
    relative_threshold: float = 1.0e-6,
    shot_noise: bool = True,
    max_iter: int = 8,
    damping: float = 1.0e-3,
    step_tolerance_m: float = 1.0e-12,
    cost_relative_tolerance: float = 1.0e-6,
    initialization: str = "linear",
    noise_floor_counts: float = 1.0,
) -> GaussNewtonMonteCarlo:
    """Fit nonlinear bright-port counts with damped Gauss-Newton iterations."""

    if n_trials < 0:
        raise ValueError("n_trials must be non-negative.")
    if piston_rms_m < 0.0:
        raise ValueError("piston_rms_m must be non-negative.")
    if max_iter < 0:
        raise ValueError("max_iter must be non-negative.")
    if damping < 0.0:
        raise ValueError("damping must be non-negative.")
    if step_tolerance_m < 0.0:
        raise ValueError("step_tolerance_m must be non-negative.")
    if cost_relative_tolerance < 0.0:
        raise ValueError("cost_relative_tolerance must be non-negative.")
    if noise_floor_counts < 0.0:
        raise ValueError("noise_floor_counts must be non-negative.")
    if initialization not in {"linear", "zero"}:
        raise ValueError("initialization must be 'linear' or 'zero'.")
    if n_trials == 0:
        empty = np.zeros(0, dtype=np.float64)
        return GaussNewtonMonteCarlo(
            input_rms_m=empty,
            recovered_rms_m=empty,
            residual_rms_m=empty,
            median_residual_rms_m=float("nan"),
            p95_residual_rms_m=float("nan"),
            median_recovered_rms_m=float("nan"),
            converged=np.zeros(0, dtype=bool),
            iterations=np.zeros(0, dtype=np.int64),
            final_weighted_cost=empty,
            convergence_fraction=float("nan"),
            max_iter_fraction=float("nan"),
            worse_than_input_fraction=float("nan"),
        )

    basis = np.asarray(piston_basis, dtype=np.float64)
    field_basis = (
        basis
        if field_piston_basis is None
        else np.asarray(field_piston_basis, dtype=np.float64)
    )
    expected = np.asarray(expected_counts, dtype=np.float64)
    variance = (
        np.asarray(variance_counts, dtype=np.float64)
        if variance_counts is not None
        else expected
    )
    background = (
        np.asarray(background_counts, dtype=np.float64)
        if background_counts is not None
        else np.zeros_like(expected)
    )
    reference = np.asarray(reference_power, dtype=np.float64)
    if expected.shape != reference.shape:
        raise ValueError(
            "expected_counts and reference_power must have the same shape."
        )
    if variance.shape != expected.shape or background.shape != expected.shape:
        raise ValueError(
            "variance_counts and background_counts must match expected_counts."
        )
    if basis.ndim != 2:
        raise ValueError("piston_basis must be a 2-D matrix.")
    if field_basis.ndim != 2 or field_basis.shape[1] != basis.shape[1]:
        raise ValueError("field_piston_basis shape is inconsistent.")
    if initialization == "linear":
        if whitened_response_per_m is None:
            raise ValueError(
                "whitened_response_per_m is required for linear initialization."
            )
        zero_estimator = np.linalg.pinv(
            np.asarray(whitened_response_per_m, dtype=np.float64),
            rcond=float(relative_threshold),
        )
        zero_noise = np.sqrt(np.maximum(variance, float(noise_floor_counts)))
    else:
        zero_estimator = None
        zero_noise = None

    grouped_model = wavelength_groups is not None
    if grouped_model and expected_counts_by_wavelength is None:
        raise ValueError(
            "expected_counts_by_wavelength is required with wavelength_groups."
        )
    if grouped_model:
        assert expected_counts_by_wavelength is not None
        expected_by_wave = _as_counts_by_wavelength(
            expected_counts_by_wavelength,
            n_ports=np.asarray(port_indices).size,
            n_wavelengths=np.asarray(wavelengths_m).size,
        )
        background_by_wave = (
            np.zeros_like(expected_by_wave)
            if background_counts_by_wavelength is None
            else _as_counts_by_wavelength(
                background_counts_by_wavelength,
                n_ports=np.asarray(port_indices).size,
                n_wavelengths=np.asarray(wavelengths_m).size,
            )
        )
        groups = _as_wavelength_groups(
            wavelength_groups,
            n_wavelengths=np.asarray(wavelengths_m).size,
        )
        counts_per_power = _counts_per_power_by_wavelength(
            u_stack=u_stack,
            input_fields=input_fields,
            wavelengths_m=wavelengths_m,
            port_indices=port_indices,
            expected_counts_by_wavelength=expected_by_wave,
        )
    else:
        expected_by_wave = None
        background_by_wave = None
        groups = None
        with np.errstate(divide="ignore", invalid="ignore"):
            counts_per_power = np.where(reference > 0.0, expected / reference, 0.0)
        counts_per_power = np.where(
            np.isfinite(counts_per_power),
            counts_per_power,
            0.0,
        )
    n_modes = int(basis.shape[0])
    n_dof = int(basis.shape[1])
    coord_sigma = (
        float(piston_rms_m) * np.sqrt(float(n_modes) / float(n_dof))
        if n_dof > 0
        else 0.0
    )

    input_rms = np.zeros(n_trials, dtype=np.float64)
    recovered_rms = np.zeros(n_trials, dtype=np.float64)
    residual_rms = np.zeros(n_trials, dtype=np.float64)
    iterations = np.zeros(n_trials, dtype=np.int64)
    final_cost = np.zeros(n_trials, dtype=np.float64)
    converged = np.zeros(n_trials, dtype=bool)
    expected_total = expected + background

    for trial in range(n_trials):
        coords_true = rng.normal(0.0, coord_sigma, size=n_dof)
        pistons_true = basis @ coords_true
        field_pistons_true = field_basis @ coords_true
        if grouped_model:
            assert groups is not None and expected_by_wave is not None
            true_total = spectral_channel_counts_for_pistons(
                u_stack=u_stack,
                input_fields=input_fields,
                wavelengths_m=wavelengths_m,
                port_indices=port_indices,
                pistons_m=field_pistons_true,
                wavelength_groups=groups,
                expected_counts_by_wavelength=expected_by_wave,
                background_counts_by_wavelength=background_by_wave,
            )
        else:
            true_powers = port_powers_for_pistons(
                u_stack=u_stack,
                input_fields=input_fields,
                wavelengths_m=wavelengths_m,
                port_indices=port_indices,
                pistons_m=field_pistons_true,
                wavelength_indices=wavelength_indices,
            )
            true_signal = np.maximum(true_powers * counts_per_power, 0.0)
            true_total = true_signal + background
        observed = rng.poisson(true_total) if shot_noise else true_total

        if initialization == "linear":
            assert zero_estimator is not None and zero_noise is not None
            coords = zero_estimator @ ((observed - expected_total) / zero_noise)
        else:
            coords = np.zeros(n_dof, dtype=np.float64)

        cost_previous = float("inf")
        for iteration in range(1, int(max_iter) + 1):
            if grouped_model:
                assert groups is not None and background_by_wave is not None
                mu, jac = _spectral_channel_counts_model_and_jacobian(
                    coords=coords,
                    u_stack=u_stack,
                    input_fields=input_fields,
                    wavelengths_m=wavelengths_m,
                    port_indices=port_indices,
                    wavelength_groups=groups,
                    field_piston_basis=field_basis,
                    counts_per_power_by_wavelength=counts_per_power,
                    background_counts_by_wavelength=background_by_wave,
                )
            else:
                mu, jac = _counts_model_and_jacobian(
                    coords=coords,
                    u_stack=u_stack,
                    input_fields=input_fields,
                    wavelengths_m=wavelengths_m,
                    port_indices=port_indices,
                    wavelength_indices=wavelength_indices,
                    field_piston_basis=field_basis,
                    counts_per_power=counts_per_power,
                    background_counts=background,
                )
            noise = np.sqrt(np.maximum(mu, float(noise_floor_counts)))
            residual = observed - mu
            weighted_residual = residual / noise
            weighted_jac = jac / noise[:, np.newaxis]
            cost = float(np.dot(weighted_residual, weighted_residual))
            lhs = weighted_jac.T @ weighted_jac
            rhs = weighted_jac.T @ weighted_residual
            diag = np.maximum(np.diag(lhs), 1.0)
            lhs = lhs + float(damping) * np.diag(diag)
            try:
                delta = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(lhs, rhs, rcond=float(relative_threshold))[0]

            step = 1.0
            accepted = False
            while step >= 1.0 / 64.0:
                candidate = coords + step * delta
                if grouped_model:
                    assert groups is not None and background_by_wave is not None
                    candidate_mu, _candidate_jac = (
                        _spectral_channel_counts_model_and_jacobian(
                            coords=candidate,
                            u_stack=u_stack,
                            input_fields=input_fields,
                            wavelengths_m=wavelengths_m,
                            port_indices=port_indices,
                            wavelength_groups=groups,
                            field_piston_basis=field_basis,
                            counts_per_power_by_wavelength=counts_per_power,
                            background_counts_by_wavelength=background_by_wave,
                        )
                    )
                else:
                    candidate_mu, _candidate_jac = _counts_model_and_jacobian(
                        coords=candidate,
                        u_stack=u_stack,
                        input_fields=input_fields,
                        wavelengths_m=wavelengths_m,
                        port_indices=port_indices,
                        wavelength_indices=wavelength_indices,
                        field_piston_basis=field_basis,
                        counts_per_power=counts_per_power,
                        background_counts=background,
                    )
                candidate_noise = np.sqrt(
                    np.maximum(candidate_mu, float(noise_floor_counts))
                )
                candidate_residual = (observed - candidate_mu) / candidate_noise
                candidate_cost = float(np.dot(candidate_residual, candidate_residual))
                if candidate_cost <= cost:
                    coords = candidate
                    final_cost[trial] = candidate_cost
                    accepted = True
                    break
                step *= 0.5

            iterations[trial] = iteration
            if not accepted:
                final_cost[trial] = cost
                # A backtracking line search that cannot improve an
                # already-small incumbent change means the trial sits at
                # the noise-limited optimum — count it as converged
                # instead of deflating convergence_fraction (review
                # 2026-06-10 Issue 30b).
                if cost_previous < float("inf"):
                    improvement = max(cost_previous - cost, 0.0)
                    if improvement <= float(cost_relative_tolerance) * max(
                        cost_previous, 1.0
                    ):
                        converged[trial] = True
                break
            step_rms = _rms(basis @ (step * delta))
            if step_rms <= float(step_tolerance_m):
                converged[trial] = True
                break
            if cost_previous < float("inf"):
                improvement = max(cost_previous - final_cost[trial], 0.0)
                if improvement <= float(cost_relative_tolerance) * max(
                    cost_previous, 1.0
                ):
                    converged[trial] = True
                    break
            cost_previous = final_cost[trial]
        else:
            # Exhausted max_iter without meeting a tolerance: not converged.
            # (The old ``max_iter == 0`` special case marked zero-iteration
            # runs converged — misleading; review 2026-06-10 Issue 30b.)
            converged[trial] = False

        pistons_hat = basis @ coords
        residual_pistons = pistons_true - pistons_hat
        input_rms[trial] = _rms(pistons_true)
        recovered_rms[trial] = _rms(pistons_hat)
        residual_rms[trial] = _rms(residual_pistons)

    max_iter_fraction = float(np.mean(iterations >= int(max_iter))) if max_iter else 0.0
    worse = residual_rms > np.maximum(input_rms, 0.0)
    return GaussNewtonMonteCarlo(
        input_rms_m=input_rms,
        recovered_rms_m=recovered_rms,
        residual_rms_m=residual_rms,
        median_residual_rms_m=float(np.median(residual_rms)),
        p95_residual_rms_m=float(np.percentile(residual_rms, 95.0)),
        median_recovered_rms_m=float(np.median(recovered_rms)),
        converged=converged.astype(bool, copy=True),
        iterations=iterations.astype(np.int64, copy=True),
        final_weighted_cost=final_cost.astype(np.float64, copy=True),
        convergence_fraction=float(np.mean(converged)),
        max_iter_fraction=max_iter_fraction,
        worse_than_input_fraction=float(np.mean(worse)),
    )


def active_coordinate_acquisition_monte_carlo(
    *,
    u_stack: NDArray[np.complexfloating],
    input_fields: NDArray[np.complexfloating],
    wavelengths_m: NDArray[np.floating],
    port_indices: NDArray[np.integer],
    wavelength_indices: NDArray[np.integer],
    piston_basis: NDArray[np.floating],
    reference_power: NDArray[np.floating],
    expected_counts: NDArray[np.floating],
    piston_rms_m: float,
    n_trials: int,
    rng: np.random.Generator,
    scan_offsets_m: NDArray[np.floating],
    n_passes: int,
    reference_mode: int = 0,
    lock_threshold_m: float = 10.0e-9,
    response_iterations: int = 0,
    probe_step_m: float = 10.0e-9,
    max_response_step_m: float = 80.0e-9,
    field_piston_basis: NDArray[np.floating] | None = None,
    variance_counts: NDArray[np.floating] | None = None,
    background_counts: NDArray[np.floating] | None = None,
    wavelength_groups: tuple[NDArray[np.integer], ...] | None = None,
    expected_counts_by_wavelength: NDArray[np.floating] | None = None,
    background_counts_by_wavelength: NDArray[np.floating] | None = None,
    calibration_frames: int = 1,
    shot_noise: bool = True,
    noise_floor_counts: float = 1.0,
    relative_threshold: float = 1.0e-6,
) -> ActiveAcquisitionMonteCarlo:
    """Acquire piston lock by actively scanning one aperture correction at a time.

    The trial procedure emulates a calibration sequence rather than a passive
    estimator. For each Monte Carlo draw it fixes one physical aperture as the
    phase reference, scans trial corrections for every other aperture, measures
    the bright-port count signature after applying each correction, and keeps
    the candidate whose signature is closest to the zero-piston calibration.
    Repeating passes turns the scan into a coordinate-descent lock attempt.
    """

    if n_trials < 0:
        raise ValueError("n_trials must be non-negative.")
    if piston_rms_m < 0.0:
        raise ValueError("piston_rms_m must be non-negative.")
    if n_passes <= 0:
        raise ValueError("n_passes must be positive.")
    if lock_threshold_m < 0.0:
        raise ValueError("lock_threshold_m must be non-negative.")
    if response_iterations < 0:
        raise ValueError("response_iterations must be non-negative.")
    if probe_step_m <= 0.0:
        raise ValueError("probe_step_m must be positive.")
    if max_response_step_m <= 0.0:
        raise ValueError("max_response_step_m must be positive.")
    if noise_floor_counts < 0.0:
        raise ValueError("noise_floor_counts must be non-negative.")
    if relative_threshold <= 0.0:
        raise ValueError("relative_threshold must be positive.")
    if calibration_frames <= 0:
        raise ValueError("calibration_frames must be positive.")
    offsets = np.asarray(scan_offsets_m, dtype=np.float64)
    if offsets.ndim != 1 or offsets.size == 0:
        raise ValueError("scan_offsets_m must be a non-empty 1-D array.")
    if not np.all(np.isfinite(offsets)):
        raise ValueError("scan_offsets_m must contain finite values.")
    if n_trials == 0:
        empty = np.zeros(0, dtype=np.float64)
        return ActiveAcquisitionMonteCarlo(
            input_rms_m=empty,
            recovered_rms_m=empty,
            residual_rms_m=empty,
            median_residual_rms_m=float("nan"),
            p95_residual_rms_m=float("nan"),
            median_recovered_rms_m=float("nan"),
            locked=np.zeros(0, dtype=bool),
            evaluations=np.zeros(0, dtype=np.int64),
            final_weighted_cost=empty,
            lock_fraction=float("nan"),
            worse_than_input_fraction=float("nan"),
        )

    basis = np.asarray(piston_basis, dtype=np.float64)
    field_basis = (
        basis
        if field_piston_basis is None
        else np.asarray(field_piston_basis, dtype=np.float64)
    )
    expected = np.asarray(expected_counts, dtype=np.float64)
    variance = (
        np.asarray(variance_counts, dtype=np.float64)
        if variance_counts is not None
        else expected
    )
    background = (
        np.asarray(background_counts, dtype=np.float64)
        if background_counts is not None
        else np.zeros_like(expected)
    )
    reference = np.asarray(reference_power, dtype=np.float64)
    if basis.ndim != 2:
        raise ValueError("piston_basis must be a 2-D matrix.")
    if field_basis.ndim != 2 or field_basis.shape[1] != basis.shape[1]:
        raise ValueError("field_piston_basis shape is inconsistent.")
    if not 0 <= int(reference_mode) < basis.shape[0]:
        raise ValueError("reference_mode must select a physical piston mode.")
    if expected.shape != reference.shape:
        raise ValueError(
            "expected_counts and reference_power must have the same shape."
        )
    if variance.shape != expected.shape or background.shape != expected.shape:
        raise ValueError(
            "variance_counts and background_counts must match expected_counts."
        )

    grouped_model = wavelength_groups is not None
    if grouped_model and expected_counts_by_wavelength is None:
        raise ValueError(
            "expected_counts_by_wavelength is required with wavelength_groups."
        )
    if grouped_model:
        assert expected_counts_by_wavelength is not None
        expected_by_wave = _as_counts_by_wavelength(
            expected_counts_by_wavelength,
            n_ports=np.asarray(port_indices).size,
            n_wavelengths=np.asarray(wavelengths_m).size,
        )
        background_by_wave = (
            np.zeros_like(expected_by_wave)
            if background_counts_by_wavelength is None
            else _as_counts_by_wavelength(
                background_counts_by_wavelength,
                n_ports=np.asarray(port_indices).size,
                n_wavelengths=np.asarray(wavelengths_m).size,
            )
        )
        groups = _as_wavelength_groups(
            wavelength_groups,
            n_wavelengths=np.asarray(wavelengths_m).size,
        )
        counts_per_power = _counts_per_power_by_wavelength(
            u_stack=u_stack,
            input_fields=input_fields,
            wavelengths_m=wavelengths_m,
            port_indices=port_indices,
            expected_counts_by_wavelength=expected_by_wave,
        )
    else:
        expected_by_wave = None
        background_by_wave = None
        groups = None
        with np.errstate(divide="ignore", invalid="ignore"):
            counts_per_power = np.where(reference > 0.0, expected / reference, 0.0)
        counts_per_power = np.where(
            np.isfinite(counts_per_power),
            counts_per_power,
            0.0,
        )
    zero_total = expected + background

    n_modes = int(basis.shape[0])
    n_dof = int(basis.shape[1])
    coord_sigma = (
        float(piston_rms_m) * np.sqrt(float(n_modes) / float(n_dof))
        if n_dof > 0
        else 0.0
    )
    active_modes = [mode for mode in range(n_modes) if mode != int(reference_mode)]

    input_rms = np.zeros(n_trials, dtype=np.float64)
    recovered_rms = np.zeros(n_trials, dtype=np.float64)
    residual_rms = np.zeros(n_trials, dtype=np.float64)
    evaluations = np.zeros(n_trials, dtype=np.int64)
    final_cost = np.zeros(n_trials, dtype=np.float64)
    locked = np.zeros(n_trials, dtype=bool)

    for trial in range(n_trials):
        coords_true = rng.normal(0.0, coord_sigma, size=n_dof)
        pistons_true = basis @ coords_true
        correction = np.zeros(n_modes, dtype=np.float64)
        best_overall_cost = float("inf")
        zero_reference = _calibrated_zero_reference(
            zero_total_counts=zero_total,
            rng=rng,
            n_frames=int(calibration_frames),
            shot_noise=bool(shot_noise),
        )
        cost_noise = np.sqrt(np.maximum(zero_reference, float(noise_floor_counts)))

        for _pass_i in range(int(n_passes)):
            for mode in active_modes:
                center = float(correction[mode])
                best_mode_value = center
                best_mode_cost = float("inf")
                for offset_m in offsets:
                    candidate = correction.copy()
                    candidate[mode] = center + float(offset_m)
                    cost = _active_acquisition_cost(
                        true_pistons_m=pistons_true,
                        correction_m=candidate,
                        piston_basis=basis,
                        field_piston_basis=field_basis,
                        u_stack=u_stack,
                        input_fields=input_fields,
                        wavelengths_m=wavelengths_m,
                        port_indices=port_indices,
                        wavelength_indices=wavelength_indices,
                        wavelength_groups=groups,
                        counts_per_power=counts_per_power,
                        background_counts=background,
                        background_counts_by_wavelength=background_by_wave,
                        zero_total_counts=zero_total,
                        zero_reference_counts=zero_reference,
                        cost_noise=cost_noise,
                        rng=rng,
                        shot_noise=bool(shot_noise),
                    )
                    evaluations[trial] += 1
                    if cost < best_mode_cost:
                        best_mode_cost = cost
                        best_mode_value = float(candidate[mode])
                correction[mode] = best_mode_value
                best_overall_cost = min(best_overall_cost, best_mode_cost)

        if response_iterations:
            correction, extra_evals, response_cost = _active_response_refinement(
                true_pistons_m=pistons_true,
                initial_correction_m=correction,
                active_modes=active_modes,
                reference_mode=int(reference_mode),
                piston_basis=basis,
                field_piston_basis=field_basis,
                u_stack=u_stack,
                input_fields=input_fields,
                wavelengths_m=wavelengths_m,
                port_indices=port_indices,
                wavelength_indices=wavelength_indices,
                wavelength_groups=groups,
                counts_per_power=counts_per_power,
                background_counts=background,
                background_counts_by_wavelength=background_by_wave,
                zero_total_counts=zero_total,
                zero_reference_counts=zero_reference,
                rng=rng,
                shot_noise=bool(shot_noise),
                noise_floor_counts=float(noise_floor_counts),
                n_iterations=int(response_iterations),
                probe_step_m=float(probe_step_m),
                max_step_m=float(max_response_step_m),
                relative_threshold=float(relative_threshold),
            )
            evaluations[trial] += extra_evals
            best_overall_cost = min(best_overall_cost, response_cost)

        correction = correction - float(np.mean(correction))
        residual = pistons_true - correction
        residual = residual - float(np.mean(residual))
        input_rms[trial] = _rms(pistons_true)
        recovered_rms[trial] = _rms(correction)
        residual_rms[trial] = _rms(residual)
        final_cost[trial] = best_overall_cost
        locked[trial] = residual_rms[trial] <= float(lock_threshold_m)

    worse = residual_rms > np.maximum(input_rms, 0.0)
    return ActiveAcquisitionMonteCarlo(
        input_rms_m=input_rms,
        recovered_rms_m=recovered_rms,
        residual_rms_m=residual_rms,
        median_residual_rms_m=float(np.median(residual_rms)),
        p95_residual_rms_m=float(np.percentile(residual_rms, 95.0)),
        median_recovered_rms_m=float(np.median(recovered_rms)),
        locked=locked.astype(bool, copy=True),
        evaluations=evaluations.astype(np.int64, copy=True),
        final_weighted_cost=final_cost.astype(np.float64, copy=True),
        lock_fraction=float(np.mean(locked)),
        worse_than_input_fraction=float(np.mean(worse)),
    )


def balanced_binary_tree_edges(n_modes: int) -> NDArray[np.int64]:
    """Return ``n_modes - 1`` connected pairwise edges for tree cophasing.

    Edges are emitted in a cluster-readable order: local edges for the left
    half, one bridge edge, then local edges for the right half. For KILO's
    two-telescope indexing this keeps each telescope's ABCD rows contiguous
    with the inter-telescope piston measurement near the middle.
    """

    if n_modes < 2:
        raise ValueError("balanced_binary_tree_edges requires n_modes >= 2.")

    def group_edges(indices: list[int]) -> list[tuple[int, int]]:
        if len(indices) <= 1:
            return []
        mid = len(indices) // 2
        left = indices[:mid]
        right = indices[mid:]
        return (
            group_edges(left)
            + [(left[0], right[0])]
            + group_edges(right)
        )

    edges = group_edges(list(range(int(n_modes))))
    return np.asarray(edges, dtype=np.int64)


def cophasing_handoff_monte_carlo(
    *,
    wavelengths_m: NDArray[np.floating],
    cophasing_counts_per_channel: NDArray[np.floating],
    piston_basis: NDArray[np.floating],
    science_whitened_response_per_m: NDArray[np.floating],
    piston_rms_m: float,
    n_trials: int,
    rng: np.random.Generator,
    edges: NDArray[np.integer] | None = None,
    cophasing_counts_by_wavelength: NDArray[np.floating] | None = None,
    wavelength_groups: tuple[NDArray[np.integer], ...] | None = None,
    tracker_fraction: float = 1.0,
    visibility: float = 1.0,
    edge_throughput: float = 1.0,
    phase_offsets_rad: NDArray[np.floating] | None = None,
    edge_count_weights: NDArray[np.floating] | None = None,
    dark_counts_per_output: float = 0.0,
    count_cap_per_output: float | None = None,
    science_drift_rms_m: float = 0.0,
    lock_threshold_m: float = 10.0e-9,
    shot_noise: bool = True,
    relative_threshold: float = 1.0e-6,
) -> CophasingHandoffMonteCarlo:
    """Acquire with a dedicated pairwise cophasing unitary, then hand off.

    The cophasing unitary is modeled as a parameterized pairwise ABCD graph.
    Each edge measures one aperture-piston difference with four phase-diverse
    outputs. Wavelength bins can be summed into finite tracker count channels
    before the ABCD phase estimate, matching broadband count-channel hardware.
    The graph solution estimates physical aperture pistons, those corrections
    are applied upstream, and the residual is then passed through the science
    unitary's small-signal Fisher response for maintenance tracking.
    """

    if n_trials < 0:
        raise ValueError("n_trials must be non-negative.")
    if piston_rms_m < 0.0:
        raise ValueError("piston_rms_m must be non-negative.")
    if not 0.0 < tracker_fraction <= 1.0:
        raise ValueError("tracker_fraction must be in (0, 1].")
    if not 0.0 < visibility <= 1.0:
        raise ValueError("visibility must be in (0, 1].")
    if not 0.0 < edge_throughput <= 1.0:
        raise ValueError("edge_throughput must be in (0, 1].")
    if dark_counts_per_output < 0.0:
        raise ValueError("dark_counts_per_output must be non-negative.")
    if count_cap_per_output is not None and count_cap_per_output <= 0.0:
        raise ValueError("count_cap_per_output must be positive when set.")
    if science_drift_rms_m < 0.0:
        raise ValueError("science_drift_rms_m must be non-negative.")
    if lock_threshold_m < 0.0:
        raise ValueError("lock_threshold_m must be non-negative.")
    if relative_threshold <= 0.0:
        raise ValueError("relative_threshold must be positive.")

    basis = np.asarray(piston_basis, dtype=np.float64)
    if basis.ndim != 2:
        raise ValueError("piston_basis must be a 2-D matrix.")
    n_modes = int(basis.shape[0])
    n_dof = int(basis.shape[1])
    wavelengths = np.asarray(wavelengths_m, dtype=np.float64)
    if wavelengths.ndim != 1 or wavelengths.size == 0:
        raise ValueError("wavelengths_m must be a non-empty 1-D array.")
    if not np.all(np.isfinite(wavelengths)) or np.any(wavelengths <= 0.0):
        raise ValueError("wavelengths_m must contain finite positive values.")
    counts = np.asarray(cophasing_counts_per_channel, dtype=np.float64)
    grouped_cophasing = wavelength_groups is not None
    if grouped_cophasing:
        groups = _as_wavelength_groups(wavelength_groups, n_wavelengths=wavelengths.size)
        if counts.shape != (len(groups),):
            raise ValueError(
                "cophasing_counts_per_channel must match wavelength_groups."
            )
        if cophasing_counts_by_wavelength is None:
            raise ValueError(
                "cophasing_counts_by_wavelength is required with wavelength_groups."
            )
        counts_by_wavelength = np.asarray(
            cophasing_counts_by_wavelength,
            dtype=np.float64,
        )
        if counts_by_wavelength.shape != wavelengths.shape:
            raise ValueError(
                "cophasing_counts_by_wavelength must match wavelengths_m."
            )
    else:
        groups = tuple(np.array([i], dtype=np.intp) for i in range(wavelengths.size))
        if counts.shape != wavelengths.shape:
            raise ValueError("cophasing_counts_per_channel must match wavelengths_m.")
        counts_by_wavelength = counts
    if not np.all(np.isfinite(counts)) or np.any(counts < 0.0):
        raise ValueError("cophasing_counts_per_channel must be non-negative.")
    if not np.all(np.isfinite(counts_by_wavelength)) or np.any(
        counts_by_wavelength < 0.0
    ):
        raise ValueError("cophasing_counts_by_wavelength must be non-negative.")
    phase_offsets = (
        np.array([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi], dtype=np.float64)
        if phase_offsets_rad is None
        else np.asarray(phase_offsets_rad, dtype=np.float64)
    )
    if phase_offsets.shape != (4,) or not np.all(np.isfinite(phase_offsets)):
        raise ValueError("phase_offsets_rad must contain four finite ABCD offsets.")
    science = np.asarray(science_whitened_response_per_m, dtype=np.float64)
    if science.ndim != 2 or science.shape[1] != n_dof:
        raise ValueError("science_whitened_response_per_m shape is inconsistent.")
    if not np.all(np.isfinite(science)):
        raise ValueError("science_whitened_response_per_m must be finite.")
    edge_arr = (
        balanced_binary_tree_edges(n_modes)
        if edges is None
        else _as_pair_edges(edges, n_modes=n_modes)
    )
    if n_trials == 0:
        empty = np.zeros(0, dtype=np.float64)
        return CophasingHandoffMonteCarlo(
            input_rms_m=empty,
            acquisition_residual_rms_m=empty,
            science_input_rms_m=empty,
            science_residual_rms_m=empty,
            median_acquisition_residual_rms_m=float("nan"),
            p95_acquisition_residual_rms_m=float("nan"),
            median_science_residual_rms_m=float("nan"),
            p95_science_residual_rms_m=float("nan"),
            median_science_input_rms_m=float("nan"),
            p95_science_input_rms_m=float("nan"),
            acquisition_locked=np.zeros(0, dtype=bool),
            science_locked=np.zeros(0, dtype=bool),
            acquisition_lock_fraction=float("nan"),
            science_lock_fraction=float("nan"),
            edges=edge_arr.astype(np.int64, copy=True),
            edge_wrap_fraction=float("nan"),
            max_edge_opd_over_lambda=float("nan"),
        )

    coord_sigma = (
        float(piston_rms_m) * np.sqrt(float(n_modes) / float(n_dof))
        if n_dof > 0
        else 0.0
    )
    drift_coord_sigma = (
        float(science_drift_rms_m) * np.sqrt(float(n_modes) / float(n_dof))
        if n_dof > 0
        else 0.0
    )
    science_estimator = np.linalg.pinv(science, rcond=float(relative_threshold))
    count_cap = None if count_cap_per_output is None else float(count_cap_per_output)
    edge_weights = _edge_count_weights(
        edge_count_weights,
        n_edges=edge_arr.shape[0],
    )

    input_rms = np.zeros(n_trials, dtype=np.float64)
    acq_residual_rms = np.zeros(n_trials, dtype=np.float64)
    science_input_rms = np.zeros(n_trials, dtype=np.float64)
    science_residual_rms = np.zeros(n_trials, dtype=np.float64)
    acq_locked = np.zeros(n_trials, dtype=bool)
    science_locked = np.zeros(n_trials, dtype=bool)
    wrapped_samples = 0
    total_edge_channel_samples = 0
    max_edge_ratio = 0.0

    for trial in range(int(n_trials)):
        coords_true = rng.normal(0.0, coord_sigma, size=n_dof)
        pistons_true = basis @ coords_true
        edge_ratio = np.abs(
            pistons_true[edge_arr[:, 0]] - pistons_true[edge_arr[:, 1]]
        )[:, np.newaxis] / wavelengths[np.newaxis, :]
        wrapped_samples += int(np.count_nonzero(edge_ratio > 0.5))
        total_edge_channel_samples += int(edge_ratio.size)
        if edge_ratio.size:
            max_edge_ratio = max(max_edge_ratio, float(np.max(edge_ratio)))
        edge_opd, edge_weight = _measure_abcd_pair_edges(
            pistons_m=pistons_true,
            edges=edge_arr,
            wavelengths_m=wavelengths,
            wavelength_groups=groups,
            counts_per_wavelength=(
                counts_by_wavelength
                * float(tracker_fraction)
                * float(edge_throughput)
            ),
            visibility=float(visibility),
            phase_offsets_rad=phase_offsets,
            edge_count_weights=edge_weights,
            dark_counts_per_output=float(dark_counts_per_output),
            count_cap_per_output=count_cap,
            rng=rng,
            shot_noise=bool(shot_noise),
        )
        pistons_hat = _solve_pair_graph_pistons(
            edges=edge_arr,
            edge_opd_m=edge_opd,
            edge_weight=edge_weight,
            piston_basis=basis,
            relative_threshold=float(relative_threshold),
        )
        acquisition_residual = pistons_true - pistons_hat
        acquisition_residual -= float(np.mean(acquisition_residual))

        drift_coords = (
            rng.normal(0.0, drift_coord_sigma, size=n_dof)
            if drift_coord_sigma > 0.0
            else np.zeros(n_dof, dtype=np.float64)
        )
        science_input = acquisition_residual + basis @ drift_coords
        science_input -= float(np.mean(science_input))
        science_coords = basis.T @ science_input
        science_noise = (
            rng.normal(0.0, 1.0, size=science.shape[0])
            if shot_noise
            else np.zeros(science.shape[0], dtype=np.float64)
        )
        science_coords_hat = science_estimator @ (science @ science_coords + science_noise)
        science_residual = science_input - basis @ science_coords_hat

        input_rms[trial] = _rms(pistons_true)
        acq_residual_rms[trial] = _rms(acquisition_residual)
        science_input_rms[trial] = _rms(science_input)
        science_residual_rms[trial] = _rms(science_residual)
        acq_locked[trial] = acq_residual_rms[trial] <= float(lock_threshold_m)
        science_locked[trial] = science_residual_rms[trial] <= float(lock_threshold_m)

    return CophasingHandoffMonteCarlo(
        input_rms_m=input_rms,
        acquisition_residual_rms_m=acq_residual_rms,
        science_input_rms_m=science_input_rms,
        science_residual_rms_m=science_residual_rms,
        median_acquisition_residual_rms_m=float(np.median(acq_residual_rms)),
        p95_acquisition_residual_rms_m=float(np.percentile(acq_residual_rms, 95.0)),
        median_science_residual_rms_m=float(np.median(science_residual_rms)),
        p95_science_residual_rms_m=float(np.percentile(science_residual_rms, 95.0)),
        median_science_input_rms_m=float(np.median(science_input_rms)),
        p95_science_input_rms_m=float(np.percentile(science_input_rms, 95.0)),
        acquisition_locked=acq_locked.astype(bool, copy=True),
        science_locked=science_locked.astype(bool, copy=True),
        acquisition_lock_fraction=float(np.mean(acq_locked)),
        science_lock_fraction=float(np.mean(science_locked)),
        edges=edge_arr.astype(np.int64, copy=True),
        edge_wrap_fraction=(
            float(wrapped_samples) / float(total_edge_channel_samples)
            if total_edge_channel_samples
            else float("nan")
        ),
        max_edge_opd_over_lambda=float(max_edge_ratio),
    )


def _as_pair_edges(
    edges: NDArray[np.integer],
    *,
    n_modes: int,
) -> NDArray[np.int64]:
    arr = np.asarray(edges, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("edges must have shape (n_edges, 2).")
    if arr.shape[0] == 0:
        raise ValueError("edges must contain at least one pair.")
    if np.any(arr < 0) or np.any(arr >= int(n_modes)):
        raise ValueError("edges contain mode indices outside the piston basis.")
    if np.any(arr[:, 0] == arr[:, 1]):
        raise ValueError("edges must connect two different modes.")
    if _graph_rank(arr, n_modes=int(n_modes)) < int(n_modes) - 1:
        raise ValueError("edges must connect all piston modes.")
    return arr.astype(np.int64, copy=True)


def _graph_rank(edges: NDArray[np.integer], *, n_modes: int) -> int:
    incidence = np.zeros((np.asarray(edges).shape[0], int(n_modes)), dtype=np.float64)
    for row, (a, b) in enumerate(np.asarray(edges, dtype=np.int64)):
        incidence[row, int(a)] = 1.0
        incidence[row, int(b)] = -1.0
    return int(np.linalg.matrix_rank(incidence))


def _edge_count_weights(
    weights: NDArray[np.floating] | None,
    *,
    n_edges: int,
) -> NDArray[np.float64]:
    if weights is None:
        return np.full(n_edges, 1.0 / float(n_edges), dtype=np.float64)
    arr = np.asarray(weights, dtype=np.float64)
    if arr.shape != (int(n_edges),):
        raise ValueError("edge_count_weights must have one value per cophasing edge.")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError("edge_count_weights must be finite and non-negative.")
    total = float(np.sum(arr))
    if total <= 0.0:
        raise ValueError("edge_count_weights must have positive sum.")
    return (arr / total).astype(np.float64, copy=False)


def _effective_group_wavelength(
    wavelengths_m: NDArray[np.floating],
    counts: NDArray[np.floating],
) -> float:
    wavelengths = np.asarray(wavelengths_m, dtype=np.float64)
    weights = np.asarray(counts, dtype=np.float64)
    total = float(np.sum(weights))
    if total <= 0.0:
        return float(np.mean(wavelengths))
    return float(np.sum(wavelengths * weights) / total)


def _measure_abcd_pair_edges(
    *,
    pistons_m: NDArray[np.floating],
    edges: NDArray[np.integer],
    wavelengths_m: NDArray[np.floating],
    wavelength_groups: tuple[NDArray[np.integer], ...],
    counts_per_wavelength: NDArray[np.floating],
    visibility: float,
    phase_offsets_rad: NDArray[np.floating],
    edge_count_weights: NDArray[np.floating],
    dark_counts_per_output: float,
    count_cap_per_output: float | None,
    rng: np.random.Generator,
    shot_noise: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Measure pairwise optical-path differences with ideal ABCD combiners."""

    pistons = np.asarray(pistons_m, dtype=np.float64)
    edge_arr = np.asarray(edges, dtype=np.int64)
    wavelengths = np.asarray(wavelengths_m, dtype=np.float64)
    groups = _as_wavelength_groups(wavelength_groups, n_wavelengths=wavelengths.size)
    counts = np.asarray(counts_per_wavelength, dtype=np.float64)
    if counts.shape != wavelengths.shape:
        raise ValueError("counts_per_wavelength must match wavelengths_m.")
    phase_offsets = np.asarray(phase_offsets_rad, dtype=np.float64)
    edge_weights = np.asarray(edge_count_weights, dtype=np.float64)
    edge_opd = np.zeros(edge_arr.shape[0], dtype=np.float64)
    edge_weight = np.zeros(edge_arr.shape[0], dtype=np.float64)

    for edge_i, (a, b) in enumerate(edge_arr):
        estimates = np.zeros(len(groups), dtype=np.float64)
        weights = np.zeros(len(groups), dtype=np.float64)
        opd = float(pistons[int(a)] - pistons[int(b)])
        for group_i, group in enumerate(groups):
            edge_counts_by_wave = (
                counts[group] * float(edge_weights[edge_i])
            )
            phase = _TWO_PI * opd / wavelengths[group]
            means = np.sum(
                0.25
                * edge_counts_by_wave[:, np.newaxis]
                * (
                    1.0
                    + float(visibility)
                    * np.cos(phase[:, np.newaxis] + phase_offsets[np.newaxis, :])
                ),
                axis=0,
            )
            means = np.maximum(means, 0.0) + float(dark_counts_per_output)
            if count_cap_per_output is not None:
                means = np.minimum(means, float(count_cap_per_output))
            observed = rng.poisson(means) if shot_noise else means
            x_quad = float(observed[0] - observed[2])
            y_quad = float(observed[3] - observed[1])
            effective_wavelength = _effective_group_wavelength(
                wavelengths[group],
                edge_counts_by_wave,
            )
            estimates[group_i] = np.arctan2(y_quad, x_quad) * effective_wavelength / _TWO_PI
            weights[group_i] = (
                max(float(np.sum(edge_counts_by_wave)), 0.0)
                * float(visibility) ** 2
                / effective_wavelength**2
            )
        weight_sum = float(np.sum(weights))
        if weight_sum > 0.0:
            edge_opd[edge_i] = float(np.sum(estimates * weights) / weight_sum)
            edge_weight[edge_i] = weight_sum
    return edge_opd, edge_weight


def _solve_pair_graph_pistons(
    *,
    edges: NDArray[np.integer],
    edge_opd_m: NDArray[np.floating],
    edge_weight: NDArray[np.floating],
    piston_basis: NDArray[np.floating],
    relative_threshold: float,
) -> NDArray[np.float64]:
    """Solve common-piston-free aperture pistons from pairwise OPDs."""

    edge_arr = np.asarray(edges, dtype=np.int64)
    basis = np.asarray(piston_basis, dtype=np.float64)
    measurements = np.asarray(edge_opd_m, dtype=np.float64)
    weights = np.asarray(edge_weight, dtype=np.float64)
    incidence = np.zeros((edge_arr.shape[0], basis.shape[0]), dtype=np.float64)
    for row, (a, b) in enumerate(edge_arr):
        incidence[row, int(a)] = 1.0
        incidence[row, int(b)] = -1.0
    valid = np.isfinite(measurements) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return np.zeros(basis.shape[0], dtype=np.float64)
    row_scale = np.sqrt(weights[valid])
    projected = incidence[valid] @ basis
    lhs = projected * row_scale[:, np.newaxis]
    rhs = measurements[valid] * row_scale
    coords = np.linalg.lstsq(lhs, rhs, rcond=float(relative_threshold))[0]
    pistons = basis @ coords
    return (pistons - float(np.mean(pistons))).astype(np.float64, copy=False)


def _active_response_refinement(
    *,
    true_pistons_m: NDArray[np.floating],
    initial_correction_m: NDArray[np.floating],
    active_modes: list[int],
    reference_mode: int,
    piston_basis: NDArray[np.floating],
    field_piston_basis: NDArray[np.floating],
    u_stack: NDArray[np.complexfloating],
    input_fields: NDArray[np.complexfloating],
    wavelengths_m: NDArray[np.floating],
    port_indices: NDArray[np.integer],
    wavelength_indices: NDArray[np.integer],
    wavelength_groups: tuple[NDArray[np.integer], ...] | None,
    counts_per_power: NDArray[np.floating],
    background_counts: NDArray[np.floating],
    background_counts_by_wavelength: NDArray[np.floating] | None,
    zero_total_counts: NDArray[np.floating],
    zero_reference_counts: NDArray[np.floating],
    rng: np.random.Generator,
    shot_noise: bool,
    noise_floor_counts: float,
    n_iterations: int,
    probe_step_m: float,
    max_step_m: float,
    relative_threshold: float,
) -> tuple[NDArray[np.float64], int, float]:
    """Refine a lock by repeatedly measuring a local active response matrix."""

    correction = np.asarray(initial_correction_m, dtype=np.float64).copy()
    zero_total = np.asarray(zero_total_counts, dtype=np.float64)
    zero_reference = np.asarray(zero_reference_counts, dtype=np.float64)
    active = np.asarray(active_modes, dtype=np.intp)
    evaluations = 0
    best_cost = float("inf")
    if active.size == 0:
        return correction, evaluations, best_cost

    for _iteration in range(int(n_iterations)):
        observed = _active_counts_after_correction(
            true_pistons_m=true_pistons_m,
            correction_m=correction,
            piston_basis=piston_basis,
            field_piston_basis=field_piston_basis,
            u_stack=u_stack,
            input_fields=input_fields,
            wavelengths_m=wavelengths_m,
            port_indices=port_indices,
            wavelength_indices=wavelength_indices,
            wavelength_groups=wavelength_groups,
            counts_per_power=counts_per_power,
            background_counts=background_counts,
            background_counts_by_wavelength=background_counts_by_wavelength,
            rng=rng,
            shot_noise=shot_noise,
        )
        evaluations += 1
        noise = np.sqrt(np.maximum(observed, float(noise_floor_counts)))
        target_delta = (zero_reference - observed) / noise
        response = np.zeros((observed.size, active.size), dtype=np.float64)
        for col, mode in enumerate(active):
            plus = correction.copy()
            minus = correction.copy()
            plus[int(mode)] += float(probe_step_m)
            minus[int(mode)] -= float(probe_step_m)
            counts_plus = _active_counts_after_correction(
                true_pistons_m=true_pistons_m,
                correction_m=plus,
                piston_basis=piston_basis,
                field_piston_basis=field_piston_basis,
                u_stack=u_stack,
                input_fields=input_fields,
                wavelengths_m=wavelengths_m,
                port_indices=port_indices,
                wavelength_indices=wavelength_indices,
                wavelength_groups=wavelength_groups,
                counts_per_power=counts_per_power,
                background_counts=background_counts,
                background_counts_by_wavelength=background_counts_by_wavelength,
                rng=rng,
                shot_noise=shot_noise,
            )
            counts_minus = _active_counts_after_correction(
                true_pistons_m=true_pistons_m,
                correction_m=minus,
                piston_basis=piston_basis,
                field_piston_basis=field_piston_basis,
                u_stack=u_stack,
                input_fields=input_fields,
                wavelengths_m=wavelengths_m,
                port_indices=port_indices,
                wavelength_indices=wavelength_indices,
                wavelength_groups=wavelength_groups,
                counts_per_power=counts_per_power,
                background_counts=background_counts,
                background_counts_by_wavelength=background_counts_by_wavelength,
                rng=rng,
                shot_noise=shot_noise,
            )
            evaluations += 2
            response[:, col] = (counts_plus - counts_minus) / (
                2.0 * float(probe_step_m)
            )

        weighted_response = response / noise[:, np.newaxis]
        delta = np.linalg.lstsq(
            weighted_response,
            target_delta,
            rcond=float(relative_threshold),
        )[0]
        step_vector = np.zeros_like(correction)
        step_vector[active] = delta
        step_vector = step_vector - float(np.mean(step_vector))
        step_rms = _rms(step_vector)
        if step_rms > float(max_step_m):
            delta = delta * (float(max_step_m) / step_rms)

        candidate, candidate_cost, line_search_evaluations = (
            _best_active_response_step(
                true_pistons_m=true_pistons_m,
                correction_m=correction,
                delta_active_m=delta,
                active_modes=active,
                reference_mode=int(reference_mode),
                piston_basis=piston_basis,
                field_piston_basis=field_piston_basis,
                u_stack=u_stack,
                input_fields=input_fields,
                wavelengths_m=wavelengths_m,
                port_indices=port_indices,
                wavelength_indices=wavelength_indices,
                wavelength_groups=wavelength_groups,
                counts_per_power=counts_per_power,
                background_counts=background_counts,
                background_counts_by_wavelength=background_counts_by_wavelength,
                zero_total_counts=zero_total,
                zero_reference_counts=zero_reference,
                noise_floor_counts=float(noise_floor_counts),
                rng=rng,
                shot_noise=shot_noise,
            )
        )
        evaluations += int(line_search_evaluations)
        best_cost = min(best_cost, candidate_cost)
        if np.allclose(candidate, correction, rtol=0.0, atol=0.0):
            break
        correction = candidate

    return correction, evaluations, best_cost


def _best_active_response_step(
    *,
    true_pistons_m: NDArray[np.floating],
    correction_m: NDArray[np.floating],
    delta_active_m: NDArray[np.floating],
    active_modes: NDArray[np.integer],
    reference_mode: int,
    piston_basis: NDArray[np.floating],
    field_piston_basis: NDArray[np.floating],
    u_stack: NDArray[np.complexfloating],
    input_fields: NDArray[np.complexfloating],
    wavelengths_m: NDArray[np.floating],
    port_indices: NDArray[np.integer],
    wavelength_indices: NDArray[np.integer],
    wavelength_groups: tuple[NDArray[np.integer], ...] | None,
    counts_per_power: NDArray[np.floating],
    background_counts: NDArray[np.floating],
    background_counts_by_wavelength: NDArray[np.floating] | None,
    zero_total_counts: NDArray[np.floating],
    zero_reference_counts: NDArray[np.floating],
    noise_floor_counts: float,
    rng: np.random.Generator,
    shot_noise: bool,
) -> tuple[NDArray[np.float64], float, int]:
    """Return the best damped step, its measured cost, and frames spent.

    The third element counts the measured-cost evaluations (one tracker
    frame each) so the caller's ``evaluations`` budget includes the line
    search — pre-fix it under-reported by 6 frames per refinement probe,
    ~17% of the acquisition budget at the kilo_keck_6 settings (review
    2026-06-10 Issue 30a).
    """

    correction = np.asarray(correction_m, dtype=np.float64)
    best = correction.copy()
    best_cost = _active_measured_cost(
        true_pistons_m=true_pistons_m,
        correction_m=best,
        piston_basis=piston_basis,
        field_piston_basis=field_piston_basis,
        u_stack=u_stack,
        input_fields=input_fields,
        wavelengths_m=wavelengths_m,
        port_indices=port_indices,
        wavelength_indices=wavelength_indices,
        wavelength_groups=wavelength_groups,
        counts_per_power=counts_per_power,
        background_counts=background_counts,
        background_counts_by_wavelength=background_counts_by_wavelength,
        zero_reference_counts=zero_reference_counts,
        noise_floor_counts=noise_floor_counts,
        rng=rng,
        shot_noise=shot_noise,
    )
    for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
        candidate = correction.copy()
        candidate[np.asarray(active_modes, dtype=np.intp)] += float(scale) * np.asarray(
            delta_active_m,
            dtype=np.float64,
        )
        candidate[int(reference_mode)] = 0.0
        cost = _active_measured_cost(
            true_pistons_m=true_pistons_m,
            correction_m=candidate,
            piston_basis=piston_basis,
            field_piston_basis=field_piston_basis,
            u_stack=u_stack,
            input_fields=input_fields,
            wavelengths_m=wavelengths_m,
            port_indices=port_indices,
            wavelength_indices=wavelength_indices,
            wavelength_groups=wavelength_groups,
            counts_per_power=counts_per_power,
            background_counts=background_counts,
            background_counts_by_wavelength=background_counts_by_wavelength,
            zero_reference_counts=zero_reference_counts,
            noise_floor_counts=noise_floor_counts,
            rng=rng,
            shot_noise=shot_noise,
        )
        if cost < best_cost:
            best = candidate
            best_cost = cost
    return best, float(best_cost), 6


def _active_measured_cost(
    *,
    true_pistons_m: NDArray[np.floating],
    correction_m: NDArray[np.floating],
    piston_basis: NDArray[np.floating],
    field_piston_basis: NDArray[np.floating],
    u_stack: NDArray[np.complexfloating],
    input_fields: NDArray[np.complexfloating],
    wavelengths_m: NDArray[np.floating],
    port_indices: NDArray[np.integer],
    wavelength_indices: NDArray[np.integer],
    wavelength_groups: tuple[NDArray[np.integer], ...] | None,
    counts_per_power: NDArray[np.floating],
    background_counts: NDArray[np.floating],
    background_counts_by_wavelength: NDArray[np.floating] | None,
    zero_reference_counts: NDArray[np.floating],
    noise_floor_counts: float,
    rng: np.random.Generator,
    shot_noise: bool,
) -> float:
    model = _active_counts_after_correction(
        true_pistons_m=true_pistons_m,
        correction_m=correction_m,
        piston_basis=piston_basis,
        field_piston_basis=field_piston_basis,
        u_stack=u_stack,
        input_fields=input_fields,
        wavelengths_m=wavelengths_m,
        port_indices=port_indices,
        wavelength_indices=wavelength_indices,
        wavelength_groups=wavelength_groups,
        counts_per_power=counts_per_power,
        background_counts=background_counts,
        background_counts_by_wavelength=background_counts_by_wavelength,
        rng=rng,
        shot_noise=shot_noise,
    )
    noise = np.sqrt(np.maximum(model, float(noise_floor_counts)))
    weighted = (model - np.asarray(zero_reference_counts, dtype=np.float64)) / noise
    return float(np.dot(weighted, weighted))


def _active_acquisition_cost(
    *,
    true_pistons_m: NDArray[np.floating],
    correction_m: NDArray[np.floating],
    piston_basis: NDArray[np.floating],
    field_piston_basis: NDArray[np.floating],
    u_stack: NDArray[np.complexfloating],
    input_fields: NDArray[np.complexfloating],
    wavelengths_m: NDArray[np.floating],
    port_indices: NDArray[np.integer],
    wavelength_indices: NDArray[np.integer],
    wavelength_groups: tuple[NDArray[np.integer], ...] | None,
    counts_per_power: NDArray[np.floating],
    background_counts: NDArray[np.floating],
    background_counts_by_wavelength: NDArray[np.floating] | None,
    zero_total_counts: NDArray[np.floating],
    zero_reference_counts: NDArray[np.floating],
    cost_noise: NDArray[np.floating],
    rng: np.random.Generator,
    shot_noise: bool,
) -> float:
    """Return zero-piston calibration mismatch after applying a correction."""

    total = _active_counts_after_correction(
        true_pistons_m=true_pistons_m,
        correction_m=correction_m,
        piston_basis=piston_basis,
        field_piston_basis=field_piston_basis,
        u_stack=u_stack,
        input_fields=input_fields,
        wavelengths_m=wavelengths_m,
        port_indices=port_indices,
        wavelength_indices=wavelength_indices,
        wavelength_groups=wavelength_groups,
        counts_per_power=counts_per_power,
        background_counts=background_counts,
        background_counts_by_wavelength=background_counts_by_wavelength,
        rng=rng,
        shot_noise=shot_noise,
    )
    weighted = (
        total - np.asarray(zero_reference_counts, dtype=np.float64)
    ) / np.asarray(cost_noise, dtype=np.float64)
    return float(np.dot(weighted, weighted))


def _active_counts_after_correction(
    *,
    true_pistons_m: NDArray[np.floating],
    correction_m: NDArray[np.floating],
    piston_basis: NDArray[np.floating],
    field_piston_basis: NDArray[np.floating],
    u_stack: NDArray[np.complexfloating],
    input_fields: NDArray[np.complexfloating],
    wavelengths_m: NDArray[np.floating],
    port_indices: NDArray[np.integer],
    wavelength_indices: NDArray[np.integer],
    wavelength_groups: tuple[NDArray[np.integer], ...] | None,
    counts_per_power: NDArray[np.floating],
    background_counts: NDArray[np.floating],
    background_counts_by_wavelength: NDArray[np.floating] | None = None,
    rng: np.random.Generator,
    shot_noise: bool,
) -> NDArray[np.float64]:
    """Return tracker counts after applying a physical correction vector."""

    physical_residual = np.asarray(true_pistons_m, dtype=np.float64) - np.asarray(
        correction_m,
        dtype=np.float64,
    )
    physical_residual = physical_residual - float(np.mean(physical_residual))
    coords_residual = np.asarray(piston_basis, dtype=np.float64).T @ physical_residual
    chip_residual = np.asarray(field_piston_basis, dtype=np.float64) @ coords_residual
    if wavelength_groups is not None:
        powers = port_powers_for_pistons(
            u_stack=u_stack,
            input_fields=input_fields,
            wavelengths_m=wavelengths_m,
            port_indices=port_indices,
            pistons_m=chip_residual,
            wavelength_indices=np.arange(np.asarray(wavelengths_m).size, dtype=np.intp),
        ).reshape(np.asarray(wavelengths_m).size, np.asarray(port_indices).size).T
        signal_by_wave = np.maximum(
            powers * np.asarray(counts_per_power, dtype=np.float64),
            0.0,
        )
        background_by_wave = (
            np.zeros_like(signal_by_wave)
            if background_counts_by_wavelength is None
            else np.asarray(background_counts_by_wavelength, dtype=np.float64)
        )
        total = _sum_spectral_channel_counts(
            signal_by_wavelength=signal_by_wave,
            background_by_wavelength=background_by_wave,
            wavelength_groups=wavelength_groups,
        )
    else:
        powers = port_powers_for_pistons(
            u_stack=u_stack,
            input_fields=input_fields,
            wavelengths_m=wavelengths_m,
            port_indices=port_indices,
            pistons_m=chip_residual,
            wavelength_indices=wavelength_indices,
        )
        signal = np.maximum(
            powers * np.asarray(counts_per_power, dtype=np.float64),
            0.0,
        )
        total = signal + np.asarray(background_counts, dtype=np.float64)
    if shot_noise:
        return rng.poisson(total).astype(np.float64, copy=False)
    return total.astype(np.float64, copy=False)


def _counts_model_and_jacobian(
    *,
    coords: NDArray[np.floating],
    u_stack: NDArray[np.complexfloating],
    input_fields: NDArray[np.complexfloating],
    wavelengths_m: NDArray[np.floating],
    port_indices: NDArray[np.integer],
    wavelength_indices: NDArray[np.integer],
    field_piston_basis: NDArray[np.floating],
    counts_per_power: NDArray[np.floating],
    background_counts: NDArray[np.floating],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return nonlinear model counts and Jacobian with respect to coordinates."""

    basis = np.asarray(field_piston_basis, dtype=np.float64)
    coord = np.asarray(coords, dtype=np.float64)
    pistons = basis @ coord
    response = intensity_response_matrix_at_pistons(
        u_stack=u_stack,
        input_fields=input_fields,
        wavelengths_m=wavelengths_m,
        port_indices=port_indices,
        pistons_m=pistons,
        wavelength_indices=wavelength_indices,
    )
    scale = np.asarray(counts_per_power, dtype=np.float64)
    background = np.asarray(background_counts, dtype=np.float64)
    if scale.shape != response.reference_power.shape:
        raise ValueError("counts_per_power length must match response rows.")
    if background.shape != response.reference_power.shape:
        raise ValueError("background_counts length must match response rows.")
    signal = np.maximum(response.reference_power * scale, 0.0)
    jacobian = (response.response_power_per_m @ basis) * scale[:, np.newaxis]
    return signal + background, jacobian.astype(np.float64, copy=False)


def _spectral_channel_counts_model_and_jacobian(
    *,
    coords: NDArray[np.floating],
    u_stack: NDArray[np.complexfloating],
    input_fields: NDArray[np.complexfloating],
    wavelengths_m: NDArray[np.floating],
    port_indices: NDArray[np.integer],
    wavelength_groups: tuple[NDArray[np.integer], ...],
    field_piston_basis: NDArray[np.floating],
    counts_per_power_by_wavelength: NDArray[np.floating],
    background_counts_by_wavelength: NDArray[np.floating],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return grouped-channel model counts and coordinate Jacobian."""

    basis = np.asarray(field_piston_basis, dtype=np.float64)
    coord = np.asarray(coords, dtype=np.float64)
    pistons = basis @ coord
    response = intensity_response_matrix_at_pistons(
        u_stack=u_stack,
        input_fields=input_fields,
        wavelengths_m=wavelengths_m,
        port_indices=port_indices,
        pistons_m=pistons,
        wavelength_indices=np.arange(np.asarray(wavelengths_m).size, dtype=np.intp),
    )
    n_wavelengths = int(np.asarray(wavelengths_m).size)
    n_ports = int(np.asarray(port_indices).size)
    scale = _as_counts_by_wavelength(
        counts_per_power_by_wavelength,
        n_ports=n_ports,
        n_wavelengths=n_wavelengths,
    )
    background = _as_counts_by_wavelength(
        background_counts_by_wavelength,
        n_ports=n_ports,
        n_wavelengths=n_wavelengths,
    )
    reference = response.reference_power.reshape(n_wavelengths, n_ports).T
    power_response = response.response_power_per_m.reshape(
        n_wavelengths,
        n_ports,
        basis.shape[0],
    ).transpose(1, 0, 2)
    groups = _as_wavelength_groups(wavelength_groups, n_wavelengths=n_wavelengths)

    model = np.zeros(n_ports * len(groups), dtype=np.float64)
    jacobian = np.zeros((n_ports * len(groups), basis.shape[1]), dtype=np.float64)
    for channel_i, group in enumerate(groups):
        rows = slice(channel_i * n_ports, (channel_i + 1) * n_ports)
        model[rows] = np.sum(
            np.maximum(reference[:, group] * scale[:, group], 0.0)
            + background[:, group],
            axis=1,
        )
        response_counts = np.sum(
            power_response[:, group, :] * scale[:, group, np.newaxis],
            axis=1,
        )
        jacobian[rows] = response_counts @ basis
    return model, jacobian.astype(np.float64, copy=False)


def _sum_spectral_channel_counts(
    *,
    signal_by_wavelength: NDArray[np.floating],
    background_by_wavelength: NDArray[np.floating],
    wavelength_groups: tuple[NDArray[np.integer], ...],
) -> NDArray[np.float64]:
    signal = np.asarray(signal_by_wavelength, dtype=np.float64)
    background = np.asarray(background_by_wavelength, dtype=np.float64)
    if signal.ndim != 2 or background.shape != signal.shape:
        raise ValueError(
            "signal_by_wavelength and background_by_wavelength must have "
            "matching 2-D shapes."
        )
    groups = _as_wavelength_groups(wavelength_groups, n_wavelengths=signal.shape[1])
    total = np.zeros(signal.shape[0] * len(groups), dtype=np.float64)
    for channel_i, group in enumerate(groups):
        rows = slice(channel_i * signal.shape[0], (channel_i + 1) * signal.shape[0])
        total[rows] = np.sum(signal[:, group] + background[:, group], axis=1)
    return total


def _calibrated_zero_reference(
    *,
    zero_total_counts: NDArray[np.floating],
    rng: np.random.Generator,
    n_frames: int,
    shot_noise: bool,
) -> NDArray[np.float64]:
    """Return a measured zero-piston reference averaged over calibration frames."""

    zero = np.asarray(zero_total_counts, dtype=np.float64)
    if int(n_frames) <= 0:
        raise ValueError("n_frames must be positive.")
    if not shot_noise:
        return zero.astype(np.float64, copy=True)
    accum = np.zeros_like(zero, dtype=np.float64)
    for _frame in range(int(n_frames)):
        accum += rng.poisson(zero).astype(np.float64, copy=False)
    return accum / float(n_frames)


def _as_unitary_stack(
    value: NDArray[np.complexfloating],
) -> NDArray[np.complex128]:
    arr = np.asarray(value, dtype=np.complex128)
    if arr.ndim != 3:
        raise ValueError(
            "u_stack must have shape (n_ports, n_modes, n_wavelength); "
            f"got {arr.shape}."
        )
    if arr.shape[0] == 0 or arr.shape[1] == 0 or arr.shape[2] == 0:
        raise ValueError("u_stack axes must all be non-empty.")
    return arr


def _as_input_fields(
    value: NDArray[np.complexfloating],
    n_modes: int,
    n_wavelengths: int,
) -> NDArray[np.complex128]:
    arr = np.asarray(value, dtype=np.complex128)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    if arr.ndim != 3 or arr.shape[0] != n_modes or arr.shape[1] != n_wavelengths:
        raise ValueError(
            "input_fields must have shape (n_modes, n_wavelength, n_pol); "
            f"got {arr.shape}, expected first axes ({n_modes}, {n_wavelengths})."
        )
    if arr.shape[2] == 0:
        raise ValueError("input_fields must include at least one polarization channel.")
    return arr


def _as_wavelengths(
    value: NDArray[np.floating],
    n_wavelengths: int,
) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (n_wavelengths,):
        raise ValueError(
            f"wavelengths_m must have shape ({n_wavelengths},), got {arr.shape}."
        )
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError("wavelengths_m must contain finite positive values.")
    return arr


def _as_indices(
    value: NDArray[np.integer],
    *,
    upper: int,
    name: str,
) -> NDArray[np.intp]:
    arr = np.asarray(value, dtype=np.intp)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D index array.")
    if arr.size and (np.any(arr < 0) or np.any(arr >= upper)):
        raise ValueError(f"{name} contains indices outside [0, {upper}).")
    return arr


def _as_wavelength_groups(
    value: tuple[NDArray[np.integer], ...],
    *,
    n_wavelengths: int,
) -> tuple[NDArray[np.intp], ...]:
    groups: list[NDArray[np.intp]] = []
    for group in value:
        arr = _as_indices(group, upper=n_wavelengths, name="wavelength_groups")
        if arr.size == 0:
            raise ValueError("wavelength_groups must not contain empty groups.")
        groups.append(arr)
    if not groups:
        raise ValueError("wavelength_groups must contain at least one group.")
    return tuple(groups)


def _as_counts_by_wavelength(
    value: NDArray[np.floating],
    *,
    n_ports: int,
    n_wavelengths: int,
) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (int(n_ports), int(n_wavelengths)):
        raise ValueError(
            "counts-by-wavelength array must have shape "
            f"({int(n_ports)}, {int(n_wavelengths)}), got {arr.shape}."
        )
    if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError("counts-by-wavelength values must be finite and non-negative.")
    return arr


def _counts_per_power_by_wavelength(
    *,
    u_stack: NDArray[np.complexfloating],
    input_fields: NDArray[np.complexfloating],
    wavelengths_m: NDArray[np.floating],
    port_indices: NDArray[np.integer],
    expected_counts_by_wavelength: NDArray[np.floating],
) -> NDArray[np.float64]:
    u_arr = _as_unitary_stack(u_stack)
    ports = _as_indices(port_indices, upper=u_arr.shape[0], name="port_indices")
    reference = port_powers_for_pistons(
        u_stack=u_arr,
        input_fields=input_fields,
        wavelengths_m=wavelengths_m,
        port_indices=ports,
        pistons_m=np.zeros(u_arr.shape[1], dtype=np.float64),
        wavelength_indices=np.arange(u_arr.shape[2], dtype=np.intp),
    ).reshape(u_arr.shape[2], ports.size).T
    counts = _as_counts_by_wavelength(
        expected_counts_by_wavelength,
        n_ports=ports.size,
        n_wavelengths=u_arr.shape[2],
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(reference > 0.0, counts / reference, 0.0)
    return np.where(np.isfinite(scale), scale, 0.0).astype(np.float64, copy=False)


def _rms(values: NDArray[np.floating]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(arr * arr)))


__all__ = [
    "IntensityResponse",
    "SpectralChannelCountResponse",
    "FractionalIntensityResponse",
    "SpectralChannelFractionalIntensityResponse",
    "ObservabilitySVD",
    "CramerRaoBound",
    "ReconstructionMonteCarlo",
    "GaussNewtonMonteCarlo",
    "ActiveAcquisitionMonteCarlo",
    "CophasingHandoffMonteCarlo",
    "PistonMeasurementCovariance",
    "active_coordinate_acquisition_monte_carlo",
    "balanced_binary_tree_edges",
    "cophasing_handoff_monte_carlo",
    "cramer_rao_bound_from_svd",
    "differential_piston_basis",
    "gauss_newton_reconstruction_monte_carlo",
    "fractional_intensity_response_matrix",
    "intensity_response_matrix",
    "intensity_response_matrix_at_pistons",
    "linearized_reconstruction_monte_carlo",
    "piston_measurement_covariance_from_svd",
    "port_powers_for_pistons",
    "reconstruction_monte_carlo",
    "scale_response_to_counts",
    "spectral_channel_count_response_matrix",
    "spectral_channel_fractional_intensity_response_matrix",
    "spectral_channel_counts_for_pistons",
    "svd_observability",
    "weakest_piston_modes",
    "whiten_response_matrix",
]
