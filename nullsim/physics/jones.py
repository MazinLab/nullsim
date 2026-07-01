"""Jones-matrix polarization primitives for the nullsim physics layer.

This module provides pure Jones algebra with no pipeline or ``SimulationState``
dependencies. Jones vectors use the horizontal/vertical basis ``[E_x, E_y]``
and the optical time convention ``exp(-i omega t)``. Stokes vectors are
returned in ``[I, Q, U, V]`` order, with positive ``V`` for the Jones state
``[1, i] / sqrt(2)``.

The KILO/NAYRA-facing helpers cover the warm dual-polarization chain:
achromatic HWP retardance, imperfect PBS arms, unpolarized PBS splitting,
and unitary PM-fiber Jones matrices with a short-fiber PER floor.

References:
    SPEC §15.2.
    Collett, E. Field Guide to Polarization (2005), Eqs. 4.1-4.12.
    Born, M. and Wolf, E. Principles of Optics, §1.4.2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class PBSConfig:
    """Polarizing beam-splitter hardware parameters.

    Attributes:
        insertion_loss_db: Bulk insertion loss per arm in positive dB.
        extinction_ratio_db: Power rejection of the orthogonal polarization in
            dB. ``30`` means ``1e-3`` fractional leakage power.
        axis_offset_rad: Mechanical offset of the PBS transmission axis from
            the instrument +x frame in radians.
    """

    insertion_loss_db: float = 0.5
    extinction_ratio_db: float = 30.0
    axis_offset_rad: float = 1.0e-3


@dataclass(frozen=True)
class HWPConfig:
    """Achromatic half-wave-plate hardware parameters.

    Attributes:
        rotation_rate_hz: Mechanical HWP rotation rate. NAYRA's warm front end
            uses kHz-class continuous modulation.
        design_wavelength_m: Center wavelength for the achromatic HWP design.
        retardance_error_rms_rad: RMS deviation from pi retardance across the
            configured fractional band half-width.
        rotation_jitter_rad: Per-cycle mechanical angle jitter in radians.
        band_half_width_lambda: Fractional wavelength half-width used to
            calibrate the parabolic retardance-error model.
    """

    rotation_rate_hz: float = 1000.0
    design_wavelength_m: float = 1.25e-6
    retardance_error_rms_rad: float = 0.02
    rotation_jitter_rad: float = 1.0e-4
    band_half_width_lambda: float = 0.25


@dataclass(frozen=True)
class PMFiberConfig:
    """Polarization-maintaining PM980-class fiber parameters.

    Attributes:
        per_db_per_100m: Datasheet polarization extinction ratio over a 100 m
            reference length in dB.
        per_db_floor: Intrinsic short-fiber PER floor in dB. Short PM980-XP
            bench runs sit at this stress-rod floor rather than improving
            indefinitely as length approaches zero.
        slow_axis_alignment_rms_rad: RMS packaging misalignment between the PBS
            output axis and PM-fiber slow axis.
        pmd_coefficient_ps_sqrt_km: Diagnostic PMD coefficient.
        beat_length_m_at_980nm: Diagnostic PM980-XP beat length.
    """

    per_db_per_100m: float = 30.0
    per_db_floor: float = 30.0
    slow_axis_alignment_rms_rad: float = 17.0e-3
    pmd_coefficient_ps_sqrt_km: float = 0.05
    beat_length_m_at_980nm: float = 2.7e-3


@dataclass(frozen=True)
class DualPolField:
    """Pair of complex scalar fields after PBS projection.

    Arm A carries the +x PBS output and arm B carries the +y PBS output.
    Both arms must have identical shapes; power is summed incoherently across
    arms.
    """

    arm_a: ArrayLike
    arm_b: ArrayLike

    def __post_init__(self) -> None:
        arm_a = _as_finite_complex_array(self.arm_a, "arm_a")
        arm_b = _as_finite_complex_array(self.arm_b, "arm_b")
        if arm_a.shape != arm_b.shape:
            raise ValueError(
                f"arm_a shape {arm_a.shape} does not match arm_b shape {arm_b.shape}."
            )
        object.__setattr__(self, "arm_a", arm_a)
        object.__setattr__(self, "arm_b", arm_b)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape shared by both polarization arms."""
        return self.arm_a.shape

    @property
    def total_power(self) -> float:
        """Return the incoherent ``sum(|E_A|² + |E_B|²)`` power."""
        return float(np.sum(np.abs(self.arm_a) ** 2) + np.sum(np.abs(self.arm_b) ** 2))


def linear_polarizer(angle_rad: float) -> NDArray[np.complex128]:
    """Return an ideal linear polarizer with transmission axis ``angle_rad``.

    Args:
        angle_rad: Transmission-axis angle in radians, measured from +x
            toward +y. Must be a finite real scalar.

    Returns:
        A 2x2 complex128 Jones projection matrix.

    Raises:
        ValueError: If ``angle_rad`` is not a finite real scalar.
    """
    angle = _as_finite_real_scalar(angle_rad, "angle_rad")
    axis = np.array([np.cos(angle), np.sin(angle)], dtype=np.complex128)
    return np.outer(axis, np.conjugate(axis)).astype(np.complex128, copy=False)


def waveplate(retardance_rad: float, fast_axis_rad: float) -> NDArray[np.complex128]:
    """Return an ideal linear retarder at an arbitrary fast-axis angle.

    The arbitrary-axis plate is constructed as ``R(theta) M0 R(-theta)``, where
    ``M0 = diag(1, exp(i * retardance_rad))``.

    Args:
        retardance_rad: Retardance in radians. Must be a finite real scalar.
        fast_axis_rad: Fast-axis angle in radians, measured from +x toward +y.
            Must be a finite real scalar.

    Returns:
        A 2x2 complex128 Jones matrix.

    Raises:
        ValueError: If either angle is not a finite real scalar.
    """
    retardance = _as_finite_real_scalar(retardance_rad, "retardance_rad")
    fast_axis = _as_finite_real_scalar(fast_axis_rad, "fast_axis_rad")

    rotation = _rotation_matrix(fast_axis)
    inverse_rotation = _rotation_matrix(-fast_axis)
    eigenbasis_plate = np.diag(
        np.array([1.0 + 0.0j, np.exp(1j * retardance)], dtype=np.complex128)
    )
    return np.asarray(rotation @ eigenbasis_plate @ inverse_rotation, dtype=np.complex128)


def half_wave_plate(fast_axis_rad: float) -> NDArray[np.complex128]:
    """Return an ideal half-wave plate at ``fast_axis_rad``.

    Args:
        fast_axis_rad: Fast-axis angle in radians. Must be a finite real scalar.

    Returns:
        A 2x2 complex128 Jones matrix with retardance pi.

    Raises:
        ValueError: If ``fast_axis_rad`` is not a finite real scalar.
    """
    return waveplate(np.pi, fast_axis_rad)


def quarter_wave_plate(fast_axis_rad: float) -> NDArray[np.complex128]:
    """Return an ideal quarter-wave plate at ``fast_axis_rad``.

    Args:
        fast_axis_rad: Fast-axis angle in radians. Must be a finite real scalar.

    Returns:
        A 2x2 complex128 Jones matrix with retardance pi/2.

    Raises:
        ValueError: If ``fast_axis_rad`` is not a finite real scalar.
    """
    return waveplate(np.pi / 2.0, fast_axis_rad)


def jones_linear_retarder(
    theta_rad: float,
    retardance_rad: float,
) -> NDArray[np.complex128]:
    """Return a linear retarder using KILO/NAYRA argument order.

    Args:
        theta_rad: Fast-axis angle in radians, measured from +x toward +y.
        retardance_rad: Retardance in radians.

    Returns:
        A 2x2 complex128 Jones matrix. The global phase follows
        :func:`waveplate`; source-observable HWP/PBS behavior is unchanged.
    """
    return waveplate(retardance_rad, theta_rad)


def jones_hwp(
    theta_rad: float,
    retardance_rad: float = np.pi,
) -> NDArray[np.complex128]:
    """Return a half-wave plate using KILO/NAYRA argument order.

    Args:
        theta_rad: Fast-axis angle in radians.
        retardance_rad: Retardance in radians. Defaults to an ideal pi HWP.

    Returns:
        A 2x2 complex128 Jones matrix.
    """
    return jones_linear_retarder(theta_rad, retardance_rad)


def hwp_retardance(
    wavelength_m: float | ArrayLike,
    design_wavelength_m: float = 1.25e-6,
    retardance_error_rms_rad: float = 0.02,
    band_half_width_lambda: float = 0.25,
) -> float | NDArray[np.float64]:
    """Return achromatic-HWP retardance versus wavelength.

    The parabolic deviation is calibrated so the RMS of ``retardance - pi``
    over uniform sampling of
    ``design_wavelength_m * (1 ± band_half_width_lambda)`` equals
    ``retardance_error_rms_rad``.

    Args:
        wavelength_m: Positive finite wavelength or array of wavelengths in m.
        design_wavelength_m: Positive finite HWP design wavelength in m.
        retardance_error_rms_rad: Non-negative finite RMS retardance error.
        band_half_width_lambda: Positive finite fractional band half-width.

    Returns:
        Retardance in radians, as a float for scalar input or float64 array for
        array input.
    """
    wavelengths = _as_positive_real_array(wavelength_m, "wavelength_m")
    design_wavelength = _as_positive_real_scalar(
        design_wavelength_m, "design_wavelength_m"
    )
    retardance_error_rms = _as_nonnegative_real_scalar(
        retardance_error_rms_rad, "retardance_error_rms_rad"
    )
    band_half_width = _as_positive_real_scalar(
        band_half_width_lambda, "band_half_width_lambda"
    )

    detuning = wavelengths / design_wavelength - 1.0
    edge_offset = np.sqrt(5.0) * retardance_error_rms
    retardance = np.pi + edge_offset * (detuning / band_half_width) ** 2
    if np.ndim(wavelength_m) == 0:
        return float(np.asarray(retardance))
    return np.asarray(retardance, dtype=np.float64)


def rotator(angle_rad: float) -> NDArray[np.complex128]:
    """Return the real active Jones-basis rotation matrix.

    Args:
        angle_rad: Rotation angle in radians. Must be a finite real scalar.

    Returns:
        A 2x2 complex128 rotation matrix.

    Raises:
        ValueError: If ``angle_rad`` is not a finite real scalar.
    """
    angle = _as_finite_real_scalar(angle_rad, "angle_rad")
    return _rotation_matrix(angle)


def identity_jones(n_modes: int) -> NDArray[np.complex128]:
    """Return one complex128 identity Jones matrix per mode.

    Args:
        n_modes: Positive number of modes/fibers.

    Returns:
        Array with shape ``(n_modes, 2, 2)``.
    """
    n_validated = _as_positive_int(n_modes, "n_modes")
    return np.broadcast_to(
        np.eye(2, dtype=np.complex128),
        (n_validated, 2, 2),
    ).copy()


def random_birefringence(
    n_modes: int,
    retardance_rms_rad: float = 0.1,
    seed: int | None = None,
    wavelength_m: float | None = None,
    reference_wavelength_m: float = 1.55e-6,
) -> NDArray[np.complex128]:
    """Return random stress-birefringence Jones matrices.

    Args:
        n_modes: Positive number of modes/fibers.
        retardance_rms_rad: Non-negative RMS retardance in radians, evaluated
            at ``reference_wavelength_m`` when ``wavelength_m`` is provided.
        seed: Optional RNG seed.
        wavelength_m: Optional operating wavelength in m. Retardance scales as
            ``reference_wavelength_m / wavelength_m`` for a fixed stress OPD.
        reference_wavelength_m: Positive finite reference wavelength in m.

    Returns:
        Complex array with shape ``(n_modes, 2, 2)``.
    """
    n_validated = _as_positive_int(n_modes, "n_modes")
    retardance_rms = _as_nonnegative_real_scalar(
        retardance_rms_rad, "retardance_rms_rad"
    )
    if wavelength_m is not None:
        wavelength = _as_positive_real_scalar(wavelength_m, "wavelength_m")
        reference_wavelength = _as_positive_real_scalar(
            reference_wavelength_m, "reference_wavelength_m"
        )
        retardance_rms *= reference_wavelength / wavelength

    rng = np.random.default_rng(seed)
    jones = np.empty((n_validated, 2, 2), dtype=np.complex128)
    for mode_index in range(n_validated):
        retardance = rng.normal(0.0, retardance_rms)
        fast_axis = rng.uniform(0.0, np.pi)
        jones[mode_index] = waveplate(retardance, fast_axis)
    return jones


def polarizing_beam_splitter_arm(
    arm: str,
    extinction_ratio_db: float = 30.0,
    axis_offset_rad: float = 0.0,
    insertion_loss_db: float = 0.5,
) -> NDArray[np.complex128]:
    """Return the Jones matrix for one imperfect PBS arm.

    Args:
        arm: ``"A"`` for the +x transmitted arm, or ``"B"`` for the +y arm.
        extinction_ratio_db: Non-negative power rejection in dB. ``inf`` is a
            perfect PBS.
        axis_offset_rad: Finite axis offset from the instrument frame.
        insertion_loss_db: Non-negative bulk insertion loss in dB.

    Returns:
        A 2x2 complex128 Jones matrix.
    """
    if arm not in ("A", "B"):
        raise ValueError(f"arm must be 'A' or 'B', got {arm!r}.")
    extinction_ratio = _as_nonnegative_db_or_inf(
        extinction_ratio_db, "extinction_ratio_db"
    )
    axis_offset = _as_finite_real_scalar(axis_offset_rad, "axis_offset_rad")
    insertion_loss = _as_nonnegative_real_scalar(
        insertion_loss_db, "insertion_loss_db"
    )

    leakage = 0.0 if np.isinf(extinction_ratio) else 10.0 ** (-extinction_ratio / 10.0)
    bulk_amplitude = 10.0 ** (-insertion_loss / 20.0)
    pass_amplitude = bulk_amplitude * np.sqrt(max(1.0 - leakage, 0.0))
    blocked_amplitude = bulk_amplitude * np.sqrt(leakage)

    if arm == "A":
        matrix = np.array(
            [[pass_amplitude, 0.0], [0.0, blocked_amplitude]],
            dtype=np.complex128,
        )
    else:
        matrix = np.array(
            [[blocked_amplitude, 0.0], [0.0, pass_amplitude]],
            dtype=np.complex128,
        )

    if axis_offset != 0.0:
        rotation = _rotation_matrix(axis_offset)
        matrix = rotation @ matrix @ rotation.T
    return np.asarray(matrix, dtype=np.complex128)


def pm_fiber_leakage_power(
    fiber_length_m: float,
    per_db_per_100m: float = 30.0,
    per_db_floor: float = 30.0,
) -> float:
    """Return PM-fiber cross-polarized leakage power.

    Leakage accumulates linearly with fiber length from the 100 m datasheet
    value, but short fibers cannot beat the intrinsic stress-rod PER floor.
    The result is capped at ``0.5`` for the fully randomized long-fiber limit.
    """
    length = _as_nonnegative_real_scalar(fiber_length_m, "fiber_length_m")
    per_100m = _as_nonnegative_real_scalar(per_db_per_100m, "per_db_per_100m")
    per_floor = _as_nonnegative_real_scalar(per_db_floor, "per_db_floor")

    leakage_at_100m = 10.0 ** (-per_100m / 10.0)
    accumulated_leakage = leakage_at_100m * (length / 100.0)
    floor_leakage = 10.0 ** (-per_floor / 10.0)
    return float(min(max(accumulated_leakage, floor_leakage), 0.5))


def pm_fiber_per_at_length(
    pm_fiber_config: PMFiberConfig,
    fiber_length_m: float,
) -> float:
    """Return PM-fiber leakage power from a :class:`PMFiberConfig`."""
    return pm_fiber_leakage_power(
        fiber_length_m,
        per_db_per_100m=pm_fiber_config.per_db_per_100m,
        per_db_floor=pm_fiber_config.per_db_floor,
    )


def jones_pm_fiber(
    n_modes: int,
    pm_fiber_config: PMFiberConfig | None = None,
    fiber_length_m: float = 30.0,
    seed: int | None = None,
) -> NDArray[np.complex128]:
    """Return unitary Jones matrices for a bundle of PM fibers.

    Each fiber gets a random retardance phase and a random slow-axis
    misalignment. The random retardance prevents the alignment rotation from
    commuting away, which is the NAYRA correction to the older symmetric KILO
    coupling model.
    """
    n_validated = _as_positive_int(n_modes, "n_modes")
    config = pm_fiber_config if pm_fiber_config is not None else PMFiberConfig()
    alignment_rms = _as_nonnegative_real_scalar(
        config.slow_axis_alignment_rms_rad, "slow_axis_alignment_rms_rad"
    )
    leakage = pm_fiber_per_at_length(config, fiber_length_m)
    leakage_amplitude = np.sqrt(leakage)
    pass_amplitude = np.sqrt(1.0 - leakage)

    rng = np.random.default_rng(seed)
    jones = np.empty((n_validated, 2, 2), dtype=np.complex128)
    for mode_index in range(n_validated):
        retardance = rng.uniform(-np.pi, np.pi)
        fiber_frame = np.array(
            [
                [pass_amplitude * np.exp(0.5j * retardance), -leakage_amplitude],
                [leakage_amplitude, pass_amplitude * np.exp(-0.5j * retardance)],
            ],
            dtype=np.complex128,
        )
        slow_axis_angle = rng.normal(0.0, alignment_rms)
        rotation = _rotation_matrix(slow_axis_angle)
        jones[mode_index] = rotation @ fiber_frame @ rotation.T
    return jones


def apply_pbs(
    field: ArrayLike,
    pbs_config: PBSConfig | None = None,
) -> DualPolField:
    """Split an unpolarized scalar field equally into two PBS arms.

    For unpolarized light, PBS extinction and axis offset do not change the
    arm powers; only bulk insertion loss enters. Polarized inputs should be
    propagated with :func:`polarizing_beam_splitter_arm` instead.
    """
    config = pbs_config if pbs_config is not None else PBSConfig()
    field_array = _as_finite_complex_array(field, "field")
    insertion_loss = _as_nonnegative_real_scalar(
        config.insertion_loss_db, "insertion_loss_db"
    )
    half_amplitude = np.sqrt(10.0 ** (-insertion_loss / 10.0) / 2.0)
    return DualPolField(
        arm_a=field_array * half_amplitude,
        arm_b=field_array * half_amplitude,
    )


def apply_hwp(
    field_or_dual_pol: ArrayLike | DualPolField,
    hwp_angle_rad: float,
    hwp_config: HWPConfig | None = None,
    wavelength_m: float | None = None,
) -> NDArray[np.complex128] | DualPolField:
    """Apply an HWP to a dual-pol field, or pass unpolarized scalar input.

    A scalar array is treated as unpolarized light upstream of the PBS, so HWP
    rotation is a no-op and a complex128 copy is returned. A
    :class:`DualPolField` is treated as a Jones vector pair and rotated by the
    HWP matrix.
    """
    angle = _as_finite_real_scalar(hwp_angle_rad, "hwp_angle_rad")
    config = hwp_config if hwp_config is not None else HWPConfig()
    if not isinstance(field_or_dual_pol, DualPolField):
        return _as_finite_complex_array(field_or_dual_pol, "field_or_dual_pol")

    if wavelength_m is None:
        retardance = np.pi
    else:
        retardance = float(
            hwp_retardance(
                wavelength_m,
                design_wavelength_m=config.design_wavelength_m,
                retardance_error_rms_rad=config.retardance_error_rms_rad,
                band_half_width_lambda=config.band_half_width_lambda,
            )
        )
    matrix = jones_hwp(angle, retardance)
    return DualPolField(
        arm_a=matrix[0, 0] * field_or_dual_pol.arm_a
        + matrix[0, 1] * field_or_dual_pol.arm_b,
        arm_b=matrix[1, 0] * field_or_dual_pol.arm_a
        + matrix[1, 1] * field_or_dual_pol.arm_b,
    )


def pdi_imperfection_factor(
    pbs_config: PBSConfig | None = None,
    hwp_config: HWPConfig | None = None,
    pm_fiber_config: PMFiberConfig | None = None,
    pm_fiber_length_m: float = 30.0,
    sigma_i_per_arm: float = 0.0,
) -> float:
    """Return the PDI imperfection coefficient ``epsilon``.

    The quadrature terms are PBS axis error, HWP retardance-error amplitude
    leakage, HWP rotation jitter, PBS extinction leakage, optional PM-fiber
    leakage, and non-common-mode per-arm scintillation.
    """
    pbs = pbs_config if pbs_config is not None else PBSConfig()
    hwp = hwp_config if hwp_config is not None else HWPConfig()
    axis_offset = _as_nonnegative_real_scalar(pbs.axis_offset_rad, "axis_offset_rad")
    hwp_error = _as_nonnegative_real_scalar(
        hwp.retardance_error_rms_rad, "retardance_error_rms_rad"
    )
    hwp_jitter = _as_nonnegative_real_scalar(
        hwp.rotation_jitter_rad, "rotation_jitter_rad"
    )
    sigma_i = _as_nonnegative_real_scalar(sigma_i_per_arm, "sigma_i_per_arm")
    extinction_ratio = _as_nonnegative_db_or_inf(
        pbs.extinction_ratio_db, "extinction_ratio_db"
    )
    pbs_leakage_power = (
        0.0 if np.isinf(extinction_ratio) else 10.0 ** (-extinction_ratio / 10.0)
    )
    epsilon_squared = (
        (2.0 * axis_offset) ** 2
        + (0.5 * hwp_error) ** 2
        + (2.0 * hwp_jitter) ** 2
        + sigma_i**2
        + pbs_leakage_power
    )
    if pm_fiber_config is not None:
        epsilon_squared += pm_fiber_per_at_length(
            pm_fiber_config, pm_fiber_length_m
        )
    return float(np.sqrt(epsilon_squared))


def apply_jones(M: ArrayLike, E: ArrayLike) -> NDArray[np.complex128]:
    """Apply a Jones matrix to one or more Jones vectors.

    Args:
        M: Jones matrix with shape ``(2, 2)``. Values must be finite.
        E: Jones vector array with shape ``(2,)`` or ``(..., 2)``. Values must
            be finite.

    Returns:
        The transformed Jones vector array as complex128 with the same shape
        as ``E``.

    Raises:
        ValueError: If ``M`` is not a finite ``(2, 2)`` matrix, or if ``E`` does
            not have last-axis length 2 or contains non-finite values.
    """
    matrix = _as_jones_matrix(M, "M")
    field = _as_jones_vectors(E, "E")
    return np.einsum("ij,...j->...i", matrix, field, optimize=True).astype(
        np.complex128,
        copy=False,
    )


def jones_to_stokes(E: ArrayLike) -> NDArray[np.float64]:
    """Convert one or more Jones vectors to Stokes ``[I, Q, U, V]`` vectors.

    Args:
        E: Jones vector array with shape ``(2,)`` or ``(..., 2)``. Values must
            be finite.

    Returns:
        Stokes vectors with shape ``(4,)`` or ``(..., 4)`` and dtype float64.

    Raises:
        ValueError: If ``E`` does not have last-axis length 2 or contains
            non-finite values.
    """
    field = _as_jones_vectors(E, "E")
    horizontal = field[..., 0]
    vertical = field[..., 1]
    cross_term = np.conjugate(horizontal) * vertical

    intensity = np.abs(horizontal) ** 2 + np.abs(vertical) ** 2
    q_linear = np.abs(horizontal) ** 2 - np.abs(vertical) ** 2
    u_linear = 2.0 * np.real(cross_term)
    v_circular = 2.0 * np.imag(cross_term)

    return np.stack(
        [intensity, q_linear, u_linear, v_circular],
        axis=-1,
    ).astype(np.float64, copy=False)


def _as_finite_real_scalar(value: float, name: str) -> np.float64:
    """Return ``value`` as a finite float64 scalar."""
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in ("i", "u", "f"):
        raise ValueError(f"{name} must be a finite real scalar.")

    scalar = np.float64(array)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite.")
    return scalar


def _as_nonnegative_real_scalar(value: float, name: str) -> np.float64:
    """Return ``value`` as a non-negative finite float64 scalar."""
    scalar = _as_finite_real_scalar(value, name)
    if scalar < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return scalar


def _as_positive_real_scalar(value: float, name: str) -> np.float64:
    """Return ``value`` as a positive finite float64 scalar."""
    scalar = _as_finite_real_scalar(value, name)
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return scalar


def _as_nonnegative_db_or_inf(value: float, name: str) -> np.float64:
    """Return a non-negative dB scalar, allowing positive infinity."""
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in ("i", "u", "f"):
        raise ValueError(f"{name} must be a non-negative real scalar.")
    scalar = np.float64(array)
    if np.isnan(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return scalar


def _as_positive_real_array(value: ArrayLike, name: str) -> NDArray[np.float64]:
    """Return ``value`` as positive finite float64 array."""
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric and convertible to float64.") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(array <= 0.0):
        raise ValueError(f"{name} must be positive.")
    return array


def _as_positive_int(value: int, name: str) -> int:
    """Return ``value`` as a positive integer."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if integer != value or integer <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return integer


def _as_finite_complex_array(value: ArrayLike, name: str) -> NDArray[np.complex128]:
    """Validate and coerce a finite complex array."""
    try:
        array = np.asarray(value, dtype=np.complex128)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric and convertible to complex128.") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _rotation_matrix(angle_rad: np.float64) -> NDArray[np.complex128]:
    """Return the active rotation matrix for a validated angle."""
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    return np.array(
        [[cos_angle, -sin_angle], [sin_angle, cos_angle]],
        dtype=np.complex128,
    )


def _as_jones_matrix(value: ArrayLike, name: str) -> NDArray[np.complex128]:
    """Validate and coerce a finite Jones matrix."""
    try:
        matrix = np.asarray(value, dtype=np.complex128)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric and convertible to complex128.") from exc

    if matrix.shape != (2, 2):
        raise ValueError(f"{name} must have shape (2, 2).")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _as_jones_vectors(value: ArrayLike, name: str) -> NDArray[np.complex128]:
    """Validate and coerce one or more finite Jones vectors."""
    try:
        vectors = np.asarray(value, dtype=np.complex128)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric and convertible to complex128.") from exc

    if vectors.ndim == 0 or vectors.shape[-1] != 2:
        raise ValueError(f"{name} must have shape (2,) or (..., 2).")
    if not np.all(np.isfinite(vectors)):
        raise ValueError(f"{name} must contain only finite values.")
    return vectors


__all__ = [
    "DualPolField",
    "HWPConfig",
    "PBSConfig",
    "PMFiberConfig",
    "apply_hwp",
    "apply_jones",
    "apply_pbs",
    "half_wave_plate",
    "hwp_retardance",
    "identity_jones",
    "jones_hwp",
    "jones_identity",
    "jones_linear_retarder",
    "jones_pbs_arm",
    "jones_pm_fiber",
    "jones_random_birefringence",
    "jones_to_stokes",
    "linear_polarizer",
    "pdi_imperfection_factor",
    "pm_fiber_leakage_power",
    "pm_fiber_per_at_length",
    "polarizing_beam_splitter_arm",
    "quarter_wave_plate",
    "random_birefringence",
    "rotator",
    "waveplate",
]

jones_identity = identity_jones
jones_random_birefringence = random_birefringence
jones_pbs_arm = polarizing_beam_splitter_arm
