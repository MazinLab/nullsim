"""Import built-in stages so registration decorators run.

Core (always-required) stages are imported unconditionally. Optional
backends that depend on extras (torch) are imported in a guarded
loop so a missing extra does not break package autoload — users without
the extra installed simply do not see those stage classes registered.
"""

from __future__ import annotations

from contextlib import suppress
from importlib import import_module

_STAGE_MODULES = (
    "nullsim.stages.scene.point_source",
    "nullsim.stages.telescope.fixed_aperture",
    "nullsim.stages.telescope.pupil_modes",
    "nullsim.stages.atmosphere.kolmogorov_screens",
    "nullsim.stages.ao.pyramid_wfs",
    "nullsim.stages.wavefront.space_residual",
    "nullsim.stages.instrument.nayra_six_chip",
    "nullsim.stages.injection.ideal",
    "nullsim.stages.injection.mode_splitter",
    "nullsim.stages.injection.single_mode_fiber",
    "nullsim.stages.polarization.optics",
    "nullsim.stages.delay_lines.geometric",
    "nullsim.stages.transport.photometric_pickoff",
    "nullsim.stages.transport.smf28",
    "nullsim.stages.chip.identity",
    "nullsim.stages.chip.kernel_mzi_mesh",
    "nullsim.stages.chip.quadrature_reference_mesh",
    "nullsim.stages.chip_optimization.ideal_kernel",
    "nullsim.stages.chip_optimization.scipy",
    "nullsim.stages.detector.ideal_counter",
    "nullsim.stages.detector.mkid",
    "nullsim.stages.detector.snspd",
    "nullsim.stages.background.nayra_sky",
    "nullsim.stages.background.sky_surface_brightness",
    "nullsim.stages.background.space_zodi",
    "nullsim.stages.fringe_tracking.closed_loop",
    "nullsim.stages.diagnostics.fringe_observability",
    "nullsim.stages.calibration.floor_lookup",
    "nullsim.stages.sensitivity.detection_curve",
    "nullsim.stages.performance.standard",
    "nullsim.stages.performance.pdi",
    "nullsim.stages.characterization.spectro",
    "nullsim.stages.characterization.detection_curve",
)

# Optional-extra stages: importing succeeds even without the extra (lazy
# import pattern in each module), but we still wrap in try/except in case
# a future module starts touching the extra at import time.
_OPTIONAL_STAGE_MODULES = (
    "nullsim.stages.chip_optimization.torch",
)

for module_name in _STAGE_MODULES:
    import_module(module_name)

for module_name in _OPTIONAL_STAGE_MODULES:
    with suppress(ImportError):
        import_module(module_name)


__all__: list[str] = []
