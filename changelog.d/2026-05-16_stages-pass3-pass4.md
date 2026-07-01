---
date: 2026-05-16
---
§15.3 Pass 3 + Pass 4: eight real stages shipped completing the full stage port.

Pass 3 (chip kernel + optimization): `chip.kernel_mzi_mesh` (`KernelMZIMesh`),
`chip_optimization.scipy` (`ChipOptimizationScipy`), `chip_optimization.torch`
(`ChipOptimizationTorch`), `chip_optimization.mlx` (`ChipOptimizationMLX`). All
three optimization backends share a common `ChipOptimizationParams` base dataclass.
Torch and MLX backends are optional extras (`nullsim[torch]`, `nullsim[mlx]`).
pyproject.toml gains `[project.optional-dependencies]`.

Pass 4 (detectors + fringe tracking + calibration + performance): `detector.mkid`
(`MKIDDetector`), `detector.snspd` (`SNSPDDetector`), `fringe_tracking.closed_loop`
(`ClosedLoopFringeTracker`), `calibration.floor_lookup` (`CalibrationFloorLookup`),
`performance.standard` (`StandardPerformance`).

Test suite grows from 348+1xfail to 446+3xfail (+98 tests, +2 xfailed KILO pinning
tripwires). All real stages from atmosphere through performance now ship.
