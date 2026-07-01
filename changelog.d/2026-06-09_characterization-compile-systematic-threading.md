---
date: 2026-06-09
---
### Characterization stages: torch.compile support, systematic-eps consistency, and null_floor emission

Both `characterization.spectro` and `characterization.detection_curve` now
accept a `compile_objective` param (bool, default false) that gates
`torch.compile` on the characterization optimizer objective. The env var
`NULLSIM_TORCH_COMPILE` is also honored. When compilation is on, restarts are
forced to serial to avoid concurrent JIT contention.

Both stages also gain `eps_calibration_residual` and
`residual_speckle_stability_factor` params (mirroring `performance.standard`).
These combine via `_combined_systematic_eps` into a single `eps_systematic`
value that is used consistently through the optimization and SNR calculation,
replacing the previous ad-hoc treatment.

Both stages now emit `eps_systematic` and `null_floor` in their results
payloads, making the realized systematic budget inspectable in
`sweep_results.parquet` and the paper bundle.

### Parallel sweep: coordinated intra-op thread budget

`nullsim run` with multiple sweep workers now sets `OMP_NUM_THREADS`,
`MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS` to `cores / n_workers` before
spawning the process pool. This prevents torch / BLAS from defaulting to a
full-core thread pool inside each worker and oversubscribing the host. An
explicit user setting for any of these variables is left untouched
(`os.environ.setdefault`).
