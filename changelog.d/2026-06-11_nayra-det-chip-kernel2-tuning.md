---
date: 2026-06-11
---
### NAYRA detection chip rebaselined: kernel_order=2 + heavy disk samples

- Iterative det-chip tuning pass over the J=5 cell
  (examples/test/nayra_eelt_det_tune{1,2,3a,3b}.toml, finalist validation
  nayra_eelt_det_final_k{2,3} runs 20260611-162428) found the
  kernel_order=3 Taylor rows compete with planet routing for the same
  unitary degrees of freedom, capping planet dark throughput at
  0.40-0.45 regardless of throughput_penalty_alpha; second-order rows
  keep ~0.58 while disk_weight=2.0 with 5 rings holds the disk floor
  near 6e-6, below the piston-driven realized null that dominates J>=5.
- examples/nayra_eelt.toml chip_opt: kernel_order 3 -> 2, disk_n_rings
  3 -> 5, disk_weight 1.0 -> 2.0, throughput_penalty_alpha 1e-3 -> 1e-2,
  tracker_observability_crlb_weight 1e-3 -> 1e-1, n_restarts 5 -> 8.
  Validated planet SNR vs the kernel_order=3 baseline (20260611-105935):
  J=5 1565 vs 486, J=9 112 vs 72, J=3 1769 vs 3250 (the old J=3 cell's
  deep disk floor plus a lucky tracker draw stays unbeaten);
  characterization SNR is best-of-any-chip at J>=5 because the spectro
  realized floor inherits the detection chip's servo bandwidth.
- Known issue recorded in the config: identical-config chip draws still
  spread ~+/-40% in SNR because tracker_observability_min_singular_per_nm
  = 1e-3 never binds, so rank-priority restart selection degenerates to
  objective loss. Raising the floor until it discriminates is the open
  variance lever.
