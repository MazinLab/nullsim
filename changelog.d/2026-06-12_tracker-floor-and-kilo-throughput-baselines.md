---
date: 2026-06-12
---
### Chip-opt restart variance tamed: NAYRA tracker floor + KILO throughput penalty

- The ~+/-40% identical-config planet-SNR spread traced to restart-basin
  luck that rank-priority selection could not see: the inherited
  `tracker_observability_min_singular_per_nm = 1e-3` floor sits ~10x below
  the naturally achieved minimum (~1.1e-2 /nm), so `rank_at_floor` is
  always full and the selection key degenerates to objective loss.
- `examples/nayra_eelt.toml` det chip_opt floor raised 1e-3 -> 1.2e-2 /nm,
  just above the achieved minimum. Floor x seed probe
  (examples/test/nayra_eelt_tracker_floor_probe.toml, run 20260612-074332,
  J=5, 3 seeds per floor): seed-to-seed SNR spread 1.60x -> 1.04x, mean
  1347 -> 1447; 3e-2 over-binds (rank 5/17) with no further gain.
- The same floor raise backfires on `examples/kilo_keck_6.toml`: the
  24-mode kernel_order=3 mesh has no DOF slack, so the observability hinge
  sells planet dark throughput (0.068 -> 0.011 in bad cells, J=4 SNR mean
  40 -> 23). There the basin stabilizer is the throughput reward itself:
  `throughput_penalty_alpha` raised 1e-3 -> 1e-2 (floor kept at 1e-3)
  collapses the 3-seed J=4 spread 1.21x -> 1.01x and beats the publication
  chip (run 20260611-083036) at every J: det SNR 40.5 -> 45.3 (J=4),
  6.96 -> 9.33 (J=6), 1.47 -> 1.73 (J=8), 0.173 -> 0.240 (J=10),
  0.0032 -> 0.0039 (J=12); characterization SNR unchanged. Validation
  runs kilo_keck_6_tp_full_{a,b}/20260612-100900; the tp+floor arm (b)
  matched tp-alone at mid-J but lost at J=4 and J=12.
- Probe/validation grids preserved under examples/test/:
  nayra_eelt_tracker_floor_probe, kilo_keck_6_tracker_floor_probe,
  kilo_keck_6_floor_tp_grid, kilo_keck_6_tp_full_{a,b}.
