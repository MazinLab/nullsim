---
date: 2026-11-07
---
### Fixed: imaging reconstruction solves, normalization, and leakage model

These fixes come from the 2026-11-07 physics review: Issue 24, the imaging part
of Issue 9, and rows 52 to 55. Only `hwo_space` runs `imaging_reconstruction`.
Its imaging products stay zero until the 8 m mode-size bug (Issue 1) is fixed
in sensitivity, so no shipped-config number was measured moving here.

- **NNLS no longer publishes failed solves (Issue 24).** The signed
  (`chip_phase`) design is solved with `lsq_linear`, which always returns an
  iterate. Before, its 200-iteration cap and "no progress" exits were accepted
  without comment. Now a status of 0 or −1 triggers a retry on the
  column-normalized design with `max(200, 20 × n_columns)` iterations. A solve
  that still fails raises a `RuntimeWarning`, and
  `tables/imaging_reconstruction.json` records it as `nnls_converged = false`
  along with `nnls_solver_status` and `nnls_solver_message`.
  `ReconstructedImage` carries the real iteration count and residual norm.
  Scipy's `nnls` does not report an iteration count, so `n_iterations` is
  `None` on the unsigned path. On an HWO-shaped resolved solve (8 modes, a
  random unitary, 512 candidates, regularization 0.01, 7 wavelengths, 25
  rolls), the first attempt stalls and the retry converges.
- **Planet and leakage use the physical stellar rate (Issue 9).** Imaging now
  reads `performance.standard`'s `stellar_rate_hz_total_uncapped` and falls
  back to the capped total only when no uncapped one is published. Without
  `performance.standard`, it sums the detector's `uncapped_port_rates`.
- **Systematic amplitude (row 52).** The leakage instability is
  `performance.standard`'s `eps_systematic_used` (null instability and residual
  speckle combined), not the bare `null_instability`. It is averaged over servo
  cycles only when `with_fringe_tracking` is true. `hwo_space` sets
  `residual_speckle_stability_factor = 0.01` with fringe tracking off, so
  imaging now sees at least ε = 0.01, with no averaging. Before, it saw 0,
  and the averaging would have cut ε by 2600× in rms. The summary JSON replaces
  `null_instability` with `eps_systematic_used` and `with_fringe_tracking`.
- **Leakage distribution (row 53).** The nominal chip's static coherent
  residual `|Σ U_dm a_m|²` stays in the ports it lands in. The rest of the
  realized null is spread in proportion to `Σ |U_dm|² |a_m|²`, which is the
  mean leakage pattern for independent input phase errors. Before, the whole
  null followed the coherent residual, which a deep null makes arbitrary: a
  1e-10 rad rotation of a Hadamard chip moved [33333, 33333, 33333] counts
  to [1e5, 0, 0].
- **Chop states share one dark leakage (row 54).** The plus and minus rows
  are a unitary recombination of the dark pairs, so the realized leakage is
  split over them jointly. Before, each state received the full null, which
  doubled the leakage shot variance relative to the planet.
- **Broadband weighting (row 55).** Band-averaged response rows and leakage
  are weighted by the supplied per-wavelength stellar rates instead of
  arithmetically. `ObservationData.wavelength_weights` carries the weights,
  so reconstructions that rebuild the response use the same ones. With all
  the starlight in one bin, the broadband counts now equal the resolved
  counts summed over wavelength.
