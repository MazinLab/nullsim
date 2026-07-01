# Calibration Stage Notes

## KILO source surveyed

Files inspected:

- `kilo/studies/calibration_time.py`
  lines 81-188: `lookup_calibration_null(wavelength_center_m, cal_jmag,
  budget_min, results_dir, design_slug)`. Reads a pre-computed JSON file
  (`calibration_time_{band}.json`) and indexes into
  `analysis[f"J{cal_jmag:.0f}"][f"{budget_min:.0f}min"]["best_null_depth"]`.
  Bands are Y, J, H with centers `1.00e-6`, `1.25e-6`, `1.65e-6` m and
  bandwidths `0.10e-6`, `0.15e-6`, `0.20e-6` m
  (`BAND_DEFINITIONS` at lines 73-78). A 0.06 µm tolerance pins
  wavelength→band assignment.
- `kilo/studies/calibration_time.py`
  lines 593-670: `_analyze_time_budgets` — produces the keys consumed
  by the lookup. Each `(jmag, budget_min)` cell records
  `best_null_depth` (median best-of-K across optimizer restarts), the
  number of independent restarts, and a `floor_estimate` from the
  measurement photon budget.
- `kilo/studies/calibration_time.py`
  lines 22-79 (constants + helpers): `N_MIN_COUNTS = 100`,
  `T_SWITCH_S = 1e-4`, `DEFAULT_CAL_JMAG = 5.0`,
  `DEFAULT_BUDGET_MIN = 30.0`, `CALIBRATION_BANDWIDTH_M = 0.10e-6`,
  `CAL_STAR_JMAGS = (3.0, 5.0, 7.0)`,
  `BUDGETS_MIN = (1, 2, 5, 10, 15, 20, 30, 45, 60)`. The lookup is a
  *table*; calibration is NOT a real-time stage in KILO.
- `kilo/performance.py` lines
  1263-1304 (`effective_null_with_disk_floor`): the calibrated null
  feeds back into the performance stage as the baseline null floor.

## NAYRA source surveyed

NAYRA does not ship a calibration_time study. The E-ELT preset reuses
the natural-design KILO calibration_time table directly (acceptable for
~30 m baselines per the 2026-05-13 design review). The
`lookup_calibration_null` legacy-fallback warning at KILO lines 142-153
is the load-bearing safety net here.

## nullsim mapping

The nullsim calibration stage `CalibrationFloorLookup` reproduces the
behavior of `lookup_calibration_null` with two changes that fit the
SPEC §6 stage protocol:

1. The JSON path is a Param (`lookup_table_path`) instead of being
   inferred from `results_dir + band`. A single stage instance reads
   one file. Wavelength→band resolution is moved to a Param
   (`band: Literal["I", "Y", "J", "H"]`) so the stage does not silently
   re-bin a wavelength grid.
2. When `lookup_table_path is None` the stage falls back to a constant
   `default_floor_log10 = -6.0`, i.e. a `1e-6` raw-null calibration
   floor. This is the conservative "design value" floor used in the
   pre-2026-05 paperv2 plots before the dedicated calibration_time
   study landed.

The subtraction rule is `null_depth_residual = max(null_depth_raw -
calibration_floor, 0.0)` — calibrating *removes* a systematic floor,
and a negative residual is unphysical (the stage clamps at zero rather
than propagating sign errors downstream). The stage keeps the legacy
`null_depth_calibrated` key as an alias for `null_depth_residual`; new
consumers should use the explicit floor/residual names so KILO/NAYRA's
published calibration-floor scalars are not confused with post-subtraction
residuals.

## JSON schema accepted by the stage

```
{
  "analysis": {
    "J5": {
      "30min": {"best_null_depth": 1.2e-6, ...},
      "60min": {"best_null_depth": 8.7e-7, ...}
    },
    "J7": {
      "30min": {"best_null_depth": 2.1e-6, ...}
    }
  }
}
```

The stage indexes `data["analysis"][f"J{round(jmag)}"][f"{round(budget_min)}min"]["best_null_depth"]`
to match `kilo.studies.calibration_time.lookup_calibration_null`.
When the exact `(jmag, budget_min)` cell is missing, the stage linearly
interpolates in `(jmag, log(budget_min))` between bracketing entries
within the same band (the budget axis is geometric in KILO's BUDGETS_MIN
grid). When the requested point falls outside the tabulated grid the
stage falls back to `10 ** default_floor_log10` rather than
extrapolating.

## Default floor justification

`10 ** -6.0 = 1e-6` is the raw-null floor implied by KILO's
`floor_estimate` heuristic in `_analyze_time_budgets` for J=5,
budget=30 min on a 10 m Keck-style aperture at J-band. The full
calibration_time study lowers this by ~0.3-0.5 dex with optimizer
restarts on bright calibrators, but the 1e-6 design value is the right
"no-lookup-table-available" placeholder for preset configs that have
not yet been re-run through the study. The fallback is logged in the
stage's results component under `source_model` so downstream parity
tests can tell when a config slipped through without a real table.

## Source lines for pin tests

- KILO `lookup_calibration_null`: lines 81-188.
- KILO `BAND_DEFINITIONS`: lines 73-78.
- KILO `_analyze_time_budgets` (table producer): lines 593-670.
- KILO `effective_null_with_disk_floor` (consumer): performance.py
  lines 1263-1304.
