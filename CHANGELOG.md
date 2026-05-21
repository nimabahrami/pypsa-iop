# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0a1] - 2026-05-21

Initial alpha release. The package is a reference implementation of
Liang-Dvorkin (2023) + Birge-Hortaçsu-Pavlin (2017) inverse-OPF
methods, wrapped as a PyPSA-compatible counterfactual layer. Math
faithful, real-data validation not yet shipped (see real-data
notebook for a working API path).

### Added
- Three inverse-OPF formulations: `noiseless`, `noisy` (canonical
  Liang-Dvorkin 2023 single-level KKT-QP, default), `zonal` (for
  EUPHEMIA-style bidding-zone clearings). The canonical
  `examples/full_lifecycle_NL.ipynb` exercises `noisy` end-to-end.
- Active-Set Temporal Batching (ASTB) for efficient multi-timestep
  calibration on pure-thermal grids; collapses to single-batch when
  intertemporal components (storage / links / global constraints)
  are present.
- Bayesian posterior via Laplace approximation; optional NUTS-MCMC
  via the `[mcmc]` extra (experimental — single skipped test gated
  on `pymc` import).
- `pio.identifiability` — per-parameter σ_post + information-gain
  flag, plus a relative-σ presentation column (`§7` of the notebook
  documents three known failure modes).
- `pio.flag_withholding` — Birge-Hortaçsu-Pavlin (2017) market-
  monitoring scorer.
- `pio.observations_from_pypsa` — one-call observation-DataFrame
  builder from a solved PyPSA network.
- `pio.validate_observations` / `pio.assess_data_quality` — pre-flight
  checks for real ENTSO-E downloads.
- ENTSO-E data integration via the `[entso_e]` extra; usage example
  in `examples/real_data_entso_e.ipynb`.
- PyPSA network read/write interface (`pio.apply` writer for the
  six supported component families).

### Real-market validation

- `examples/real_data_DE_LU_validation.ipynb` calibrates against real
  Open Power System Data (OPSD 2020-10-06 release) DE_LU day-ahead
  clearings for the week of 4 Nov 2019; cross-validates on the
  week of 11 Nov 2019.
- **Calibrated forecast: €5.66 / MWh RMSE.**
- **Engineering reference: €22.80 / MWh RMSE.**
- **Improvement: −75.2 % on real EPEX data.**
- 38 KB validation slice ships with the package at
  `examples/data/de_lu_2019_validation.csv`.

### Caveats explicitly NOT shipped
- The "88 % RMSE improvement" reported in `examples/full_lifecycle_NL.ipynb`
  §10 is a *synthetic* stress-test against a deliberately-50%-perturbed
  engineering-reference baseline. The 75 % number above is the real-
  market figure to act on.
- The MCMC posterior path is gated on PyMC and only sanity-checked,
  not integration-tested.
