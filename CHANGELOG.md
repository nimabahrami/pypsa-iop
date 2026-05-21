# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0a3] - 2026-05-21

Metadata-only release. No code or behaviour changes from 0.1.0a2.
Added a missing `[project.urls]` block to `pyproject.toml` so the
PyPI project page renders sidebar links (Homepage, Repository,
Issues, Changelog, Documentation) back to the GitHub repo. Also
added PyPI search keywords and two extra PyPI classifiers
(Mathematics / Physics topics, OS-independent flag) to improve
search ranking and signal the package's scope.

### Added
- `[project.urls]`: Homepage / Repository / Issues / Changelog /
  Documentation, all pointing at the GitHub repo.
- `keywords` list for PyPI search.
- Two additional classifiers.

## [0.1.0a2] - 2026-05-21

Documentation + framing tightening. No code or behaviour changes
relative to 0.1.0a1 — the public API, recovered numbers, and test
suite are identical. The release exists so the new README renders
on PyPI (a re-upload of the same version is not permitted).

### Changed
- Soften "matches reality" everywhere to "reproduces the observed
  clearings on the calibration window" — the model fits clearings,
  it does not reveal the confidential true bid book.
- Sharpen the use-case claim to BESS-revenue-relevant hours
  (evening peak, morning ramp, residual-load regime) instead of a
  blanket "thermals set the marginal price most hours". The latter
  is increasingly false as zero/negative-price hours grow; the
  package's value lives in the residual-load windows.
- Move the "one week of late 2019, pre-COVID / pre-gas-crisis /
  pre-renewables-buildout / pre-CO₂-doubling" caveat to ride
  shotgun with the 75 % real-EPEX headline, not after it.
- Add an explicit "two different RMSE numbers, do not conflate"
  block separating 75 % (real EPEX) from 88 % (synthetic stress
  test) from 79 % (synthetic marginal-gen-id accuracy — real EPEX
  does not publish ground-truth on the marginal unit).
- New precision-of-recovery callout in the README opener stating
  the recovered bids are calibration-consistent projections onto
  a convex DCOPF with quadratic-affine offers, not the
  confidential real bid book; mechanism gap (EUPHEMIA step-
  function bids + complex orders) acknowledged.
- PyPI short description updated to match.

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

### Real-market validation (one week, late 2019)

- `examples/real_data_DE_LU_validation.ipynb` calibrates against real
  Open Power System Data (OPSD 2020-10-06 release) DE_LU day-ahead
  clearings for the week of 4 Nov 2019; cross-validates on the
  week of 11 Nov 2019.
- **Calibrated forecast: €5.66 / MWh RMSE.**
- **Engineering reference: €22.80 / MWh RMSE.**
- **Improvement: −75.2 % on real EPEX data.**
- 38 KB validation slice ships with the package at
  `examples/data/de_lu_2019_validation.csv`.

**Caveat that rides shotgun with the headline number.** This is *one*
week of late 2019: pre-COVID, pre-gas-crisis, pre-renewables-buildout-
at-scale, pre-CO₂-price-doubling. The regime where it was validated is
qualitatively different from the regime where you'd deploy it. Treat
the 75 % as proof-of-concept evidence, not a generalisation guarantee.
Rolling-week validation across 2018-2024 is the next-paper-worth-of-
work.

### What we recover (precisely)

The recovered bids are *calibration-consistent projections* of the
real bid book onto a convex DCOPF with quadratic-affine offer curves.
Real EUPHEMIA bids are step-functions with complex orders and
paradoxically-rejected blocks; the recovery can absorb some of that
mechanism gap into the recovered numbers. We do not claim to recover
the confidential true bid book — we recover a bid vector that
reproduces observed clearings and feeds downstream counterfactuals
on something better than engineering reference.

### Numbers to NOT conflate

- **75 %** = real EPEX held-out-week RMSE reduction (the bankable one).
- **88 %** = synthetic controlled-truth stress test (`full_lifecycle_NL.ipynb` §10).
- **79 %** = "marginal-generator identification accuracy", *synthetic only*
  — real EPEX does not publish ground-truth on which unit was marginal.

### Caveats explicitly NOT shipped
- The MCMC posterior path is gated on PyMC and only sanity-checked,
  not integration-tested.
