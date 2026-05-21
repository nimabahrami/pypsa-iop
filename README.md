# pypsa-invopt

**Calibrate your PyPSA grid model to observed market clearings — then ask "what if?"**

So you've built a beautiful PyPSA network of the Dutch grid. CCGTs at Maasvlakte, batteries at AM, the COBRAcable to Denmark, the whole thing. You feed it engineering-reference costs (gas heat-rate × fuel + CO₂ + O&M) and run it. The LMPs come out... fine. Plausible. But you know they're off, because if you plot them against yesterday's actual EPEX clearings the RMSE is 20+ €/MWh and the peaks land in the wrong hours.

That's because nobody in the real market bids their engineering reference. They bid hedging positions, ramp-up opportunity costs, strategic markups, unit-commitment overhang. Your textbook PyPSA model has none of that. **This package fills that gap.**

You give it `(network, observed_LMPs)` → it returns a bid vector that explains those LMPs. Drop the bids back into the network, re-solve, and your model now reproduces the *observed clearings* on the calibration window. Then go ask the question you actually care about: "what if I add 600 MW of BES here?" The forward solver gives you a defensible answer because the costs underneath are calibrated to recent behaviour, not engineered.

> **Be precise about what we recover.** The recovered bids are *calibration-consistent projections* of the real bid book onto a convex DCOPF with quadratic-affine offer curves. Real EUPHEMIA bids are step-functions of quantity with complex orders and paradoxically-rejected blocks — that mechanism gap is real and inverse-OPF can absorb some of it into the recovered numbers. We are not claiming to recover the confidential true bid book. We are recovering a bid vector that, fed back into the same DCOPF, reproduces the observed clearings and lets downstream counterfactuals run on something better than engineering reference.

The math is one sparse convex QP — Liang & Dvorkin's KKT reformulation from their 2023 e-Energy paper — and HiGHS solves it in tens of milliseconds. Quadratic-affine bid recovery (the `α + β·q` heat-rate slopes) follows Birge-Hortaçsu-Pavlin 2017. We didn't invent any of this. We wrapped it for PyPSA, added a temporal-batching trick for big thermal-only grids (ASTB), and built an identifiability layer that tells you which recovered bids you can trust and which ones to ignore.

## Wait — isn't this just an LMP forecaster?

No, and please don't use it as one. **ML is way better at LMP point forecasting.** Modo Energy, Aurora, Eperion — they all use gradient-boosted trees or LSTMs, and they get 3–5 €/MWh RMSE on real European DA markets. This package gets ~7 €/MWh on the §10 cross-validation in the example notebook. Worse than ML. We won't pretend otherwise.

So why would you ever reach for this instead?

Because **ML is structurally incapable of answering counterfactual questions.** It learned a mapping from features to LMP. Ask it "what's the LMP if I add a 600 MW battery at AM bus?" and it has no training data for that — that state doesn't exist in history. Ask "what if CCGT-X trips offline?" same problem. "What if CO₂ price doubles?" extrapolates badly. "Decompose tomorrow's €95 peak into thermal-bid + congestion + carbon-shadow"? It can't — it's a black box. "Is this CCGT strategically withholding?" — the learned LMP *includes* the withholding, so it can't separate it out.

| What you're trying to do | Use ML | Use this |
|---|---|---|
| Point-forecast tomorrow's LMP | yes | no |
| Size a new BES project (need to know LMP post-installation) | no | yes |
| "What if CCGT-X trips?" | no | yes |
| Decompose LMP into bid + congestion + CO₂ shadow | no | yes |
| Detect strategic withholding by a competitor | no | yes |
| "What if CO₂ price doubles?" | no | yes |
| Defend a number to your board / regulator | "model said so" | physical price-formation story |
| Latency in production | ms | seconds–minutes |

The correct architecture is **both**, not either. Run ML for the headline forecast. Run this package for the scenario fan around it. A BES desk that uses ML-only is flying blind on counterfactuals — which is every investment decision. A desk that uses this package only is leaving 30 %+ point-forecast accuracy on the table.

## When this earns its keep

The package is built for **residual-load hours** — the hours where dispatchable thermal capacity is at the margin and where BES revenue actually lives. That means evening peak (17-22 h CET), morning ramp, and the residual-load regime more broadly. In NL/DE 2024 those hours represent maybe 60-70 % of the year but ~95 % of BES revenue; outside that window (midday solar surplus, deep-night low-load) the recovered bids are weakly informed and the package's identifiability layer flags them accordingly.

The number of "renewable-marginal" hours (zero or negative prices) has grown ~3-5 pp/year in NL/DE. We are not pretending that trend reverses; we are pointing out the windows where bid recovery still does most of the work for the decisions you actually have to defend.

Three concrete regimes where inverse-OPF beats every alternative we know of:

**1. Infrastructure planning.** You're sizing a new asset and counterfactuals on textbook costs would give you garbage. A BESS developer evaluating 600 MW at Amsterdam needs to know what the LMP curve *would* look like post-installation, not what the engineering model thinks it should. The §9 walkthrough in `examples/full_lifecycle_NL.ipynb` is this workflow end-to-end.

**2. Market monitoring.** You suspect a generator is strategically withholding capacity. Birge-Hortaçsu-Pavlin (2017) on MISO is the canonical paper; `pio.flag_withholding` implements the same scorer (with the textbook renewables-false-positive filter on by default). You compare recovered marginal cost against engineering reference (fuel + heat-rate + CO₂ + O&M); large positive gap → flag for further investigation.

**3. Decomposed LMP attribution.** You want to know *why* an LMP spike happened. Bid markup, congestion on a specific line, or CO₂ cap binding? With inverse-OPF + the network duals, each piece falls out of the QP — you can do risk decomposition that ML cannot.

For NL/DE specifically: case 1 is the sweet spot. Big project pipelines, tightening grids, planners running PyPSA with textbook costs are feeding their planner numbers that diverge from observed clearings by 20+ €/MWh.

## What's in the box

- **Three formulations.** `noiseless` (KKT-equality LP, fast, for clean synthetic data), `noisy` (KKT-residual QP, the default, canonical L-D 2023), `zonal` (bidding-zone level for EUPHEMIA-style DA clearings).
- **Active-Set Temporal Batching (ASTB)** for pure-thermal grids — the package partitions the snapshot set by active-set signature and solves each batch independently, then aggregates via BLUE. Collapses to one batch when storage or links or global constraints are present (the math forces this — they couple snapshots intertemporally — not an algorithmic limitation).
- **Bayesian posterior** over recovered bids — Laplace approximation (default, fast) or full NUTS-MCMC (slow, but exact when you need it). Both give you per-parameter σ.
- **Identifiability flagging.** Not every bid is recoverable from every dataset — if a generator is always at p_max it can bid anything ≤ LMP and the data won't pin it. The package flags these explicitly via `pio.identifiability`; you can then fall back to engineering reference for the unidentifiable ones.
- **Strategic-withholding scorer** (`pio.flag_withholding`) — BHP-2017 reproduced.
- **`observations_from_pypsa(network)`** — one call to extract the observation DataFrame from a solved PyPSA network. No manual schema wrangling.
- **`validate_observations` + `assess_data_quality`** — pre-flight checks that catch gappy or spiky data *before* you spend a solver run. Essential for real ENTSO-E downloads (Europe).

## Install

```bash
pip install pypsa-invopt              # core
pip install pypsa-invopt[mcmc]        # + pymc + arviz (for NUTS posterior)
pip install pypsa-invopt[entso_e]     # + entsoe-py (for European DA data)
pip install pypsa-invopt[all]         # everything
```

## Quick start

```python
import pypsa
import pypsa_invopt as pio

# 1. Load your PyPSA network, solve forward once, extract observations
n = pypsa.Network('my_network.nc')
n.optimize(solver_name='highs')
obs = pio.observations_from_pypsa(n)            # price_<bus>, dispatch_<gen>, …

# Optional but recommended for real data — catches gaps/spikes early
pio.validate_observations(obs)
print(pio.assess_data_quality(obs))

# 2. Recover the bids — one HiGHS solve, ~50 ms on a small grid
result = pio.calibrate(
    network=n, observations=obs,
    formulation='noisy',          # canonical L-D 2023 KKT-QP
    lambda_reg=1e-6, obs_sigma=1.0,
)

# 3. Write them back into the network
pio.apply(result, n)

# 4. Diagnostics — what's trustworthy, what isn't, who's withholding
posterior = pio.posterior(n, obs, result, method='laplace',
                          prior_std=5.0, obs_std=1.0)
report    = pio.identifiability(posterior)
flags     = pio.flag_withholding(
    theta_hat=result.theta_hat,
    generator_carriers={g: n.generators.at[g, 'carrier'] for g in n.generators.index},
    posterior_identifiability=report,
    fuel_prices={'gas': 32, 'coal': 12, 'wind': 0},
    co2_price=75,
)

# 5. Now ask the question you actually care about
n.add('StorageUnit', 'new_BES', bus='AM', p_nom=600, max_hours=4)
n.optimize(solver_name='highs')                 # forward DCOPF with calibrated bids
new_lmps = n.buses_t.marginal_price
```

### Real ENTSO-E zonal data (Europe)

European DA markets clear at the bidding-zone level (EUPHEMIA), so use `formulation='zonal'`:

```python
obs = pio.load_entso_e(bidding_zone='NL', start='2025-06-15',
                       end='2025-06-16', synthesize=True)
result = pio.calibrate(
    network=n, observations=obs,
    formulation='zonal',
    bus_to_zone={'b1': 'NL', 'b2': 'NL'},  # map each bus to its zone
    lambda_reg=1e-6,
)
```

## Does it actually work?

**Yes — on a held-out week of real EPEX data.** Headline numbers from `examples/real_data_DE_LU_validation.ipynb` (DE_LU day-ahead, week of 11 Nov 2019, calibrated on the preceding week):

| Held-out-week LMP RMSE | EUR/MWh |
|---|---|
| Engineering reference (fuel + heat-rate + CO₂ + O&M) | **22.80** |
| Inverse-OPF calibrated bids | **5.66** |
| **Improvement** | **−75.2 %** |

> **Caveat that rides shotgun with the headline.** This is **one week** of late 2019 (pre-COVID, pre-gas-crisis, pre-renewables-buildout-at-scale, pre-CO₂-price-doubling). The regime where the package was validated is qualitatively different from the regime where you'd actually deploy it. Rolling-week validation across 2018-2024 is the next-paper-worth-of-work; until that lands, treat this number as **proof-of-concept evidence**, not a generalisation guarantee. The synthetic stress test in `examples/full_lifecycle_NL.ipynb` §10 hits 88 % RMSE reduction on a controlled-truth case — that's the *upper bound* when you know the answer. Real-market across regimes is the open question.

Data source: Open Power System Data (OPSD) 2020-10-06 release, ENTSO-E Transparency snapshot. Public, citeable, no API key. Shipped slice at `examples/data/de_lu_2019_validation.csv` (38 KB).

### Two different RMSE numbers, do not conflate them

- **75 % RMSE reduction** — real EPEX clearings, DE_LU, one held-out week of Nov 2019. The number above. The bankable one.
- **88 % RMSE reduction** — synthetic 6-bus NL grid in `full_lifecycle_NL.ipynb` §10, against a deliberately-50%-perturbed engineering reference. Stress test on a controlled-truth case.
- **79 % "marginal-generator identification accuracy"** — also in the synthetic NL notebook §10. This is a *synthetic* metric: real EPEX does not publish ground-truth on which unit was on the margin (confidential). The 79 % number is *not* the same kind of evidence as the 75 % real-EPEX number; it's a synthetic illustration of *why* calibration helps. If you cite the package, cite the 75 %.

### The rest of the test stack

- **94 unit + integration tests pass, 85 % line coverage.** Core math modules (formulations / calibration / network / identifiability / reference-cost / posterior) sit at 83–100 %. Optional integration paths (`bayes/mcmc.py` needs PyMC) skip gracefully.
- **Notebook-execution integration test** runs all three example notebooks headless in CI on every push.
- **Storage-marginality test.** Recovers truth `c_s = €45 → €44.98` on a 2-bus network where the battery sets the marginal price at evening peak.
- **CO₂ shadow-price recovery.** Within ±$15/tCO₂ of the analytical KKT switch-point value.
- **Real-data ENTSO-E path.** `examples/real_data_entso_e.ipynb` wires the same pipeline to the live ENTSO-E Transparency API (NL bidding zone). Runs in full with an `ENTSOE_API_KEY` env var; falls back to a fixture otherwise.

## What this package is *not*

- **Not an LMP forecaster.** For point prediction, ML beats this. We will not pretend otherwise. See the ML-vs-this table above.
- **Not a replacement for your ML / statistical forecasting stack.** Use this *alongside* it, for the counterfactual questions ML can't answer.
- **Not a trading product.** Using *competitor-specific* recovered bids to inform your own bidding edges into EU competition-law territory. The safe regime is portfolio-level LMP forecasting + market monitoring, not "I see CCGT-X is bidding €68 so I'll undercut at €67."
- **Not a market-monitor replacement.** Regulators (FERC, ACER, ACM) use proprietary audited tools. The `flag_withholding` scorer is a *screening* tool, not a finding.
- **Not a step-function bid recoverer.** Real EUPHEMIA bids are step functions of quantity (multiple price-quantity blocks). This package recovers linear + quadratic only — matches L-D 2023 / BHP 2017 SOTA. Step recovery is open research.
- **Not a mixed-integer inverse-OPF.** Unit commitment with binary on/off + startup costs requires complementarity conditions, not linear KKT. Also open research.

## References

- **Liang Z., Dvorkin Y. (2023)** "Data-Driven Inverse Optimization for Marginal Offer Price Recovery in Electricity Markets." arXiv:2302.05498. *The canonical single-level KKT-QP reformulation this package implements.*
- **Birge J.R., Hortaçsu A., Pavlin J.M. (2017)** "Inverse Optimization for the Recovery of Market Structure from Market Outcomes." *Operations Research* 65(4). *Quadratic-affine bid recovery + the market-monitoring application behind `flag_withholding`.*
- **Aswani A., Shen Z.J., Siddiq A. (2018)** "Inverse Optimization with Noisy Data." *Operations Research* 66(3). *The noisy-KKT reformulation behind `formulation='noisy'`.*
- **Stuart A.M. (2010)** "Inverse Problems: A Bayesian Perspective." *Acta Numerica* 19. *Laplace + posterior σ for parameter identifiability.*
- **Brown T., Hörsch J., Schlachtberger D. (2018)** "PyPSA: Python for Power System Analysis." *JORS* 6(4).

## License

MIT
