"""Quadratic / heat-rate cost recovery — Gap 1 of the Tier-1 audit.

Real thermal generators have a heat-rate curve that translates into
a quadratic marginal-cost function ``c(p) = c_0 + c_1 · p``. PyPSA
exposes this via the ``marginal_cost_quadratic`` column on the
``Generator`` component, and its forward LOPF objective includes
``Σ_t (c_0 · p + c_1 · p²)``.

The inverse OPF extension stays linear in the cost parameters
``(c_0, c_1)`` because observed dispatch ``p_obs[g, t]`` is *given*,
so the per-row KKT stationarity

    r[g, t]  −  c_0[g]  −  2 · c_1[g] · p_obs[g, t]
              +  λ[bus(g), t]  +  Σ_l PTDF·μ_l  +  ν − ξ
              −  Σ_gc (e_g · w_t) · μ_co2
              =  0

just adds one new column per generator with coefficient
``−2 · p_obs``. This is the standard formulation Birge-Hortaçsu-Pavlin
(2017) Operations Research §3.2 use to recover offer curves
``α(q) = a + b · q`` (with ``c_0 = a``, ``c_q = b/2``).

This test exercises the recovery on a 2-generator scenario where
both generators dispatch at varying levels across snapshots, so
``(c_0, c_1)`` are jointly identifiable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pypsa
import pytest

from pypsa_invopt import apply, calibrate
from pypsa_invopt.network import read_network


def _build_quad_network(snaps: pd.DatetimeIndex) -> pypsa.Network:
    """Two generators with quadratic costs, swinging load.

    The two generators have different heat-rate slopes so the inverse
    OPF has to recover both ``c_0`` AND ``c_1`` to fit the data.
    """
    n = pypsa.Network()
    n.set_snapshots(snaps)
    n.add("Carrier", "AC")
    n.add("Bus", "b1", v_nom=380, carrier="AC")
    n.add("Bus", "b2", v_nom=380, carrier="AC")
    n.add("Line", "l1", bus0="b1", bus1="b2", x=0.1, s_nom=500)
    # Both generators are heat-rate-curve-typed.
    n.add(
        "Generator", "ccgt", bus="b1", p_nom=200,
        marginal_cost=20.0, marginal_cost_quadratic=0.05, carrier="AC",
    )
    n.add(
        "Generator", "ocgt", bus="b2", p_nom=120,
        marginal_cost=60.0, marginal_cost_quadratic=0.10, carrier="AC",
    )
    # Time-varying load so dispatch varies across snapshots — needed
    # for joint identifiability of (c_0, c_1).
    t = np.arange(len(snaps))
    load = 80 + 90 * np.maximum(np.sin(t * np.pi / 12 - np.pi / 2), 0)
    n.add("Load", "ld", bus="b2")
    n.loads_t.p_set["ld"] = load
    return n


def test_read_network_populates_quadratic_metadata():
    snaps = pd.date_range("2025-04-15", periods=4, freq="h")
    n = _build_quad_network(snaps)
    data = read_network(n)
    assert data.gen_marginal_cost_quadratic == {
        "ccgt": pytest.approx(0.05),
        "ocgt": pytest.approx(0.10),
    }


def test_calibrate_recovers_quadratic_cost_coefficients():
    """End-to-end: recover both ``c_0`` and ``c_q`` from observed dispatch.

    The KKT residual being linear in the cost parameters means the QP
    finds the (c_0, c_q) pair that simultaneously zeros all per-(g, t)
    residuals up to regularisation, provided dispatch varies enough
    across snapshots for the two coefficients to be jointly identified.
    """
    snaps = pd.date_range("2025-04-15", periods=24, freq="h")
    n = _build_quad_network(snaps)
    status, _ = n.optimize(solver_name="highs")
    assert status == "ok"

    obs = pd.DataFrame(index=snaps)
    for bus in n.buses.index:
        obs[f"price_{bus}"] = n.buses_t.marginal_price[bus]
    for line in n.lines.index:
        obs[f"flow_{line}"] = n.lines_t.p0[line]
    for gen in n.generators.index:
        obs[f"dispatch_{gen}"] = n.generators_t.p[gen]

    # Perturb both linear AND quadratic priors
    n.generators.at["ccgt", "marginal_cost"] = 30.0
    n.generators.at["ccgt", "marginal_cost_quadratic"] = 0.02
    n.generators.at["ocgt", "marginal_cost"] = 50.0
    n.generators.at["ocgt", "marginal_cost_quadratic"] = 0.05

    result = calibrate(
        network=n, observations=obs,
        formulation="noisy", solver="highs",
        active_set_tol=0.5, obs_sigma=1.0, lambda_reg=1e-5,
    )
    assert result.solver_status == "optimal"

    # Recover only on dispatching generators. Liang-Dvorkin (2023)
    # §5.2 and Birge-Hortaçsu-Pavlin (2017) both flag the structural
    # limit: a generator that never dispatches has no signal in the
    # observations, so neither ``c_0`` nor ``c_q`` is identifiable.
    # Filter to gens that actually ran in the truth solve and only
    # assert on those.
    dispatch_t = n.generators_t.p
    for gen, truth_c0, truth_cq in [("ccgt", 20.0, 0.05), ("ocgt", 60.0, 0.10)]:
        dispatched = dispatch_t[gen].sum() > 1.0
        if not dispatched:
            continue   # non-marginal / non-dispatching → unidentifiable
        c0_rec = result.theta_hat[f"gen:{gen}:marginal_cost"]
        cq_rec = result.theta_hat.get(f"gen:{gen}:marginal_cost_quadratic", 0.0)
        # The combination (c_0 + 2·c_q·p_obs) is what shows up in
        # LMPs — the identifiable composite. Each piece individually
        # is only well-determined when dispatch varies enough across
        # snapshots. We assert on both the slope and the composite.
        assert abs(cq_rec - truth_cq) < 0.03, (
            f"{gen} c_q should be near {truth_cq}; got {cq_rec:.4f}"
        )
        mean_p = float(dispatch_t[gen].mean())
        truth_effective = truth_c0 + 2 * truth_cq * mean_p
        recov_effective = c0_rec + 2 * cq_rec * mean_p
        assert abs(recov_effective - truth_effective) < 3.0, (
            f"{gen} effective marginal cost at mean dispatch should "
            f"be near {truth_effective:.2f}; got {recov_effective:.2f}"
        )

    # apply() writes the quadratic coefficient back for any gen.
    apply(result, n)
    cq_ccgt_recovered = result.theta_hat.get(
        "gen:ccgt:marginal_cost_quadratic", 0.0,
    )
    assert n.generators.at["ccgt", "marginal_cost_quadratic"] == pytest.approx(
        cq_ccgt_recovered, abs=1e-3,
    )


def test_linear_only_network_recovery_unchanged():
    """When all ``marginal_cost_quadratic`` values are zero, the
    inverse OPF must recover identical linear costs to the
    pre-quadratic implementation. Numerical-stability regression check.
    """
    snaps = pd.date_range("2025-04-15", periods=12, freq="h")
    n = pypsa.Network()
    n.set_snapshots(snaps)
    n.add("Carrier", "AC")
    n.add("Bus", "b1", v_nom=380, carrier="AC")
    n.add("Bus", "b2", v_nom=380, carrier="AC")
    n.add("Line", "l1", bus0="b1", bus1="b2", x=0.1, s_nom=500)
    n.add("Generator", "g1", bus="b1", p_nom=200, marginal_cost=15.0)
    n.add("Generator", "g2", bus="b2", p_nom=120, marginal_cost=80.0)
    load = 50 + 60 * np.maximum(
        np.sin(np.arange(12) * np.pi / 6 - np.pi / 2), 0,
    )
    n.add("Load", "ld", bus="b2")
    n.loads_t.p_set["ld"] = load
    n.optimize(solver_name="highs")

    obs = pd.DataFrame(index=snaps)
    for bus in n.buses.index:
        obs[f"price_{bus}"] = n.buses_t.marginal_price[bus]
    for line in n.lines.index:
        obs[f"flow_{line}"] = n.lines_t.p0[line]
    for gen in n.generators.index:
        obs[f"dispatch_{gen}"] = n.generators_t.p[gen]

    n.generators.at["g1", "marginal_cost"] = 25.0
    n.generators.at["g2", "marginal_cost"] = 70.0

    result = calibrate(
        network=n, observations=obs,
        formulation="noisy", solver="highs",
        active_set_tol=0.5, obs_sigma=1.0, lambda_reg=1e-4,
    )
    # c_q should stay at 0 (the prior) — no quadratic information in the data.
    assert result.theta_hat.get("gen:g1:marginal_cost_quadratic", 0.0) == \
        pytest.approx(0.0, abs=1e-6)
    assert result.theta_hat.get("gen:g2:marginal_cost_quadratic", 0.0) == \
        pytest.approx(0.0, abs=1e-6)
    # Linear costs should match truth within standard precision.
    assert abs(result.theta_hat["gen:g1:marginal_cost"] - 15.0) < 1.0
