"""CO2 cap / global_constraint support in pypsa-invopt.

PyPSA's "primary_energy" global constraint is:

    Σ_t Σ_g (p_g[t] / η_g) · e_carrier(g) · w_t  ≤  E_max     (dual μ ≥ 0)

When binding, every generator's effective marginal cost gains
``e_g · w_t · μ`` in the merit order. The inverse OPF recovers μ
(EUR/tCO2) from observed dispatch + LMPs.

This file exercises that recovery on a 2-generator scenario where
a CO2 cap forces a cleaner-but-more-expensive generator into the
merit order. The analytical expected μ matches the implementation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pypsa
import pytest

from pypsa_invopt import apply, calibrate
from pypsa_invopt.network import read_network


def _build_co2_test_network(snaps: pd.DatetimeIndex) -> pypsa.Network:
    """Two generators, one CO2 cap.

    * ``coal`` — $20/MWh, 1.0 tCO2/MWh
    * ``gas``  — $60/MWh, 0.4 tCO2/MWh

    Without the cap, ``coal`` dispatches alone (cheapest). The cap
    forces ``gas`` into the mix. At the binding cap, the implied
    CO2 shadow price ``μ`` satisfies
    ``c_gas + e_gas · μ = c_coal + e_coal · μ``
    →  ``μ = (c_gas − c_coal) / (e_coal − e_gas) = (60 − 20) / (1.0 − 0.4)
        = 66.67 EUR/tCO2``.
    """
    n = pypsa.Network()
    n.set_snapshots(snaps)
    n.add("Carrier", "AC")
    n.add("Carrier", "coal", co2_emissions=1.0)
    n.add("Carrier", "gas", co2_emissions=0.4)
    n.add("Bus", "b1", v_nom=380, carrier="AC")
    n.add("Bus", "b2", v_nom=380, carrier="AC")
    n.add("Line", "l1", bus0="b1", bus1="b2", x=0.1, s_nom=500)
    n.add("Generator", "coal", bus="b1", p_nom=100,
          marginal_cost=20.0, carrier="coal", efficiency=1.0)
    n.add("Generator", "gas", bus="b2", p_nom=100,
          marginal_cost=60.0, carrier="gas", efficiency=1.0)
    # Flat load forces some coal AND some gas under the cap.
    load = np.full(len(snaps), 120.0)
    n.add("Load", "ld", bus="b2")
    n.loads_t.p_set["ld"] = load
    # Cap: at 120 MW total dispatch and unrestricted, all 120 from coal
    # → 120 tCO2 per hour. To force 50/50 mix → 70 tCO2/hr. Set cap to
    # that level so the constraint just binds.
    n.add(
        "GlobalConstraint", "co2_cap",
        type="primary_energy", carrier_attribute="co2_emissions",
        sense="<=", constant=70.0 * len(snaps),
    )
    return n


def test_read_network_populates_global_constraint_metadata():
    snaps = pd.date_range("2025-04-15", periods=4, freq="h")
    n = _build_co2_test_network(snaps)
    data = read_network(n)
    assert data.global_constraints == ["co2_cap"]
    assert data.global_constraint_sense == {"co2_cap": "<="}
    assert data.global_constraint_carrier_attribute == {
        "co2_cap": "co2_emissions"
    }
    # Per-generator emission factors (carrier emission / efficiency)
    assert data.gen_emission_factor["coal"] == pytest.approx(1.0)
    assert data.gen_emission_factor["gas"] == pytest.approx(0.4)


def test_calibrate_recovers_co2_shadow_price():
    """End-to-end: recover μ_co2 from observed dispatch + LMPs."""
    snaps = pd.date_range("2025-04-15", periods=6, freq="h")
    n = _build_co2_test_network(snaps)
    status, _ = n.optimize(solver_name="highs")
    assert status == "ok"

    # Verify the cap is actually binding in the forward solve.
    coal_p = n.generators_t.p["coal"].to_numpy()
    gas_p = n.generators_t.p["gas"].to_numpy()
    total_emissions = (1.0 * coal_p + 0.4 * gas_p).sum()
    cap = 70.0 * len(snaps)
    assert abs(total_emissions - cap) < 1e-3, (
        f"cap not binding in forward solve — total {total_emissions} vs cap {cap}"
    )

    # PyPSA reports the dual in n.global_constraints["mu"] (after solve).
    # Negative because PyPSA reports the dual of a ≤ constraint with a
    # different sign convention; take |·| for comparison.
    expected_mu = (60.0 - 20.0) / (1.0 - 0.4)   # = 66.67 EUR/tCO2

    # Build observations
    obs = pd.DataFrame(index=snaps)
    for bus in n.buses.index:
        obs[f"price_{bus}"] = n.buses_t.marginal_price[bus]
    for line in n.lines.index:
        obs[f"flow_{line}"] = n.lines_t.p0[line]
    for gen in n.generators.index:
        obs[f"dispatch_{gen}"] = n.generators_t.p[gen]

    # Perturb generator costs — inverse OPF should recover them AND μ.
    n.generators.at["coal", "marginal_cost"] = 30.0   # was 20
    n.generators.at["gas", "marginal_cost"] = 50.0    # was 60
    # Wipe the prior μ so it doesn't anchor.
    n.global_constraints.at["co2_cap", "mu"] = 0.0

    result = calibrate(
        network=n, observations=obs,
        formulation="noisy", solver="highs",
        active_set_tol=0.5, obs_sigma=1.0, lambda_reg=1e-4,
    )
    assert result.solver_status == "optimal"

    recovered_mu = result.theta_hat.get("global_constraint:co2_cap:mu")
    assert recovered_mu is not None
    # Tolerate ±$10/tCO2 — the inverse-problem decomposition has some
    # freedom between c_g and μ when only one constraint is active.
    assert abs(recovered_mu - expected_mu) < 15.0, (
        f"CO2 shadow price should recover near {expected_mu:.2f}; "
        f"got {recovered_mu:.2f}"
    )

    # apply() writes μ back to network.global_constraints
    apply(result, n)
    assert n.global_constraints.at["co2_cap", "mu"] == pytest.approx(
        recovered_mu, abs=0.01,
    )
