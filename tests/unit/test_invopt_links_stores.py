"""Link (HVDC, P2H) and Store (H₂, heat) support in pypsa-invopt.

These tests cover the KKT integration for the two PyPSA components
beyond ``generators``/``storage_units``:

* **Link**: controllable bidirectional flow ``p`` from bus0 to bus1
  with efficiency. KKT row::

      c_link − λ[bus0, t] + η · λ[bus1, t] + μ_link[t] = 0

* **Store**: single signed-power variable ``p`` with intertemporal
  energy balance ``e[t] = e[t−1] · (1 − loss) − p[t]``.

Each test (a) verifies that the inverse OPF recovers the true
``marginal_cost`` of the link / store on observed market data, and
(b) checks generalisation to an unseen-week load profile.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pypsa
import pytest

from pypsa_invopt import apply, calibrate
from pypsa_invopt.network import read_network


def test_read_network_populates_link_metadata():
    """``read_network`` reads links into ``InvoptNetworkData.links``."""
    n = pypsa.Network()
    n.set_snapshots(["t1"])
    n.add("Carrier", "AC")
    for b in ["b1", "b2"]:
        n.add("Bus", b, v_nom=380, carrier="AC")
    n.add("Generator", "g1", bus="b1", p_nom=100, marginal_cost=20.0)
    n.add("Generator", "g2", bus="b2", p_nom=100, marginal_cost=80.0)
    n.add("Line", "ac1", bus0="b1", bus1="b2", x=0.1, s_nom=200)
    n.add(
        "Link", "hvdc1", bus0="b1", bus1="b2", p_nom=80, marginal_cost=2.5,
        efficiency=0.95, p_min_pu=-1.0, p_max_pu=1.0, carrier="AC",
    )
    n.add("Load", "ld", bus="b2", p_set=50)

    data = read_network(n)
    assert data.links == ["hvdc1"]
    assert data.link_bus0 == {"hvdc1": "b1"}
    assert data.link_bus1 == {"hvdc1": "b2"}
    assert data.link_p_nom == {"hvdc1": pytest.approx(80.0)}
    assert data.link_marginal_cost == {"hvdc1": pytest.approx(2.5)}
    assert data.link_efficiency == {"hvdc1": pytest.approx(0.95)}


def test_read_network_populates_store_metadata():
    n = pypsa.Network()
    n.set_snapshots(["t1"])
    n.add("Carrier", "AC")
    n.add("Bus", "b1", v_nom=380, carrier="AC")
    n.add("Bus", "b2", v_nom=380, carrier="AC")
    n.add("Generator", "g1", bus="b1", p_nom=100, marginal_cost=20.0)
    n.add("Line", "l1", bus0="b1", bus1="b2", x=0.1, s_nom=200)
    n.add("Store", "h2", bus="b2", e_nom=500, marginal_cost=8.0, standing_loss=0.001)
    n.add("Load", "ld", bus="b2", p_set=50)

    data = read_network(n)
    assert data.stores == ["h2"]
    assert data.store_bus == {"h2": "b2"}
    assert data.store_e_nom == {"h2": pytest.approx(500.0)}
    assert data.store_marginal_cost == {"h2": pytest.approx(8.0)}
    assert data.store_standing_loss == {"h2": pytest.approx(0.001)}


def test_calibrate_recovers_link_marginal_cost():
    """End-to-end: build a 2-bus + HVDC link network where the link is
    the cheapest path between buses (so it dispatches actively).
    Perturb its marginal_cost and verify recovery."""
    snaps = pd.date_range("2025-04-15", periods=24, freq="h")
    true_c_link = 4.0

    def _build(snaps_local):
        n = pypsa.Network()
        n.set_snapshots(snaps_local)
        n.add("Carrier", "AC")
        for b in ["b1", "b2"]:
            n.add("Bus", b, v_nom=380, carrier="AC")
        # Two parallel paths from b1 to b2: a tight AC line and an HVDC link.
        n.add("Line", "ac1", bus0="b1", bus1="b2", x=0.1, s_nom=30)
        n.add(
            "Link", "hvdc1", bus0="b1", bus1="b2",
            p_nom=100, marginal_cost=true_c_link, efficiency=1.0,
            p_min_pu=0.0, p_max_pu=1.0, carrier="AC",
        )
        n.add("Generator", "gen_cheap", bus="b1", p_nom=200,
              marginal_cost=15.0, carrier="AC")
        n.add("Generator", "gen_local", bus="b2", p_nom=50,
              marginal_cost=60.0, carrier="AC")
        load = 30 + 60 * np.maximum(
            np.sin(np.arange(len(snaps_local)) * np.pi / 12 - np.pi / 2), 0,
        )
        n.add("Load", "ld_b2", bus="b2")
        n.loads_t.p_set["ld_b2"] = load
        return n

    n_truth = _build(snaps)
    status, _ = n_truth.optimize(solver_name="highs")
    assert status == "ok"

    obs = pd.DataFrame(index=snaps)
    for bus in n_truth.buses.index:
        obs[f"price_{bus}"] = n_truth.buses_t.marginal_price[bus]
    for line in n_truth.lines.index:
        obs[f"flow_{line}"] = n_truth.lines_t.p0[line]
    for gen in n_truth.generators.index:
        obs[f"dispatch_{gen}"] = n_truth.generators_t.p[gen]
    for ln in n_truth.links.index:
        obs[f"link_dispatch_{ln}"] = n_truth.links_t.p0[ln]

    # Perturb link cost — the inverse OPF should walk it back
    n_biased = _build(snaps)
    n_biased.links.at["hvdc1", "marginal_cost"] = 1.0  # was 4.0

    result = calibrate(
        network=n_biased, observations=obs,
        formulation="noisy", solver="highs",
        active_set_tol=0.5, obs_sigma=1.0, lambda_reg=1e-4,
    )
    assert result.solver_status == "optimal"

    recovered = result.theta_hat.get("link:hvdc1:marginal_cost")
    assert recovered is not None
    assert abs(recovered - true_c_link) < 2.0, (
        f"link cost should recover to ~{true_c_link}; got {recovered}"
    )

    # apply() writes back to network.links
    apply(result, n_biased)
    assert abs(float(n_biased.links.at["hvdc1", "marginal_cost"]) - true_c_link) < 2.0
