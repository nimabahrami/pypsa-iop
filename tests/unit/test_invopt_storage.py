"""Storage-unit support in pypsa-invopt — Phase 1 (metadata + graceful recovery).

Phase 1 scope (this file):
    * ``read_network`` populates ``storage_units`` and related fields on
      :class:`InvoptNetworkData`.
    * Networks containing ``StorageUnit`` components don't break the
      inverse OPF.
    * When storage operates at its capacity bound (saturated discharge
      or charge), the line and generator KKT machinery picks up the
      slack and generator-cost recovery remains correct.

Phase 2 (open, documented elsewhere): When storage is the sole
marginal price-setter — *not* at a bound — the LMP equals
``marginal_cost + intertemporal SOC opportunity cost``, which is
endogenous to the storage's own optimization. Recovering it requires
modelling the SOC dynamics in the inverse problem. The current
formulation does NOT do this; the recovered generator costs in those
hours carry a bias that needs Phase 2 to remove.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pypsa
import pytest

from pypsa_invopt import calibrate
from pypsa_invopt.network import read_network


def _basic_network_with_battery() -> pypsa.Network:
    """3-bus network with a battery at the load bus."""
    n = pypsa.Network()
    snaps = pd.date_range("2025-04-15", periods=24, freq="h")
    n.set_snapshots(snaps)
    n.add("Carrier", "AC")
    for b in ["b1", "b2", "b3"]:
        n.add("Bus", b, v_nom=380, carrier="AC")
    n.add("Line", "l12", bus0="b1", bus1="b2", x=0.1, s_nom=200)
    n.add("Line", "l23", bus0="b2", bus1="b3", x=0.1, s_nom=200)
    n.add("Generator", "gen_base", bus="b1", p_nom=300, marginal_cost=20.0,
          carrier="AC")
    n.add("Generator", "gen_peak", bus="b3", p_nom=100, marginal_cost=80.0,
          carrier="AC")
    n.add(
        "StorageUnit", "batt_b3", bus="b3", p_nom=50, max_hours=4.0,
        marginal_cost=5.0, efficiency_store=0.95, efficiency_dispatch=0.95,
        carrier="AC",
    )
    load = 100 + 80 * np.sin(np.arange(24) * np.pi / 12 - np.pi / 2).clip(0)
    n.add("Load", "ld_b3", bus="b3")
    n.loads_t.p_set["ld_b3"] = load
    return n


def test_read_network_populates_storage_metadata():
    """``read_network`` reads storage_units into the InvoptNetworkData."""
    n = _basic_network_with_battery()
    data = read_network(n)
    assert data.storage_units == ["batt_b3"]
    assert data.storage_bus == {"batt_b3": "b3"}
    assert data.storage_p_nom == {"batt_b3": pytest.approx(50.0)}
    assert data.storage_marginal_cost == {"batt_b3": pytest.approx(5.0)}
    assert data.storage_max_hours == {"batt_b3": pytest.approx(4.0)}
    assert data.storage_efficiency_store == {"batt_b3": pytest.approx(0.95)}
    assert data.storage_efficiency_dispatch == {"batt_b3": pytest.approx(0.95)}
    assert data.storage_index == {"batt_b3": 0}
    assert data.bus_storage == {"b1": [], "b2": [], "b3": ["batt_b3"]}


def test_read_network_storageless_network_unchanged():
    """No storage_units → all storage fields empty, no impact on existing flow."""
    n = pypsa.Network()
    n.set_snapshots(["t1"])
    n.add("Carrier", "AC")
    n.add("Bus", "b1", v_nom=380, carrier="AC")
    n.add("Bus", "b2", v_nom=380, carrier="AC")
    n.add("Line", "l1", bus0="b1", bus1="b2", x=0.1, s_nom=200)
    n.add("Generator", "g1", bus="b1", p_nom=100, marginal_cost=20.0)
    data = read_network(n)
    assert data.storage_units == []
    assert data.storage_bus == {}
    assert data.bus_storage == {"b1": [], "b2": []}


def test_calibrate_recovers_storage_marginal_cost():
    """Phase 2 storage recovery — full intertemporal KKT.

    Build a 2-bus network with a tight cross-line, a cheap upstream
    generator, an expensive local generator, and a battery at the load
    bus. Design the loads so the battery actively arbitrages: charges
    off-peak (when LMP = cheap-gen-cost) and discharges at peak (when
    the battery itself is the marginal price-setter on the local bus).

    Then perturb the battery's marginal_cost (the parameter we want
    to recover) and re-fit. The inverse OPF should walk it back to
    truth using the arbitrage relationship
    ``c_s = λ[t_disp] − λ[t_store] / η_rt``.
    """
    n = pypsa.Network()
    snaps = pd.date_range("2025-04-15", periods=24, freq="h")
    n.set_snapshots(snaps)
    n.add("Carrier", "AC")
    for b in ["b1", "b2"]:
        n.add("Bus", b, v_nom=380, carrier="AC")
    n.add("Line", "l12", bus0="b1", bus1="b2", x=0.1, s_nom=200)
    n.add("Generator", "gen_cheap", bus="b1", p_nom=80,
          marginal_cost=15.0, carrier="AC")
    n.add("Generator", "gen_local", bus="b2", p_nom=30,
          marginal_cost=120.0, carrier="AC")
    true_c_s = 45.0
    n.add(
        "StorageUnit", "batt_b2", bus="b2", p_nom=100, max_hours=4.0,
        marginal_cost=true_c_s,
        efficiency_store=0.95, efficiency_dispatch=0.95, carrier="AC",
    )
    load = 30 + 70 * np.maximum(
        np.sin(np.arange(24) * np.pi / 12 - np.pi / 2), 0,
    )
    n.add("Load", "ld_b2", bus="b2")
    n.loads_t.p_set["ld_b2"] = load
    status, _ = n.optimize(solver_name="highs")
    assert status == "ok"

    # Build observations including storage time-series
    obs = pd.DataFrame(index=snaps)
    for bus in n.buses.index:
        obs[f"price_{bus}"] = n.buses_t.marginal_price[bus]
    for line in n.lines.index:
        obs[f"flow_{line}"] = n.lines_t.p0[line]
    for gen in n.generators.index:
        obs[f"dispatch_{gen}"] = n.generators_t.p[gen]
    for s in n.storage_units.index:
        obs[f"storage_dispatch_{s}"] = n.storage_units_t.p_dispatch[s]
        obs[f"storage_store_{s}"] = n.storage_units_t.p_store[s]
        obs[f"storage_soc_{s}"] = n.storage_units_t.state_of_charge[s]

    # Perturb the storage cost — calibrate should walk it back
    n.storage_units.at["batt_b2", "marginal_cost"] = 20.0

    result = calibrate(
        network=n, observations=obs,
        formulation="noisy", solver="highs",
        active_set_tol=0.5, obs_sigma=1.0, lambda_reg=1e-4,
    )
    assert result.solver_status == "optimal"

    recovered = result.theta_hat.get("storage:batt_b2:marginal_cost")
    assert recovered is not None
    # We expect the recovery to be in the ballpark of truth — exact
    # match would require perfect KKT (the forward LOPF uses cyclic
    # SOC by default which matches our model). Tolerate ~5 EUR/MWh.
    assert abs(recovered - true_c_s) < 10.0, (
        f"storage cost should recover to {true_c_s}; got {recovered}"
    )


def test_storage_recovery_generalises_to_unseen_week():
    """Cross-validation test for storage marginal-cost recovery.

    Calibrate the battery's marginal_cost on week A (real ENTSO-E-style
    arbitrage scenario). Then re-run the forward LOPF on a *different*
    week B (with the same network topology + battery but a *different*
    load profile) using the recovered cost. Verify two things:

    1. **Training-week recovery is accurate** — recovered ≈ truth.
    2. **The recovered cost generalises** — when applied to the unseen
       week B's loads, the forward LOPF reproduces the LMPs that the
       truth-network forward LOPF on week B would produce. This is
       the same train/test protocol we used for the European
       ENTSO-E demo, but for the storage parameter rather than
       generator parameters.
    """
    snaps_a = pd.date_range("2025-04-15", periods=24, freq="h")
    snaps_b = pd.date_range("2025-04-22", periods=24, freq="h")
    true_c_s = 45.0

    def _build(snaps, load_profile):
        n = pypsa.Network()
        n.set_snapshots(snaps)
        n.add("Carrier", "AC")
        for b in ["b1", "b2"]:
            n.add("Bus", b, v_nom=380, carrier="AC")
        n.add("Line", "l12", bus0="b1", bus1="b2", x=0.1, s_nom=200)
        n.add("Generator", "gen_cheap", bus="b1", p_nom=80,
              marginal_cost=15.0, carrier="AC")
        n.add("Generator", "gen_local", bus="b2", p_nom=30,
              marginal_cost=120.0, carrier="AC")
        n.add(
            "StorageUnit", "batt_b2", bus="b2", p_nom=100, max_hours=4.0,
            marginal_cost=true_c_s,
            efficiency_store=0.95, efficiency_dispatch=0.95, carrier="AC",
        )
        n.add("Load", "ld_b2", bus="b2")
        n.loads_t.p_set["ld_b2"] = load_profile
        return n

    # Week A — a known arbitrage-driving load shape
    load_a = 30 + 70 * np.maximum(
        np.sin(np.arange(24) * np.pi / 12 - np.pi / 2), 0,
    )
    # Week B — a DIFFERENT shape: peak shifted earlier, baseline higher
    load_b = 40 + 60 * np.maximum(
        np.sin((np.arange(24) - 2) * np.pi / 12 - np.pi / 2), 0,
    )

    # ---- Phase 1: forward LOPF on truth networks, collect observations
    n_truth_a = _build(snaps_a, load_a)
    n_truth_a.optimize(solver_name="highs")
    n_truth_b = _build(snaps_b, load_b)
    n_truth_b.optimize(solver_name="highs")

    # ---- Phase 2: calibrate on week A, verify recovery
    obs_a = pd.DataFrame(index=snaps_a)
    for bus in n_truth_a.buses.index:
        obs_a[f"price_{bus}"] = n_truth_a.buses_t.marginal_price[bus]
    for line in n_truth_a.lines.index:
        obs_a[f"flow_{line}"] = n_truth_a.lines_t.p0[line]
    for gen in n_truth_a.generators.index:
        obs_a[f"dispatch_{gen}"] = n_truth_a.generators_t.p[gen]
    for s in n_truth_a.storage_units.index:
        obs_a[f"storage_dispatch_{s}"] = n_truth_a.storage_units_t.p_dispatch[s]
        obs_a[f"storage_store_{s}"] = n_truth_a.storage_units_t.p_store[s]
        obs_a[f"storage_soc_{s}"] = n_truth_a.storage_units_t.state_of_charge[s]

    n_biased = _build(snaps_a, load_a)
    n_biased.storage_units.at["batt_b2", "marginal_cost"] = 20.0   # wrong prior
    result = calibrate(
        network=n_biased, observations=obs_a,
        formulation="noisy", solver="highs",
        active_set_tol=0.5, obs_sigma=1.0, lambda_reg=1e-4,
    )
    recovered_c_s = result.theta_hat["storage:batt_b2:marginal_cost"]
    assert abs(recovered_c_s - true_c_s) < 1.0, (
        f"training-week recovery off: got {recovered_c_s} vs truth {true_c_s}"
    )

    # ---- Phase 3: apply recovered θ̂ to a fresh network with WEEK B's loads,
    #             then forward LOPF and compare LMPs to the unseen-week truth.
    from pypsa_invopt import apply
    n_val = _build(snaps_b, load_b)
    n_val.storage_units.at["batt_b2", "marginal_cost"] = 20.0   # start at biased
    apply(result, n_val)
    n_val.optimize(solver_name="highs")

    # Compare LMPs on the *unseen* week B
    truth_b = n_truth_b.buses_t.marginal_price.to_numpy()
    calibrated_b = n_val.buses_t.marginal_price.to_numpy()
    rmse_validation = float(np.sqrt(np.mean((truth_b - calibrated_b) ** 2)))

    # Also compute baseline: forward LOPF on week B with the *biased* cost
    n_biased_b = _build(snaps_b, load_b)
    n_biased_b.storage_units.at["batt_b2", "marginal_cost"] = 20.0
    n_biased_b.optimize(solver_name="highs")
    biased_b = n_biased_b.buses_t.marginal_price.to_numpy()
    rmse_baseline = float(np.sqrt(np.mean((truth_b - biased_b) ** 2)))

    # The calibrated prediction on the unseen week should be *substantially*
    # better than the biased one. Tight tolerance: the recovered θ̂ should
    # reproduce week-B LMPs to within ~1 EUR/MWh because the topology and
    # cost are correct; only the load profile changes.
    assert rmse_validation < 1.0, (
        f"validation LMP RMSE {rmse_validation:.3f} too large — "
        f"the recovered θ̂ does not generalise to the unseen week."
    )
    assert rmse_validation < rmse_baseline, (
        f"calibrated ({rmse_validation:.3f}) should beat biased ({rmse_baseline:.3f}) "
        "on the unseen week."
    )


def test_calibrate_runs_on_network_with_storage_at_bounds():
    """Inverse OPF must not crash on a network with storage; recovery still
    correct when the battery saturates (line/generator duals absorb its role)."""
    n = _basic_network_with_battery()
    status, _ = n.optimize(solver_name="highs")
    assert status == "ok"

    obs = pd.DataFrame(index=n.snapshots.copy())
    for bus in n.buses.index:
        obs[f"price_{bus}"] = n.buses_t.marginal_price[bus]
    for line in n.lines.index:
        obs[f"flow_{line}"] = n.lines_t.p0[line]
    for gen in n.generators.index:
        obs[f"dispatch_{gen}"] = n.generators_t.p[gen]

    # Perturb gen costs; the inverse OPF should recover them
    n.generators.at["gen_base", "marginal_cost"] = 30.0
    n.generators.at["gen_peak", "marginal_cost"] = 60.0

    result = calibrate(
        network=n, observations=obs,
        formulation="noisy", solver="highs",
        active_set_tol=0.5, obs_sigma=1.0, lambda_reg=1e-3,
    )
    assert result.solver_status == "optimal"
    # Generator costs should be back near truth
    recovered_base = result.theta_hat.get("gen:gen_base:marginal_cost")
    recovered_peak = result.theta_hat.get("gen:gen_peak:marginal_cost")
    assert recovered_base is not None
    assert abs(recovered_base - 20.0) < 1.0, (
        f"gen_base should recover to ~20.0, got {recovered_base}"
    )
    # gen_peak may stay at prior if it isn't dispatched — that's the
    # standard "non-marginal = unidentifiable" outcome. We only check
    # the recovery is *finite and sensible*, not exact.
    assert recovered_peak is not None
    assert 0.0 < recovered_peak < 200.0
