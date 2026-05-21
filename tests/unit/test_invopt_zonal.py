"""Zonal formulation tests."""
import numpy as np

from pypsa_invopt.formulations.zonal import ZonalFormulation
from pypsa_invopt.network import read_network
from pypsa_invopt.utils.active_set import ActiveSet, ActiveSetBatch


def test_zonal_5zone_cost_recovery(five_zone_network):
    """Zonal formulation recovers zone-level costs."""
    zones = ["Z1", "Z2", "Z3", "Z4", "Z5"]
    true_costs = [20.0, 30.0, 40.0, 50.0, 60.0]

    T = 24
    prices = np.tile(true_costs, (T, 1))

    batch = ActiveSetBatch(
        pattern=ActiveSet(), timestep_indices=list(range(T)), cluster_id=0,
    )

    net_data = read_network(five_zone_network)
    form = ZonalFormulation()
    model = form.build_model(
        net_data, batch, {"prices": prices},
        zones=zones,
        bus_to_zone={z: z for z in zones},
        lambda_reg=0.001,
    )
    result = form.solve(model)

    assert result["status"] in ("optimal", "globallyoptimal", "locallyoptimal")
    for z, true_c in zip(zones, true_costs, strict=True):
        recovered = result["theta"][f"zone:{z}:marginal_cost"]
        assert abs(recovered - true_c) < 5.0, (
            f"Zone {z}: recovered {recovered:.1f}, expected ~{true_c}"
        )


def test_zonal_recovers_ntc_from_congested_flows(five_zone_network):
    """Implicit NTC = max |observed flow| on congested pair-timesteps.

    Setup: two zones with a 30 EUR/MWh price differential. The inter-
    zone flow is observed at 280 MW with an NTC prior of 300 MW (so the
    detector flags it congested at ntc_active_tol=0.9). The recovered
    NTC must be ≥ 280 MW (the binding observation) and the recovered
    shadow price magnitudes should reflect the 30 EUR/MWh spread.
    """
    zones = ["Z1", "Z2"]
    T = 12
    # Z1 is cheaper exporter (20), Z2 is importer at higher price (50).
    prices = np.tile([20.0, 50.0], (T, 1))
    # Observed flow z1 → z2 at near-NTC magnitude.
    f_obs = np.full(T, 280.0)

    batch = ActiveSetBatch(
        pattern=ActiveSet(), timestep_indices=list(range(T)), cluster_id=0,
    )

    net_data = read_network(five_zone_network)
    form = ZonalFormulation()
    model = form.build_model(
        net_data, batch, {"prices": prices},
        zones=zones,
        bus_to_zone={z: z for z in zones},
        zone_pairs=[("Z1", "Z2")],
        flows_per_pair={"Z1_Z2": f_obs},
        ntc_prior={("Z1", "Z2"): 300.0},
        lambda_reg=0.001,
        ntc_active_tol=0.9,
    )
    result = form.solve(model)

    assert result["status"] in ("optimal", "globallyoptimal", "locallyoptimal")
    # Recovered NTC must be at least the observed flow (constraint
    # ntc[zp] >= |f*[zp, t]| for congested timesteps).
    assert result["theta"]["ntc:Z1_Z2"] >= 280.0 - 1e-6
    # Shadow-price diagnostic should be present and non-trivial.
    assert "ntc_shadow_mean:Z1_Z2" in result["theta"]
    assert abs(result["theta"]["ntc_shadow_mean:Z1_Z2"]) > 1.0


def test_zonal_no_flows_falls_back_to_cost_only(five_zone_network):
    """When flows_per_pair is empty, the model reduces to cost-only.

    No congestion is detected → all μ_pair pinned to zero by the
    uncongested constraint → λ_z = c_z, matching the original behaviour
    that the existing test relies on.
    """
    zones = ["Z1", "Z2"]
    T = 4
    prices = np.tile([25.0, 25.0], (T, 1))

    batch = ActiveSetBatch(
        pattern=ActiveSet(), timestep_indices=list(range(T)), cluster_id=0,
    )

    net_data = read_network(five_zone_network)
    form = ZonalFormulation()
    model = form.build_model(
        net_data, batch, {"prices": prices},
        zones=zones,
        bus_to_zone={z: z for z in zones},
        zone_pairs=[("Z1", "Z2")],
        ntc_prior={("Z1", "Z2"): 300.0},
        lambda_reg=0.001,
    )
    result = form.solve(model)

    # Without observed flows, no congestion ⇒ no shadow price ⇒
    # c_z fits the observed prices directly.
    for z in zones:
        recovered = result["theta"][f"zone:{z}:marginal_cost"]
        assert abs(recovered - 25.0) < 2.0
    # No shadow-price diagnostic should be present.
    assert "ntc_shadow_mean:Z1_Z2" not in result["theta"]
