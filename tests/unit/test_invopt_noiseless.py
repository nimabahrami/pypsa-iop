"""Noiseless formulation tests."""
import numpy as np

from pypsa_invopt.formulations.noiseless import NoiselessFormulation
from pypsa_invopt.network import read_network
from pypsa_invopt.utils.active_set import ActiveSet, ActiveSetBatch


def test_noiseless_2bus_recovery(two_bus_network):
    """Roundtrip: known costs → simulate prices → recover costs."""
    net_data = read_network(two_bus_network)

    # Synthetic observations: prices = marginal costs (no congestion)
    T = 4
    prices = np.array([
        [20.0, 50.0],  # bus A, bus B
        [20.0, 50.0],
        [20.0, 50.0],
        [20.0, 50.0],
    ])

    batch = ActiveSetBatch(
        pattern=ActiveSet(),  # No congestion
        timestep_indices=list(range(T)),
        cluster_id=0,
    )
    obs = {"prices": prices}

    form = NoiselessFormulation()
    model = form.build_model(net_data, batch, obs)
    result = form.solve(model)

    assert result["status"] in ("optimal", "globallyoptimal", "locallyoptimal")
    # Recovered costs should match
    c_gA = result["theta"]["gen:gA:marginal_cost"]
    c_gB = result["theta"]["gen:gB:marginal_cost"]
    assert abs(c_gA - 20.0) < 5.0  # Within 5 EUR/MWh
    assert abs(c_gB - 50.0) < 5.0


def test_noiseless_residuals_near_zero(two_bus_network):
    """Noiseless with consistent data should give near-zero residuals."""
    net_data = read_network(two_bus_network)
    T = 4
    prices = np.tile([20.0, 50.0], (T, 1))

    batch = ActiveSetBatch(
        pattern=ActiveSet(), timestep_indices=list(range(T)), cluster_id=0,
    )
    form = NoiselessFormulation()
    model = form.build_model(net_data, batch, {"prices": prices})
    result = form.solve(model)

    assert np.all(result["residuals"] < 1.0)
