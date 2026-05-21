"""Noisy formulation tests."""
import numpy as np

from pypsa_invopt.formulations.noisy import NoisyFormulation
from pypsa_invopt.network import read_network
from pypsa_invopt.utils.active_set import ActiveSet, ActiveSetBatch


def test_noisy_recovery_with_noise(two_bus_network):
    """Noisy formulation recovers costs under Gaussian noise."""
    net_data = read_network(two_bus_network)
    rng = np.random.default_rng(42)
    T = 24
    true_prices = np.tile([20.0, 50.0], (T, 1))
    noisy_prices = true_prices + rng.normal(0, 3.0, (T, 2))

    batch = ActiveSetBatch(
        pattern=ActiveSet(), timestep_indices=list(range(T)), cluster_id=0,
    )
    form = NoisyFormulation()
    model = form.build_model(
        net_data, batch, {"prices": noisy_prices},
        lambda_reg=0.01, obs_sigma=3.0,
    )
    result = form.solve(model)

    assert result["status"] in ("optimal", "globallyoptimal", "locallyoptimal")
    c_gA = result["theta"]["gen:gA:marginal_cost"]
    c_gB = result["theta"]["gen:gB:marginal_cost"]
    # Should be close to true values
    assert abs(c_gA - 20.0) < 15.0
    assert abs(c_gB - 50.0) < 15.0


def test_regularization_improves_recovery(two_bus_network):
    """Higher regularization toward prior → closer to prior."""
    net_data = read_network(two_bus_network)
    rng = np.random.default_rng(42)
    T = 24
    noisy_prices = np.tile([20.0, 50.0], (T, 1)) + rng.normal(0, 10.0, (T, 2))

    batch = ActiveSetBatch(
        pattern=ActiveSet(), timestep_indices=list(range(T)), cluster_id=0,
    )
    form = NoisyFormulation()

    # Low regularization
    m_low = form.build_model(
        net_data, batch, {"prices": noisy_prices}, lambda_reg=0.001,
    )
    r_low = form.solve(m_low)

    # High regularization
    m_high = form.build_model(
        net_data, batch, {"prices": noisy_prices}, lambda_reg=10.0,
    )
    r_high = form.solve(m_high)

    # High-reg result should be closer to network defaults
    c_gA_low = r_low["theta"]["gen:gA:marginal_cost"]
    c_gA_high = r_high["theta"]["gen:gA:marginal_cost"]
    prior_gA = 20.0  # network default

    assert abs(c_gA_high - prior_gA) <= abs(c_gA_low - prior_gA) + 1.0


def test_noisy_with_active_lines_and_maxed_gen():
    """Batches that exercise both μ (line shadow) and ν (maxed-gen) slacks
    must solve cleanly.

    Without the generator-bound slack variables in the KKT residual,
    non-marginal-generator recovery is biased; this test guards them.
    """
    import pandas as pd
    import pypsa

    from pypsa_invopt.calibration import calibrate

    network = pypsa.Network()
    network.set_snapshots(pd.date_range("2025-01-01", periods=6, freq="h"))
    network.add("Carrier", "AC")
    network.add("Bus", "A", v_nom=110.0, carrier="AC")
    network.add("Bus", "B", v_nom=110.0, carrier="AC")
    network.add("Line", "A-B", bus0="A", bus1="B", s_nom=100.0, x=0.1, r=0.0)
    network.add("Generator", "gA", bus="A", p_nom=100.0, marginal_cost=20.0)
    network.add("Generator", "gB", bus="B", p_nom=100.0, marginal_cost=50.0)
    network.add("Load", "load_B", bus="B", p_set=50.0)

    n_t = len(network.snapshots)
    obs = pd.DataFrame(
        {
            "price_A": [20.0] * n_t,
            "price_B": [50.0] * n_t,
            "flow_A-B": [100.0] * n_t,     # at s_nom → congested
            "dispatch_gA": [100.0] * n_t,  # at p_nom → maxed
            "dispatch_gB": [50.0] * n_t,
        },
        index=network.snapshots,
    )

    result = calibrate(
        network=network,
        observations=obs,
        formulation="noisy",
        solver="highs",
        active_set_tol=0.5,
        lambda_reg=0.01,
        obs_sigma=2.0,
    )

    assert result.solver_status == "optimal"
    assert result.n_active_sets >= 1
    for value in result.theta_hat.values():
        assert np.isfinite(value)
        assert value > 0
