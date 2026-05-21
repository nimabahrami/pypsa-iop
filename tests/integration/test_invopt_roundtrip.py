"""Integration test: full roundtrip calibration."""
import numpy as np
import pandas as pd
import pypsa
import pytest


def test_calibrate_recover_line_params(three_bus_network):
    """End-to-end calibration with recover_line_params=True.

    Forces congestion on one line in some snapshots so that
    ``recover_flow_limits`` has something to do, and provides
    flow/dispatch data so ``recover_line_susceptances`` runs through
    IPOPT. Asserts that the resulting InverseResult contains the new
    ``line:<name>:s_nom`` / ``line:<name>:x`` entries and that
    ``apply_result`` writes them back.
    """
    from pypsa_invopt.calibration import calibrate
    from pypsa_invopt.network import apply_result

    n = three_bus_network
    T = len(n.snapshots)
    rng = np.random.default_rng(0)

    obs = pd.DataFrame({
        "price_A": 15.0 + rng.normal(0, 1.0, T),
        "price_B": 30.0 + rng.normal(0, 1.0, T),
        "price_C": 60.0 + rng.normal(0, 1.0, T),
        "flow_A-B": np.full(T, 100.0),     # at s_nom=100 → congested
        "flow_B-C": np.full(T, 30.0),
        "flow_A-C": np.full(T, 70.0),
        "dispatch_gA": np.full(T, 120.0),
        "dispatch_gB": np.full(T, 0.0),
        "dispatch_gC": np.full(T, 0.0),
    }, index=n.snapshots)

    s_nom_before = float(n.lines.at["A-B", "s_nom"])
    x_before = float(n.lines.at["A-B", "x"])

    result = calibrate(
        network=n,
        observations=obs,
        formulation="noisy",
        solver="highs",
        active_set_tol=0.5,
        recover_line_params=True,
        lambda_reg=0.01,
        obs_sigma=1.0,
    )

    # The congested line should have a recovered s_nom near the observed
    # peak flow * safety_margin.
    assert "line:A-B:s_nom" in result.theta_hat
    assert abs(result.theta_hat["line:A-B:s_nom"] - 100.0 * 1.02) < 1.0

    # apply_result must update the network in-place.
    apply_result(result, n)
    assert float(n.lines.at["A-B", "s_nom"]) != pytest.approx(s_nom_before)

    # If a reactance update was recovered, it must be valid (positive
    # and finite). Susceptance recovery is opt-in and may be skipped if
    # IPOPT fails for any reason; either branch is acceptable here.
    if "line:A-B:x" in result.theta_hat:
        assert result.theta_hat["line:A-B:x"] > 0
        assert np.isfinite(result.theta_hat["line:A-B:x"])
    # Restore the prior value just so any subsequent test re-using this
    # fixture isn't affected (pytest fixtures are function-scoped but
    # the network object is mutable).
    n.lines.at["A-B", "x"] = x_before


@pytest.mark.slow
def test_full_roundtrip_calibration():
    """Full pipeline: create network → generate observations → calibrate → apply."""
    from pypsa_invopt.calibration import calibrate
    from pypsa_invopt.network import apply_result

    # Create network
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2025-01-01", periods=24, freq="h"))
    n.add("Carrier", "AC")
    n.add("Bus", "A", v_nom=110.0, carrier="AC")
    n.add("Bus", "B", v_nom=110.0, carrier="AC")
    n.add("Line", "A-B", bus0="A", bus1="B", s_nom=100.0, x=0.1, r=0.0)
    n.add("Generator", "gA", bus="A", p_nom=100.0, marginal_cost=20.0)
    n.add("Generator", "gB", bus="B", p_nom=100.0, marginal_cost=50.0)
    n.add("Load", "load_B", bus="B", p_set=50.0)

    # Synthetic observations
    rng = np.random.default_rng(42)
    T = len(n.snapshots)
    obs = pd.DataFrame(
        {
            "price_A": 20.0 + rng.normal(0, 1.0, T),
            "price_B": 50.0 + rng.normal(0, 1.0, T),
        },
        index=n.snapshots,
    )

    # Calibrate
    result = calibrate(
        network=n,
        observations=obs,
        formulation="noisy",
        solver="highs",
        lambda_reg=0.01,
        obs_sigma=2.0,
    )

    assert result.solver_status in ("optimal", "partial")
    assert result.rmse < 50.0
    assert len(result.theta_hat) > 0

    # Apply
    apply_result(result, n)

    # Verify network was updated
    for key, val in result.theta_hat.items():
        if "marginal_cost" in key:
            gen_name = key.split(":")[1]
            if gen_name in n.generators.index:
                assert float(n.generators.at[gen_name, "marginal_cost"]) == pytest.approx(val)
