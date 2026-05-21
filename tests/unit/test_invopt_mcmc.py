"""MCMC posterior tests — gated on PyMC availability.

The MCMC posterior is an optional extra (``pip install
pypsa-invopt[mcmc]``); without PyMC installed these tests skip
rather than fail.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pypsa
import pytest

pymc = pytest.importorskip("pymc")

from pypsa_invopt import calibrate, posterior  # noqa: E402 (must follow importorskip)


def _build_2bus_mcmc_network() -> pypsa.Network:
    n = pypsa.Network()
    snaps = pd.date_range("2025-04-15", periods=6, freq="h")
    n.set_snapshots(snaps)
    n.add("Carrier", "AC")
    for b in ["b1", "b2"]:
        n.add("Bus", b, v_nom=380, carrier="AC")
    n.add("Line", "l12", bus0="b1", bus1="b2", x=0.1, s_nom=200)
    n.add("Generator", "gen1", bus="b1", p_nom=80,
          marginal_cost=20.0, carrier="AC")
    n.add("Generator", "gen2", bus="b2", p_nom=80,
          marginal_cost=60.0, carrier="AC")
    n.add("Load", "ld_b2", bus="b2")
    n.loads_t.p_set["ld_b2"] = [30, 50, 80, 100, 70, 40]
    return n


@pytest.mark.slow
def test_mcmc_posterior_returns_samples():
    """Smoke test: NUTS runs, returns sample arrays for each θ."""
    n = _build_2bus_mcmc_network()
    n.optimize(solver_name="highs")
    obs = pd.DataFrame(index=n.snapshots)
    for bus in n.buses.index:
        obs[f"price_{bus}"] = n.buses_t.marginal_price[bus]
    for line in n.lines.index:
        obs[f"flow_{line}"] = n.lines_t.p0[line]
    for gen in n.generators.index:
        obs[f"dispatch_{gen}"] = n.generators_t.p[gen]

    result = calibrate(
        network=n, observations=obs, formulation="noisy",
        solver="highs", lambda_reg=1e-4, obs_sigma=1.0,
    )

    post = posterior(
        network=n, observations=obs, result=result,
        method="mcmc",
        n_samples=50, n_chains=2,    # tiny budget for CI
        prior_std=5.0, obs_std=1.0,
        target_accept=0.85,
    )

    assert post.method == "mcmc"
    assert post.samples is not None
    # Each tracked parameter should have ~n_samples*n_chains draws
    for arr in post.samples.values():
        assert isinstance(arr, np.ndarray)
        assert arr.size > 0
