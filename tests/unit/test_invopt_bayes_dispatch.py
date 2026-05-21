"""Unit tests for the bayes/__init__.py method dispatch."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pypsa
import pytest

from pypsa_invopt import calibrate, posterior
from pypsa_invopt.results import InverseResult


def _tiny_network() -> tuple[pypsa.Network, pd.DataFrame, InverseResult]:
    """Build the smallest possible network + observations + InverseResult
    that the dispatch can chew on."""
    n = pypsa.Network()
    snaps = pd.date_range("2025-04-15", periods=4, freq="h")
    n.set_snapshots(snaps)
    n.add("Carrier", "AC")
    n.add("Bus", "b1", v_nom=380, carrier="AC")
    n.add("Bus", "b2", v_nom=380, carrier="AC")
    n.add("Line", "l12", bus0="b1", bus1="b2", x=0.1, s_nom=200)
    n.add("Generator", "g_base", bus="b1", p_nom=60, marginal_cost=20.0)
    n.add("Generator", "g_peak", bus="b1", p_nom=60, marginal_cost=80.0)
    n.add("Load", "ld", bus="b2")
    n.loads_t.p_set["ld"] = pd.Series([40, 60, 90, 50], index=snaps).astype(float)

    status, _ = n.optimize(solver_name="highs")
    assert status == "ok"

    from pypsa_invopt import observations_from_pypsa
    obs = observations_from_pypsa(n)
    res = calibrate(
        network=n, observations=obs,
        formulation="noisy", solver="highs", lambda_reg=1e-5,
    )
    return n, obs, res


def test_posterior_dispatch_to_laplace_default():
    """Calling ``posterior(..., method='laplace')`` (the default) goes
    through the dispatch and returns a finite Σ_post."""
    n, obs, res = _tiny_network()
    post = posterior(n, obs, res, method="laplace",
                     prior_std=5.0, obs_std=1.0)
    assert post.mean
    assert all(np.isfinite(list(post.mean.values())))


def test_posterior_dispatch_default_argument():
    """No ``method=`` defaults to Laplace."""
    n, obs, res = _tiny_network()
    post = posterior(n, obs, res, prior_std=5.0, obs_std=1.0)
    assert post.mean


def test_posterior_dispatch_unknown_method_raises():
    """An unknown method label must raise ValueError with a helpful message."""
    n, obs, res = _tiny_network()
    with pytest.raises(ValueError, match=r"Unknown posterior method"):
        posterior(n, obs, res, method="vi")


def test_posterior_dispatch_mcmc_branch_when_pymc_missing(monkeypatch):
    """The 'mcmc' branch lazily imports pymc; without pymc installed it
    must surface an ImportError, not silently fall through to Laplace."""
    pytest.importorskip("pypsa_invopt.bayes.mcmc")
    pymc_available = True
    try:
        import pymc  # noqa: F401
    except ImportError:
        pymc_available = False

    n, obs, res = _tiny_network()
    if pymc_available:
        # Real run — just confirm it returns something
        from pypsa_invopt.bayes.mcmc import mcmc_posterior  # noqa: F401
        post = posterior(n, obs, res, method="mcmc", n_samples=50, n_tune=50)
        assert post.mean
    else:
        with pytest.raises(ImportError):
            posterior(n, obs, res, method="mcmc")
