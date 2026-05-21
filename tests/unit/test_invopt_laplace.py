"""Laplace posterior tests."""
import numpy as np
import pandas as pd

from pypsa_invopt.bayes.laplace import laplace_posterior
from pypsa_invopt.results import InverseResult


def test_laplace_returns_covariance(two_bus_network):
    """Laplace posterior produces a valid covariance matrix."""
    result = InverseResult(
        theta_hat={"gen:gA:marginal_cost": 20.0, "gen:gB:marginal_cost": 50.0},
        rmse=0.5, kkt_residuals=np.array([0.1, 0.2]),
        active_set={}, solver_status="optimal", formulation="noisy",
    )

    post = laplace_posterior(
        network=two_bus_network,
        observations=None,
        result=result,
        prior_std=1.0,
        finite_diff_eps=1e-3,
    )

    assert post.method == "laplace"
    assert post.cov is not None
    assert post.cov.shape == (2, 2)
    # Covariance should be symmetric
    np.testing.assert_allclose(post.cov, post.cov.T, atol=1e-10)
    # Diagonal should be positive
    assert np.all(np.diag(post.cov) > 0)
    # Mean should match MAP
    assert abs(post.mean["gen:gA:marginal_cost"] - 20.0) < 1e-6


def test_laplace_empty_params(two_bus_network):
    """Empty parameter set returns zero-size covariance."""
    result = InverseResult(
        theta_hat={}, rmse=0.0, kkt_residuals=np.array([]),
        active_set={}, solver_status="optimal", formulation="noisy",
    )
    post = laplace_posterior(
        network=two_bus_network, observations=None, result=result,
    )
    assert post.cov.shape == (0, 0)


def test_laplace_data_shrinks_posterior(two_bus_network):
    """With observations, the posterior std must be tighter than the prior.

    This is the load-bearing check that the KKT-residual likelihood is
    actually wired into the Hessian — before the fix, the returned cov
    was identically the prior, so the with-data and no-data runs were
    indistinguishable.
    """
    map_theta = {"gen:gA:marginal_cost": 20.0, "gen:gB:marginal_cost": 50.0}
    result = InverseResult(
        theta_hat=map_theta,
        rmse=0.5, kkt_residuals=np.array([0.1]),
        active_set={}, solver_status="optimal", formulation="noisy",
    )

    # Synthetic observations centred at the MAP, low noise (so the
    # likelihood is informative).
    T = 48
    idx = pd.date_range("2025-01-01", periods=T, freq="h")
    rng = np.random.default_rng(0)
    obs = pd.DataFrame({
        "price_A": 20.0 + rng.normal(0, 0.5, T),
        "price_B": 50.0 + rng.normal(0, 0.5, T),
    }, index=idx)

    prior_only = laplace_posterior(
        network=two_bus_network, observations=None, result=result,
        prior_std=2.0,
    )
    with_data = laplace_posterior(
        network=two_bus_network, observations=obs, result=result,
        prior_std=2.0, obs_std=0.5,
    )

    # Diagonal stds with data must be strictly smaller than prior-only
    # for both generators.
    prior_diag = np.sqrt(np.diag(prior_only.cov))
    data_diag = np.sqrt(np.diag(with_data.cov))
    assert np.all(data_diag < prior_diag - 1e-6), (
        f"posterior std did not shrink: prior={prior_diag}, with_data={data_diag}"
    )


def test_laplace_congested_projection_changes_posterior(three_bus_network):
    """When the active set is non-empty, the NNLS-projected likelihood
    differs from the uncongested fallback.

    Verifies the load-bearing claim of the new ``_LikelihoodContext``:
    a congested timestep routes through ``scipy.optimize.nnls`` and the
    resulting projected SSE is different from the trivial ``c - λ_obs``
    SSE the previous version would have computed. The covariance must
    stay finite and positive-definite either way.
    """
    map_theta = {
        "gen:gA:marginal_cost": 15.0,
        "gen:gB:marginal_cost": 30.0,
        "gen:gC:marginal_cost": 60.0,
    }
    T = 24
    idx = pd.date_range("2025-01-01", periods=T, freq="h")
    rng = np.random.default_rng(11)
    obs = pd.DataFrame(
        {
            "price_A": 15.0 + rng.normal(0, 0.5, T),
            "price_B": 30.0 + rng.normal(0, 0.5, T),
            "price_C": 60.0 + rng.normal(0, 0.5, T),
        },
        index=idx,
    )

    no_congestion = {
        t: {"congested_lines": [], "maxed_generators": []} for t in range(T)
    }
    with_congestion = {
        t: {"congested_lines": ["A-B"], "maxed_generators": []} for t in range(T)
    }

    def _result(active_set: dict) -> InverseResult:
        return InverseResult(
            theta_hat=map_theta,
            rmse=0.5,
            kkt_residuals=np.array([0.1]),
            active_set=active_set,
            solver_status="optimal",
            formulation="noisy",
        )

    post_baseline = laplace_posterior(
        network=three_bus_network,
        observations=obs,
        result=_result(no_congestion),
        prior_std=2.0,
        obs_std=0.5,
    )
    post_projected = laplace_posterior(
        network=three_bus_network,
        observations=obs,
        result=_result(with_congestion),
        prior_std=2.0,
        obs_std=0.5,
    )

    diag_baseline = np.sqrt(np.diag(post_baseline.cov))
    diag_projected = np.sqrt(np.diag(post_projected.cov))
    assert not np.allclose(diag_baseline, diag_projected, atol=1e-4), (
        f"projection had no effect: baseline={diag_baseline}, "
        f"projected={diag_projected}"
    )
    assert np.all(np.isfinite(diag_projected))
    eigvals = np.linalg.eigvalsh(post_projected.cov)
    assert np.all(eigvals > 0), f"projected cov not PSD: eigvals={eigvals}"
