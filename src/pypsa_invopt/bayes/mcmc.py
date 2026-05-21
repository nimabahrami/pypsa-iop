"""MCMC posterior via PyMC's NUTS sampler.

Uses the *uncongested* closed-form Gaussian log-likelihood over
generator costs, exposed to PyMC through a ``Potential``. The Laplace
path (:mod:`pypsa_invopt.bayes.laplace`) projects through NNLS on the
congested-line shadow prices; PyMC cannot differentiate through NNLS
without a custom Op, so this module deliberately ignores congestion in
the likelihood. For congested grids prefer the Laplace path.

Requires the optional ``[mcmc]`` extra (pymc>=5.0, arviz>=0.16); the
import is lazy.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from pypsa_invopt.bayes.laplace import _prepare_likelihood_inputs, _resolve_prior_std
from pypsa_invopt.results import PosteriorResult

if TYPE_CHECKING:
    import pandas as pd
    import pypsa

    from pypsa_invopt.results import InverseResult


def mcmc_posterior(
    network: pypsa.Network,
    observations: pd.DataFrame | None,
    result: InverseResult,
    *,
    n_samples: int = 2000,
    n_chains: int = 4,
    prior_std: float | dict[str, float] = 1.0,
    obs_std: float = 5.0,
    target_accept: float = 0.9,
    **kwargs: Any,
) -> PosteriorResult:
    """Draw samples from the full posterior with NUTS.

    Args:
        network: The PyPSA network used during calibration.
        observations: Market data DataFrame. If ``None`` the chain
            samples the prior only.
        result: Calibration result whose ``theta_hat`` initialises NUTS.
        n_samples: Samples per chain after burn-in.
        n_chains: Number of parallel chains.
        prior_std: Prior std (scalar broadcast, or per-parameter dict).
        obs_std: Observation noise std σ_obs (EUR/MWh).
        target_accept: NUTS target acceptance probability.

    Returns:
        A :class:`PosteriorResult` with ``method="mcmc"``, flattened
        samples per parameter, and the full arviz ``InferenceData`` if
        arviz is installed.

    Raises:
        ImportError: If pymc is not installed.
    """
    try:
        import pymc as pm
    except ImportError as exc:
        raise ImportError(
            "MCMC posterior requires PyMC. Install with "
            "'pip install pypsa-invopt[mcmc]'."
        ) from exc

    try:
        import arviz  # noqa: F401 — only used to gate trace return
        have_arviz = True
    except ImportError:
        have_arviz = False

    param_names = sorted(result.theta_hat)
    n_params = len(param_names)
    if n_params == 0:
        return PosteriorResult(method="mcmc", mean={}, parameter_order=())

    theta_map = np.array([result.theta_hat[k] for k in param_names])
    sigma_prior = _resolve_prior_std(prior_std, param_names, n_params)

    _, prices, gen_to_price_col = _prepare_likelihood_inputs(
        network=network, observations=observations, param_names=param_names,
    )

    like_param_idx, obs_sum, obs_sum_sq, t_obs = _precompute_likelihood_stats(
        prices=prices,
        gen_to_price_col=gen_to_price_col,
        param_names=param_names,
    )

    with pm.Model():
        theta = pm.Normal(
            "theta", mu=theta_map, sigma=sigma_prior, shape=n_params,
        )
        if like_param_idx.size > 0:
            inv_var = 1.0 / (obs_std ** 2)
            theta_like = theta[like_param_idx]
            # Closed-form per-parameter SSE: expanding (θ_i - p_t)² and
            # summing over t leaves only sums of p and p² to pre-compute.
            sse = (
                t_obs * theta_like ** 2
                - 2.0 * theta_like * obs_sum
                + obs_sum_sq
            )
            log_lik = -0.5 * inv_var * pm.math.sum(sse) / max(t_obs, 1.0)
            pm.Potential("log_likelihood", log_lik)

        trace = pm.sample(
            draws=n_samples,
            chains=n_chains,
            target_accept=target_accept,
            return_inferencedata=True,
            progressbar=False,
            initvals={"theta": theta_map},
        )

    samples = _flatten_samples(trace, param_names)
    return PosteriorResult(
        method="mcmc",
        mean={name: float(np.mean(values)) for name, values in samples.items()},
        samples=samples,
        arviz_data=trace if have_arviz else None,
        parameter_order=tuple(param_names),
    )


def _flatten_samples(trace, param_names: list[str]) -> dict[str, np.ndarray]:
    """Reshape ``(chains, draws, params)`` into ``{param_name: 1-D array}``."""
    flat = trace.posterior["theta"].values.reshape(-1, len(param_names))
    return {name: flat[:, i] for i, name in enumerate(param_names)}


def _precompute_likelihood_stats(
    *,
    prices: np.ndarray | None,
    gen_to_price_col: dict[str, int],
    param_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Pre-compute the observed-price moments the closed-form SSE uses.

    Returns ``(like_param_idx, obs_sum, obs_sum_sq, t_obs)``. When the
    likelihood is inactive (no observations or no matching parameters)
    ``like_param_idx`` is empty and the other arrays carry zeros.
    """
    if prices is None or not gen_to_price_col:
        empty = np.array([], dtype=int)
        return empty, np.zeros(0), np.zeros(0), 0.0

    like_param_idx = np.array(
        [i for i, name in enumerate(param_names) if name in gen_to_price_col],
        dtype=int,
    )
    like_price_col = np.array(
        [gen_to_price_col[param_names[i]] for i in like_param_idx],
        dtype=int,
    )
    obs_sum = prices[:, like_price_col].sum(axis=0)
    obs_sum_sq = (prices[:, like_price_col] ** 2).sum(axis=0)
    return like_param_idx, obs_sum, obs_sum_sq, float(prices.shape[0])


__all__ = ["mcmc_posterior"]
