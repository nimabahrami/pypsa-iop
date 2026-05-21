"""Bayesian posterior sub-package.

Public entry point: :func:`posterior`, which dispatches to either the
Laplace approximation (default, fast) or NUTS-MCMC (full, requires the
``[mcmc]`` extra).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pypsa_invopt.bayes.laplace import laplace_posterior

if TYPE_CHECKING:
    import pandas as pd
    import pypsa

    from pypsa_invopt.results import InverseResult, PosteriorResult

PosteriorMethod = Literal["laplace", "mcmc"]


def posterior(
    network: pypsa.Network,
    observations: pd.DataFrame | None,
    result: InverseResult,
    *,
    method: PosteriorMethod = "laplace",
    **kwargs: Any,
) -> PosteriorResult:
    """Compute the Bayesian posterior over recovered parameters.

    Args:
        network: The calibrated ``pypsa.Network``.
        observations: Market observation DataFrame (may be ``None`` to
            recover the prior-only fallback).
        result: The :class:`InverseResult` from
            :func:`pypsa_invopt.calibrate`, used as the MAP centre.
        method: ``"laplace"`` for the fast Hessian-based Gaussian
            posterior, ``"mcmc"`` for the full NUTS sampling path.
        **kwargs: Forwarded to the method-specific implementation
            (e.g. ``prior_std``, ``obs_std``, ``n_samples``).

    Raises:
        ValueError: If ``method`` is not one of the supported labels.
    """
    if method == "laplace":
        return laplace_posterior(
            network=network, observations=observations, result=result, **kwargs,
        )
    if method == "mcmc":
        from pypsa_invopt.bayes.mcmc import mcmc_posterior
        return mcmc_posterior(
            network=network, observations=observations, result=result, **kwargs,
        )
    raise ValueError(
        f"Unknown posterior method '{method}'. Use 'laplace' or 'mcmc'."
    )


__all__ = ["PosteriorMethod", "posterior"]
