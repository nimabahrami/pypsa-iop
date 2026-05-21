"""Result dataclasses returned by the public API.

Two frozen records, one per pipeline stage:
:class:`InverseResult` (calibration) and :class:`PosteriorResult`
(posterior over recovered parameters).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class InverseResult:
    """Result of an inverse OPF calibration.

    Attributes:
        theta_hat: Recovered parameters keyed by component name.
            Keys follow ``'gen:{name}:marginal_cost'`` and
            ``'line:{name}:susceptance'`` naming conventions.
        rmse: RMSE of simulated vs observed prices (EUR/MWh).
        kkt_residuals: Per-timestep KKT residual norms. Shape ``(T,)``.
        active_set: Which constraints were binding per time period.
            Keys are timestep indices, values are dicts with
            ``'congested_lines'`` and ``'maxed_generators'`` lists.
        solver_status: ``'optimal'``, ``'feasible'``, or ``'infeasible'``.
        formulation: Which formulation was used (``'noiseless'``,
            ``'noisy'``, ``'zonal'``).
        warnings: Data quality warnings from validation.
        n_active_sets: Number of unique active-set patterns detected (K).
        wall_time_s: Total wall-clock time in seconds.
    """

    theta_hat: dict[str, float]
    rmse: float
    kkt_residuals: np.ndarray
    active_set: dict[int, dict[str, list[str]]]
    solver_status: str
    formulation: str
    warnings: list[str] = field(default_factory=list)
    n_active_sets: int = 0
    wall_time_s: float = 0.0


@dataclass
class PosteriorResult:
    """Result of Bayesian posterior computation.

    Attributes:
        method: ``'laplace'`` or ``'mcmc'``.
        mean: Posterior mean per parameter (same keys as
            ``InverseResult.theta_hat``).
        cov: Covariance matrix (Laplace only). Shape
            ``(n_params, n_params)``. ``None`` for MCMC.
        samples: MCMC samples per parameter (MCMC only).
            Keys match ``mean``, values are 1-D arrays of length
            ``n_samples``. ``None`` for Laplace.
        arviz_data: Full arviz ``InferenceData`` object (MCMC only).
            ``None`` for Laplace, or if arviz is not installed.
        parameter_order: Ordered tuple of parameter names matching
            the covariance matrix rows/columns.
    """

    method: str
    mean: dict[str, float]
    cov: np.ndarray | None = None
    samples: dict[str, np.ndarray] | None = None
    arviz_data: Any = None  # arviz.InferenceData when available
    parameter_order: tuple[str, ...] = ()


__all__ = [
    "InverseResult",
    "PosteriorResult",
]
