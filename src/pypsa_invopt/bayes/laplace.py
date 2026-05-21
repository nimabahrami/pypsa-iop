"""Laplace approximation of the parameter posterior.

Negative log-posterior::

    -log p(θ|obs) = (1 / 2σ²_obs) · Σ_t ‖r*_t(θ)‖²
                  + (1/2) (θ − θ₀)ᵀ Σ₀⁻¹ (θ − θ₀)

``r*_t(θ) = min_{μ ≥ 0} ‖r_t(θ, μ)‖²`` is the active-set KKT residual
projected onto the non-negative orthant of line shadow prices. For
uncongested snapshots the projection is trivial (``r*_t = c − λ_obs``);
for congested ones we run NNLS (Lawson-Hanson 1974) via
:func:`scipy.optimize.nnls`. At the MAP the Laplace covariance is the
inverse finite-difference Hessian of the projected functional.

References: Stuart (2010) *Acta Numerica* 19 (Bayesian inverse-problem
framework); Aswani-Shen-Siddiq (2018) *OR* 66(3) (KKT-residual loss);
Liang-Dvorkin (2023) eq. (11a) (single-level data-driven IO); Bishop
(2006) PRML §4.4 (Laplace derivation).
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import nnls

from pypsa_invopt.network import read_network
from pypsa_invopt.results import PosteriorResult

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pandas as pd
    import pypsa

    from pypsa_invopt.network import InvoptNetworkData
    from pypsa_invopt.results import InverseResult


def laplace_posterior(
    network: pypsa.Network,
    observations: pd.DataFrame | None,
    result: InverseResult,
    *,
    prior_std: float | dict[str, float] = 1.0,
    obs_std: float = 5.0,
    finite_diff_eps: float = 1e-2,
    **kwargs: Any,
) -> PosteriorResult:
    """Compute ``N(theta_MAP, H⁻¹)`` at the MAP.

    Args:
        network: The ``pypsa.Network`` used during calibration. Required
            to look up which observed-price column matches each
            generator parameter and to obtain the PTDF used by the
            active-set projection.
        observations: Market data DataFrame. Must contain
            ``price_<bus>`` columns to engage the likelihood. Pass
            ``None`` to fall back to a prior-only posterior.
        result: Calibration result whose ``theta_hat`` is the MAP and
            whose ``active_set`` map drives the μ-projection.
        prior_std: Prior std in parameter space. Scalar broadcast
            across all parameters, or a per-parameter dict.
        obs_std: Observation noise std σ_obs (EUR/MWh).
        finite_diff_eps: Step size for the central-difference Hessian.

    Returns:
        A :class:`PosteriorResult` with ``method="laplace"``.
    """
    param_names = sorted(result.theta_hat)
    n_params = len(param_names)
    if n_params == 0:
        return PosteriorResult(
            method="laplace",
            mean={},
            cov=np.zeros((0, 0)),
            parameter_order=(),
        )

    theta_map = np.array([result.theta_hat[k] for k in param_names])
    sigma_prior = _resolve_prior_std(prior_std, param_names, n_params)

    likelihood_ctx = _build_likelihood_context(
        network=network,
        observations=observations,
        result=result,
        param_names=param_names,
        obs_std=obs_std,
    )

    neg_log_post = _make_neg_log_posterior(
        theta_map=theta_map,
        sigma_prior=sigma_prior,
        likelihood_ctx=likelihood_ctx,
    )
    hessian = _compute_hessian(neg_log_post, theta_map, eps=finite_diff_eps)

    # The FD Hessian already carries both prior and likelihood curvature.
    # The analytic prior precision is added so FD noise can never drive
    # the precision matrix indefinite.
    prior_precision = np.diag(1.0 / sigma_prior ** 2)
    total_precision = 0.5 * (hessian + hessian.T) + prior_precision
    cov = _invert_precision(total_precision, sigma_prior)

    return PosteriorResult(
        method="laplace",
        mean=dict(zip(param_names, theta_map, strict=True)),
        cov=cov,
        parameter_order=tuple(param_names),
    )


# ---------------------------------------------------------------------------
# Pre-computed likelihood context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _LikelihoodContext:
    """Everything the projected KKT-residual likelihood needs per call.

    Built once before the Hessian-evaluation loop so the closure can be
    cheap. Sizes:

    * ``prices_per_gen`` — ``(T, n_tracked_gens)``: observed price at
      each tracked generator's bus, per timestep.
    * ``gen_param_indices`` — indices into the parameter vector ``θ``
      that correspond to the tracked generators (others — line costs,
      ntc, etc. — do not enter this likelihood).
    * ``active_matrices`` — per timestep, ``None`` (uncongested) or an
      ``(n_tracked_gens, |A_t|)`` matrix whose columns are PTDF rows
      restricted to the congested lines at that timestep.
    * ``obs_std`` — observation noise σ.
    """

    prices_per_gen: np.ndarray
    gen_param_indices: np.ndarray
    active_matrices: tuple[np.ndarray | None, ...]
    obs_std: float

    @property
    def n_timesteps(self) -> int:
        return int(self.prices_per_gen.shape[0])


def _build_likelihood_context(
    *,
    network: pypsa.Network,
    observations: pd.DataFrame | None,
    result: InverseResult,
    param_names: list[str],
    obs_std: float,
) -> _LikelihoodContext | None:
    """Pre-compute prices, gen→bus mapping, and active-set projection matrices.

    Returns ``None`` if observations are missing, the network is
    invalid, or no parameter binds to an observed bus — the caller
    then falls back to a prior-only posterior.
    """
    if observations is None or len(observations) == 0:
        return None
    try:
        net_data = read_network(network)
    except Exception as exc:
        logger.warning(
            "Laplace posterior: read_network failed (%s); falling back to "
            "prior-only posterior. Calibration succeeded on this network — "
            "investigate the network state.", exc,
        )
        return None

    bus_to_col = _resolve_price_columns(net_data, observations)
    if not bus_to_col:
        return None
    prices_all = observations[
        [f"price_{bus}" for bus in bus_to_col]
    ].to_numpy(dtype=float)

    tracked_gens, param_indices, bus_cols = _select_tracked_generators(
        param_names=param_names, gen_bus=net_data.gen_bus, bus_to_col=bus_to_col,
    )
    if not tracked_gens:
        return None

    prices_per_gen = prices_all[:, bus_cols]
    active_matrices = _build_active_matrices(
        net_data=net_data,
        result_active_set=result.active_set,
        n_timesteps=prices_per_gen.shape[0],
        tracked_gens=tracked_gens,
    )

    return _LikelihoodContext(
        prices_per_gen=prices_per_gen,
        gen_param_indices=np.array(param_indices, dtype=int),
        active_matrices=active_matrices,
        obs_std=obs_std,
    )


def _resolve_price_columns(
    net_data: InvoptNetworkData,
    observations: pd.DataFrame,
) -> dict[str, int]:
    """Bus name → column index in the order they appear in the DataFrame."""
    bus_to_col: dict[str, int] = {}
    for bus in net_data.buses:
        if f"price_{bus}" in observations.columns:
            bus_to_col[bus] = len(bus_to_col)
    return bus_to_col


def _select_tracked_generators(
    *,
    param_names: list[str],
    gen_bus: dict[str, str],
    bus_to_col: dict[str, int],
) -> tuple[list[str], list[int], list[int]]:
    """Pick the generator-cost parameters whose bus has an observed price."""
    tracked: list[str] = []
    param_idx: list[int] = []
    bus_cols: list[int] = []
    for i, name in enumerate(param_names):
        if not (name.startswith("gen:") and name.endswith(":marginal_cost")):
            continue
        gen_name = name.split(":")[1]
        bus = gen_bus.get(gen_name)
        if bus is None:
            continue
        col = bus_to_col.get(bus)
        if col is None:
            continue
        tracked.append(gen_name)
        param_idx.append(i)
        bus_cols.append(col)
    return tracked, param_idx, bus_cols


def _build_active_matrices(
    *,
    net_data: InvoptNetworkData,
    result_active_set: dict[int, dict[str, list[str]]],
    n_timesteps: int,
    tracked_gens: list[str],
) -> tuple[np.ndarray | None, ...]:
    """Build the per-timestep ``M_t`` projection matrices.

    Caches by congested-line tuple so two timesteps in the same batch
    share one ``M_t`` reference.
    """
    cache: dict[tuple[str, ...], np.ndarray] = {}
    matrices: list[np.ndarray | None] = []
    for t in range(n_timesteps):
        congested = tuple(
            sorted(result_active_set.get(t, {}).get("congested_lines", ()))
        )
        if not congested:
            matrices.append(None)
            continue
        if congested not in cache:
            cache[congested] = _project_ptdf_columns(
                net_data=net_data,
                tracked_gens=tracked_gens,
                congested_lines=congested,
            )
        matrices.append(cache[congested])
    return tuple(matrices)


def _project_ptdf_columns(
    *,
    net_data: InvoptNetworkData,
    tracked_gens: list[str],
    congested_lines: tuple[str, ...],
) -> np.ndarray:
    """``M[g, l] = PTDF[line_index[l], bus_index[bus(g)]]`` for the active set.

    Rows are tracked generators in their canonical order; columns are
    the congested lines in the canonical (sorted) order. Lines that
    do not appear in the network are silently dropped (their column is
    omitted) — this can only happen if the caller's active-set record
    is stale relative to the network.
    """
    valid_lines = [ln for ln in congested_lines if ln in net_data.line_index]
    matrix = np.zeros((len(tracked_gens), len(valid_lines)))
    for row, gen in enumerate(tracked_gens):
        bus_idx = net_data.bus_index[net_data.gen_bus[gen]]
        for col, line in enumerate(valid_lines):
            matrix[row, col] = float(
                net_data.ptdf[net_data.line_index[line], bus_idx]
            )
    return matrix


# ---------------------------------------------------------------------------
# Negative log-posterior closure (projected likelihood)
# ---------------------------------------------------------------------------


def _make_neg_log_posterior(
    *,
    theta_map: np.ndarray,
    sigma_prior: np.ndarray,
    likelihood_ctx: _LikelihoodContext | None,
) -> Callable[[np.ndarray], float]:
    """Return the negative log-posterior closure passed to the FD Hessian."""

    def neg_log_posterior(theta: np.ndarray) -> float:
        nll = 0.5 * float(np.sum(((theta - theta_map) / sigma_prior) ** 2))
        if likelihood_ctx is None:
            return nll
        return nll + _projected_likelihood(theta, likelihood_ctx)

    return neg_log_posterior


def _projected_likelihood(
    theta: np.ndarray,
    ctx: _LikelihoodContext,
) -> float:
    """Compute (1 / 2σ²_obs) · mean_t  ‖r*_t(θ)‖² where r* is the μ-projection.

    Per-timestep NNLS:

        μ*_t  =  argmin_{μ ≥ 0}  ‖ M_t · μ − (λ_obs_t − c) ‖²

    ``scipy.optimize.nnls`` returns ``(μ*, ‖residual‖)`` for the
    congested timesteps; for uncongested timesteps ``M_t`` is ``None``
    and the residual is just ``c − λ_obs_t``.
    """
    inv_var = 1.0 / (ctx.obs_std ** 2)
    c_vec = theta[ctx.gen_param_indices]
    n_t = ctx.n_timesteps

    sse = 0.0
    for t in range(n_t):
        target = ctx.prices_per_gen[t] - c_vec
        matrix = ctx.active_matrices[t]
        if matrix is None or matrix.shape[1] == 0:
            sse += float(np.dot(target, target))
        else:
            _, residual_norm = nnls(matrix, target)
            sse += float(residual_norm) ** 2

    return 0.5 * inv_var * sse / max(n_t, 1)


# ---------------------------------------------------------------------------
# Backwards-compatible helpers (still imported by bayes.mcmc)
# ---------------------------------------------------------------------------


def _resolve_prior_std(
    prior_std: float | dict[str, float],
    param_names: list[str],
    n_params: int,
) -> np.ndarray:
    """Return the prior-std vector aligned with ``param_names``."""
    if isinstance(prior_std, dict):
        return np.array([prior_std.get(k, 1.0) for k in param_names])
    return np.full(n_params, float(prior_std))


def _prepare_likelihood_inputs(
    *,
    network: pypsa.Network,
    observations: pd.DataFrame | None,
    param_names: list[str],
) -> tuple[InvoptNetworkData | None, np.ndarray | None, dict[str, int]]:
    """MCMC-only helper returning ``(net_data, prices, gen_to_price_col)``.

    The MCMC path uses the *uncongested* closed-form Gaussian
    likelihood — exposing the NNLS μ-projection (used by the Laplace
    path) to PyMC would require a custom differentiable Op. Limitation
    is documented in :mod:`pypsa_invopt.bayes.mcmc`.
    """
    if observations is None or len(observations) == 0:
        return None, None, {}
    try:
        net_data = read_network(network)
    except Exception as exc:
        logger.warning("MCMC posterior: read_network failed (%s); "
                       "falling back to prior-only.", exc)
        return None, None, {}

    bus_to_col = _resolve_price_columns(net_data, observations)
    if not bus_to_col:
        return net_data, None, {}
    prices = observations[[f"price_{bus}" for bus in bus_to_col]].to_numpy(
        dtype=float,
    )

    _, param_idx, bus_cols = _select_tracked_generators(
        param_names=param_names, gen_bus=net_data.gen_bus, bus_to_col=bus_to_col,
    )
    gen_to_price_col = {
        param_names[i]: bus_cols[k] for k, i in enumerate(param_idx)
    }
    return net_data, prices, gen_to_price_col


# ---------------------------------------------------------------------------
# Linear-algebra helpers
# ---------------------------------------------------------------------------


def _invert_precision(precision: np.ndarray, sigma_prior: np.ndarray) -> np.ndarray:
    """Invert a precision matrix via floored eigendecomposition.

    Falls back to the prior covariance if the inversion fails.
    """
    try:
        eigvals, eigvecs = np.linalg.eigh(precision)
    except np.linalg.LinAlgError:
        return np.diag(sigma_prior ** 2)

    # Floor the smallest eigenvalue so unidentifiable directions get a
    # finite posterior σ. The cap is 10⁶× the widest prior σ — any
    # parameter the data leaves at the prior reads back as σ_prior, not
    # ∞. The +1 prevents division by zero when sigma_prior collapses.
    prec_floor = 1.0 / max(float(np.max(sigma_prior)) * 1e6, 1.0) ** 2
    eigvals = np.clip(eigvals, prec_floor, None)
    cov = (eigvecs * (1.0 / eigvals)) @ eigvecs.T
    return 0.5 * (cov + cov.T)


def _compute_hessian(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    eps: float = 1e-3,
) -> np.ndarray:
    """Central-difference Hessian of ``f`` at ``x0``."""
    n = len(x0)
    hessian = np.zeros((n, n))
    f0 = f(x0)

    for i in range(n):
        hessian[i, i] = _hessian_diagonal_entry(f, x0, i, eps, f0)

    for i in range(n):
        for j in range(i + 1, n):
            mixed = _hessian_mixed_entry(f, x0, i, j, eps)
            hessian[i, j] = hessian[j, i] = mixed

    return hessian


def _hessian_diagonal_entry(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    i: int,
    eps: float,
    f0: float,
) -> float:
    """``∂²f / ∂x_i²`` via the three-point central-difference stencil."""
    x_plus = x0.copy()
    x_plus[i] += eps
    x_minus = x0.copy()
    x_minus[i] -= eps
    return (f(x_plus) - 2.0 * f0 + f(x_minus)) / (eps ** 2)


def _hessian_mixed_entry(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    i: int,
    j: int,
    eps: float,
) -> float:
    """``∂²f / ∂x_i ∂x_j`` via the four-point central-difference stencil."""
    xpp = x0.copy()
    xpp[i] += eps
    xpp[j] += eps
    xpm = x0.copy()
    xpm[i] += eps
    xpm[j] -= eps
    xmp = x0.copy()
    xmp[i] -= eps
    xmp[j] += eps
    xmm = x0.copy()
    xmm[i] -= eps
    xmm[j] -= eps
    return (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4.0 * eps ** 2)


__all__ = ["laplace_posterior"]
