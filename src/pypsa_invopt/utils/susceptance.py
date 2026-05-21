"""Line-parameter recovery from observed dispatch and flows.

Two post-processing steps that the main cost-recovery LP does not
address:

1. **Flow-limit recovery** (closed form). When a line was congested at
   some timesteps, its thermal limit ``s_nom`` cannot be smaller than
   the largest observed ``|f*|``. :func:`recover_flow_limits` returns
   that bound (with a small safety margin) for every congested line.

2. **Susceptance recovery** (bounded nonlinear least squares). Re-fits
   each line's reactance ``x_l = 1 / b_l`` so that the DC-power-flow
   identity ``f_l = b_l · (θ_{bus0} − θ_{bus1})`` and the nodal balance
   ``Σ_l incident  b_l · sign · Δθ_l = injection`` match the observations
   over ``T`` snapshots. The problem is bilinear in ``(b, θ)``, so we
   solve it with :func:`scipy.optimize.least_squares` (trust-region
   reflective method, supports bounded variables). A Tikhonov anchor
   to the engineering prior is mandatory: DC-PF is invariant under
   ``(b, θ) → (α b, θ / α)`` so without the anchor only the ratios of
   line susceptances are identifiable.

The slack-bus angle is **eliminated from the variable vector** (set
implicitly to zero); the nodal-balance equations and the slack-bus
``θ = 0`` condition both become residuals in the least-squares stack,
weighted heavily enough that the optimum respects them.

Both steps are opt-in via the ``recover_line_params`` argument of
:func:`pypsa_invopt.calibrate`. Their outputs go straight into the
``theta_hat`` dict and are consumed by
:func:`pypsa_invopt.network.apply_result`.

Reference for the DC-PF parametrisation: Brown et al. (2018), PyPSA
paper; Conejo et al. (2006) ch. 4 for the KKT structure. Algorithm:
Branch T.F., Coleman T.F., Li Y. (1999). A Subspace, Interior, and
Conjugate Gradient Method for Large-Scale Bound-Constrained
Minimization Problems. *SIAM J. Sci. Comput.* 21(1).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import least_squares

from pypsa_invopt._constants import (
    BALANCE_PENALTY,
)

if TYPE_CHECKING:
    from pypsa_invopt.network import InvoptNetworkData
    from pypsa_invopt.solvers import SolverConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flow-limit recovery (unchanged, closed form)
# ---------------------------------------------------------------------------


def recover_flow_limits(
    *,
    network_data: InvoptNetworkData,
    flows: np.ndarray,
    congested_per_line: dict[str, list[int]],
    safety_margin: float = 1.02,
) -> dict[str, float]:
    """Recover line thermal limits from congested observations.

    Args:
        network_data: Network topology / current parameters.
        flows: Observed flows, shape ``(T, n_lines)``.
        congested_per_line: Line name → list of timestep indices at
            which the active-set detector flagged it as congested.
        safety_margin: Multiplicative cushion above the largest
            observed magnitude (``1.02`` leaves ~2% headroom).

    Returns:
        ``{line_name: s_nom_hat}`` — only for lines with at least one
        congested observation. Lines absent from the result should
        retain their existing ``s_nom``.
    """
    recovered: dict[str, float] = {}
    for ln, indices in congested_per_line.items():
        if not indices:
            continue
        l_idx = network_data.line_index.get(ln)
        if l_idx is None:
            continue
        max_obs = float(np.max(np.abs(flows[indices, l_idx])))
        if max_obs > 0:
            recovered[ln] = max_obs * safety_margin
    return recovered


# ---------------------------------------------------------------------------
# Susceptance recovery via scipy.optimize.least_squares
# ---------------------------------------------------------------------------


def recover_line_susceptances(
    *,
    network_data: InvoptNetworkData,
    flows: np.ndarray,
    dispatch: np.ndarray | None,
    loads_per_bus: np.ndarray | None,
    solver_config: SolverConfig | None = None,
    bounds_factor: tuple[float, float] = (0.25, 4.0),
    sample_size: int | None = 50,
    prior_weight: float = 0.1,
) -> dict[str, float]:
    """Recover line susceptances ``b_l = 1 / x_l`` by bounded NLS.

    Solves

        min over (b, θ)  Σ_t Σ_l (f*_l[t] − b_l · Δθ_l[t])²
                          + balance_penalty · Σ_{b,t} (injection − Σ incident · b · Δθ)²
                          + prior_weight   · Σ_l ((b_l − b_prior_l) / b_prior_l)²
        s.t.  x_l · bounds_factor[0]  ≤  1 / b_l  ≤  x_l · bounds_factor[1]

    The slack bus's angle is eliminated by construction (always zero).
    The Tikhonov anchor breaks the gauge degeneracy described in the
    module docstring.

    Args:
        network_data: Topology + prior parameter values.
        flows: Observed flows, shape ``(T, n_lines)``.
        dispatch: Observed generator dispatch, shape ``(T, n_gens)``.
            Passing ``None`` drops the nodal-balance residuals.
        loads_per_bus: Observed (or static) per-bus demand, shape
            ``(T, n_buses)``. ``None`` drops nodal balance.
        solver_config: Reserved for future per-call solver tuning.
            Currently unused — scipy's defaults work for this shape.
        bounds_factor: ``(low, high)`` multipliers on the prior
            reactance defining the search box.
        sample_size: Sub-sample at most this many timesteps. ``None``
            uses all.
        prior_weight: Weight on the gauge-breaking Tikhonov anchor.

    Returns:
        ``{line_name: x_hat}``. Lines hitting a bound — interpreted as
        the data being uninformative for that line — are silently
        dropped.
    """
    del solver_config  # currently unused — kept for API stability
    if flows.shape[0] == 0:
        return {}

    buses = list(network_data.buses)
    lines = list(network_data.lines)
    if len(lines) == 0 or len(buses) < 2:
        return {}

    sample_idx = _select_sample(flows.shape[0], sample_size)
    inject = _compute_injection_table(
        network_data=network_data,
        dispatch=dispatch,
        loads_per_bus=loads_per_bus,
    )
    if inject is not None:
        inject = inject[sample_idx]

    spec = _SusceptanceSpec.from_inputs(
        network_data=network_data,
        buses=buses,
        lines=lines,
        flows=flows[sample_idx],
        inject=inject,
        bounds_factor=bounds_factor,
        prior_weight=prior_weight,
    )

    return _solve_susceptance_nls(spec)


# ---------------------------------------------------------------------------
# Build-time spec for the NLS problem
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SusceptanceSpec:
    """Pre-resolved inputs for the bounded NLS problem.

    Attributes:
        lines: Canonical line order.
        buses: Canonical bus order. ``buses[0]`` is the slack bus.
        flows: ``(T, n_lines)`` observed flows after sub-sampling.
        inject: ``(T, n_buses)`` net injection or ``None`` if the
            caller dropped balance information.
        line_bus0_idx, line_bus1_idx: Vectorised endpoint indices.
        non_slack_buses: Indices of buses that carry a free θ variable.
        bus_to_var: Bus index → variable-vector position (slack maps
            to -1 so the residual builder substitutes 0).
        x_prior: ``(n_lines,)`` prior reactance per line.
        b_prior: ``(n_lines,)`` prior susceptance per line.
        b_lower, b_upper: Bounds on ``b`` derived from ``bounds_factor``.
        prior_weight: Tikhonov coefficient.
        n_t, n_lines, n_buses, n_free_buses: Cached sizes.
    """

    lines: list[str]
    buses: list[str]
    flows: np.ndarray
    inject: np.ndarray | None
    line_bus0_idx: np.ndarray
    line_bus1_idx: np.ndarray
    non_slack_buses: np.ndarray
    bus_to_var: np.ndarray
    x_prior: np.ndarray
    b_prior: np.ndarray
    b_lower: np.ndarray
    b_upper: np.ndarray
    prior_weight: float
    n_t: int
    n_lines: int
    n_buses: int
    n_free_buses: int

    @classmethod
    def from_inputs(
        cls,
        *,
        network_data: InvoptNetworkData,
        buses: list[str],
        lines: list[str],
        flows: np.ndarray,
        inject: np.ndarray | None,
        bounds_factor: tuple[float, float],
        prior_weight: float,
    ) -> _SusceptanceSpec:
        n_t = flows.shape[0]
        n_lines = len(lines)
        n_buses = len(buses)
        bus_idx = network_data.bus_index

        line_bus0_idx = np.array(
            [bus_idx[network_data.line_bus0[ln]] for ln in lines], dtype=int,
        )
        line_bus1_idx = np.array(
            [bus_idx[network_data.line_bus1[ln]] for ln in lines], dtype=int,
        )
        non_slack_buses = np.array(
            [i for i in range(n_buses) if i != 0], dtype=int,
        )
        bus_to_var = np.full(n_buses, -1, dtype=int)
        bus_to_var[non_slack_buses] = np.arange(len(non_slack_buses))

        x_prior = np.array(
            [float(network_data.line_x[ln]) for ln in lines], dtype=float,
        )
        b_prior = 1.0 / x_prior
        b_lower = 1.0 / (bounds_factor[1] * x_prior)
        b_upper = 1.0 / (bounds_factor[0] * x_prior)

        # Reorder flows so it indexes by network-canonical line order.
        flows_reordered = flows[:, [network_data.line_index[ln] for ln in lines]]

        return cls(
            lines=lines,
            buses=buses,
            flows=flows_reordered,
            inject=inject,
            line_bus0_idx=line_bus0_idx,
            line_bus1_idx=line_bus1_idx,
            non_slack_buses=non_slack_buses,
            bus_to_var=bus_to_var,
            x_prior=x_prior,
            b_prior=b_prior,
            b_lower=b_lower,
            b_upper=b_upper,
            prior_weight=prior_weight,
            n_t=n_t,
            n_lines=n_lines,
            n_buses=n_buses,
            n_free_buses=len(non_slack_buses),
        )


# ---------------------------------------------------------------------------
# Variable packing
# ---------------------------------------------------------------------------


def _initial_guess(spec: _SusceptanceSpec) -> np.ndarray:
    """Initial point: ``b = b_prior``, ``θ_free = 0``."""
    b0 = spec.b_prior.copy()
    theta0 = np.zeros(spec.n_free_buses * spec.n_t)
    return np.concatenate([b0, theta0])


def _bounds(spec: _SusceptanceSpec) -> tuple[np.ndarray, np.ndarray]:
    """Bound vectors aligned with the packed variable layout."""
    lb = np.concatenate([
        spec.b_lower,
        np.full(spec.n_free_buses * spec.n_t, -np.inf),
    ])
    ub = np.concatenate([
        spec.b_upper,
        np.full(spec.n_free_buses * spec.n_t, np.inf),
    ])
    return lb, ub


# ---------------------------------------------------------------------------
# Residual function
# ---------------------------------------------------------------------------


def _angle_diffs(
    spec: _SusceptanceSpec, theta_free: np.ndarray,
) -> np.ndarray:
    """Per-(line, timestep) angle difference ``θ_{bus0} − θ_{bus1}``.

    The slack bus's angle is the constant zero; everywhere else we
    pick from the free-variable slot indexed by ``bus_to_var``.
    """
    theta = np.zeros((spec.n_buses, spec.n_t))
    theta_grid = theta_free.reshape(spec.n_free_buses, spec.n_t)
    theta[spec.non_slack_buses] = theta_grid
    return theta[spec.line_bus0_idx] - theta[spec.line_bus1_idx]


def _residual(x: np.ndarray, spec: _SusceptanceSpec) -> np.ndarray:
    """Stack flow residuals, balance residuals, and the prior anchor.

    Shape: ``n_lines·T  +  (n_free_buses·T if balance is engaged else 0)
                       +  n_lines``.
    """
    b = x[: spec.n_lines]
    theta_free = x[spec.n_lines :]

    angle_diff = _angle_diffs(spec, theta_free)             # (n_lines, n_t)
    predicted_flow = b[:, None] * angle_diff                 # (n_lines, n_t)
    flow_residual = (spec.flows.T - predicted_flow).ravel()  # row-major

    blocks: list[np.ndarray] = [flow_residual]

    if spec.inject is not None:
        blocks.append(_balance_residual(spec, b, angle_diff))

    anchor = np.sqrt(spec.prior_weight / max(spec.n_lines, 1)) * (
        (b - spec.b_prior) / spec.b_prior
    )
    blocks.append(anchor)

    return np.concatenate(blocks)


def _balance_residual(
    spec: _SusceptanceSpec,
    b: np.ndarray,
    angle_diff: np.ndarray,
) -> np.ndarray:
    """Penalty-weighted nodal-balance residual for the non-slack buses.

    For each non-slack bus ``n`` and timestep ``t``:

        Σ_{l: bus0(l) = n}  b_l · Δθ_l[t]
      − Σ_{l: bus1(l) = n}  b_l · Δθ_l[t]
      − injection[n, t]

    The penalty weight ``√BALANCE_PENALTY`` puts these residuals on
    par with the flow residuals (scipy's least-squares routine sees a
    flat residual vector; the relative weight is what shifts the
    optimum).
    """
    assert spec.inject is not None  # checked by the caller
    flow_per_line = b[:, None] * angle_diff  # (n_lines, n_t)
    balance = np.zeros((spec.n_buses, spec.n_t))
    np.add.at(balance, spec.line_bus0_idx, flow_per_line)
    np.subtract.at(balance, spec.line_bus1_idx, flow_per_line)
    residual = balance[spec.non_slack_buses] - spec.inject[:, spec.non_slack_buses].T
    return np.sqrt(BALANCE_PENALTY) * residual.ravel()


# ---------------------------------------------------------------------------
# Solve + result extraction
# ---------------------------------------------------------------------------


def _solve_susceptance_nls(spec: _SusceptanceSpec) -> dict[str, float]:
    """Run scipy.optimize.least_squares and post-process the recovered ``b``."""
    lb, ub = _bounds(spec)
    try:
        result = least_squares(
            fun=_residual,
            x0=_initial_guess(spec),
            bounds=(lb, ub),
            args=(spec,),
            method="trf",
            xtol=1e-10,
            ftol=1e-10,
        )
    except (ValueError, RuntimeError) as exc:
        logger.warning("Susceptance NLS failed (%s); keeping prior reactances.", exc)
        return {}

    if not result.success:
        logger.info(
            "Susceptance NLS terminated without success (status=%d, msg=%s); "
            "keeping prior reactances.",
            result.status, result.message,
        )
        return {}

    b_recovered = result.x[: spec.n_lines]
    return _accept_recoveries(spec, b_recovered)


def _accept_recoveries(
    spec: _SusceptanceSpec, b_recovered: np.ndarray,
) -> dict[str, float]:
    """Keep recovered reactances strictly inside the bounds box.

    A line whose recovered ``b`` saturated a bound is interpreted as
    *the data not being informative for that line* and is dropped.
    """
    recovered: dict[str, float] = {}
    for l_idx, line in enumerate(spec.lines):
        b_value = float(b_recovered[l_idx])
        if b_value <= 0:
            continue
        x_hat = 1.0 / b_value
        x_lo = 1.0 / spec.b_upper[l_idx]
        x_hi = 1.0 / spec.b_lower[l_idx]
        if x_lo + 1e-9 < x_hat < x_hi - 1e-9:
            recovered[line] = x_hat
    return recovered


# ---------------------------------------------------------------------------
# Input helpers (sub-sampling, injection table)
# ---------------------------------------------------------------------------


def _select_sample(n_obs: int, sample_size: int | None) -> np.ndarray:
    """Return the timestep indices to feed into the NLS."""
    if sample_size is None or n_obs <= sample_size:
        return np.arange(n_obs)
    rng = np.random.default_rng(0)
    return np.sort(rng.choice(n_obs, sample_size, replace=False))


def _compute_injection_table(
    *,
    network_data: InvoptNetworkData,
    dispatch: np.ndarray | None,
    loads_per_bus: np.ndarray | None,
) -> np.ndarray | None:
    """Aggregate dispatch − loads to ``(T, n_buses)`` net injection.

    Returns ``None`` when either input is missing — the caller then
    runs the NLS without the nodal-balance residual block.
    """
    if dispatch is None or loads_per_bus is None:
        return None

    bus_idx = network_data.bus_index
    n_buses = len(network_data.buses)
    gen_to_bus_col = np.array(
        [bus_idx[network_data.gen_bus[g]] for g in network_data.generators],
        dtype=int,
    )
    inject = np.zeros((dispatch.shape[0], n_buses))
    for g_idx, b_col in enumerate(gen_to_bus_col):
        inject[:, b_col] += dispatch[:, g_idx]
    return inject - loads_per_bus


__all__ = ["recover_flow_limits", "recover_line_susceptances"]
