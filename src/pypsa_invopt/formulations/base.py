"""Abstract base class and shared helpers for inverse-OPF formulations.

Three things live here:

* :class:`InverseFormulation` — the abstract interface every concrete
  formulation implements.
* :class:`BuildSpec` — pre-resolved active-set and topology lookups
  the QP-style formulations share when building their sparse
  matrices. Avoids re-deriving the same arrays in every file.
* :func:`ptdf_projection` — small numpy helper that pulls a few
  PTDF rows out of the full matrix.

Pure numpy + the project's own dataclasses — the formulations build
sparse matrices directly and hand them off to :mod:`highspy`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from pypsa_invopt.network import InvoptNetworkData
    from pypsa_invopt.solvers import SolverConfig
    from pypsa_invopt.utils.active_set import ActiveSetBatch


@runtime_checkable
class InverseFormulation(Protocol):
    """Structural protocol every inverse-OPF formulation satisfies.

    A formulation is anything with a ``build_model`` / ``solve`` /
    ``residuals`` triple matching the signatures below. ``build_model``
    may return any object — one of the dataclass-wrapped QPs the native
    back ends use, or a ``linopy.Model`` — that ``solve`` knows how to
    consume.

    Using :class:`typing.Protocol` keeps the contract honest (calibrate
    duck-types these anyway) without an inheritance dependency on the
    three concrete classes.
    """

    name: ClassVar[str]

    def build_model(
        self,
        network_data: InvoptNetworkData,
        batch: ActiveSetBatch,
        observations: dict[str, np.ndarray],
        **kwargs: Any,
    ) -> Any:
        """Build a model object for one active-set batch."""
        ...

    def solve(
        self,
        model: Any,
        solver_config: SolverConfig | None = None,
    ) -> dict[str, Any]:
        """Solve the model and return ``{theta, residuals, status, objective}``."""
        ...

    def residuals(
        self,
        theta: dict[str, float],
        network_data: InvoptNetworkData,
        observations: dict[str, np.ndarray],
        batch: ActiveSetBatch,
    ) -> np.ndarray:
        """Per-timestep KKT residual norms for a given parameter vector."""
        ...


@dataclass(frozen=True)
class BuildSpec:
    """Pre-resolved active-set and topology lookups for one batch.

    Both nodal formulations (``noiseless`` and ``noisy``) translate
    the same metadata into the same handful of arrays
    before building its sparse matrices. Centralising the translation
    here keeps each formulation file focused on the *math* that
    distinguishes it.

    Attributes:
        network_data: Raw :class:`InvoptNetworkData`.
        generators: Canonical generator order.
        buses: Canonical bus order — matches ``ptdf`` column order.
        active_lines: Lines flagged congested by the batch, sorted.
        maxed_gens: Generators at their upper bound, sorted.
        min_gens: Generators at their lower bound, sorted.
        gen_bus_idx: ``int`` array of length ``n_gens``; each entry is
            the bus index of the corresponding generator. Used as a
            vectorised PTDF row picker.
        maxed_gen_idx: Generator → its row in the ν block.
        min_gen_idx: Generator → its row in the ξ block.
        n_gens, n_buses, n_t: Cached sizes.
    """

    network_data: InvoptNetworkData
    generators: list[str]
    buses: list[str]
    active_lines: list[str]
    maxed_gens: list[str]
    min_gens: list[str]
    gen_bus_idx: np.ndarray
    maxed_gen_idx: dict[str, int]
    min_gen_idx: dict[str, int]
    n_gens: int
    n_buses: int
    n_t: int

    @classmethod
    def from_inputs(
        cls,
        *,
        network_data: InvoptNetworkData,
        batch: ActiveSetBatch,
    ) -> BuildSpec:
        generators = list(network_data.generators)
        buses = list(network_data.buses)
        active_lines = sorted(batch.pattern.congested_lines)
        maxed = sorted(batch.pattern.maxed_generators)
        min_gens = sorted(batch.pattern.min_bound_generators)

        gen_bus_idx = np.array(
            [network_data.bus_index[network_data.gen_bus[g]] for g in generators],
            dtype=int,
        )
        return cls(
            network_data=network_data,
            generators=generators,
            buses=buses,
            active_lines=active_lines,
            maxed_gens=maxed,
            min_gens=min_gens,
            gen_bus_idx=gen_bus_idx,
            maxed_gen_idx={g: i for i, g in enumerate(maxed)},
            min_gen_idx={g: i for i, g in enumerate(min_gens)},
            n_gens=len(generators),
            n_buses=len(buses),
            n_t=len(batch.timestep_indices),
        )


def ptdf_projection(
    *,
    network_data: InvoptNetworkData,
    active_lines: list[str],
    gen_bus_idx: np.ndarray,
) -> np.ndarray:
    """``(|A_lines|, n_gens)`` slice of PTDF — rows = lines, cols = gen buses.

    Empty when the active set has no congested lines. Used as the
    matrix that multiplies ``μ`` in the KKT-stationarity row.
    """
    if not active_lines:
        return np.zeros((0, len(gen_bus_idx)))
    line_rows = np.array(
        [network_data.line_index[ln] for ln in active_lines], dtype=int,
    )
    return network_data.ptdf[np.ix_(line_rows, gen_bus_idx)]


def maxed_or_min_gen_indices(spec: BuildSpec, gen_names: list[str]) -> np.ndarray:
    """Original generator indices of an ordered subset list.

    Used by every QP builder to map ``spec.maxed_gens`` /
    ``spec.min_gens`` (which carry generator *names*) onto the
    canonical generator-index axis the KKT row block enumerates.
    """
    if not gen_names:
        return np.array([], dtype=int)
    gen_name_to_idx = {g: i for i, g in enumerate(spec.generators)}
    return np.array([gen_name_to_idx[g] for g in gen_names], dtype=int)


def mu_kkt_block(
    *,
    n_t: int,
    ptdf_active: np.ndarray,
    mu_offset: int,
    t_range: np.ndarray,
    sign: float = -1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The ``sign · PTDF · μ`` contribution to a KKT row block.

    ``sign`` is ``-1`` for the noisy convention (``r = c − λ +
    PTDF·μ + …``) and ``+1`` for the noiseless convention
    (``c − λ + PTDF·μ + … = 0``).

    Structurally-zero PTDF entries are skipped — power-network
    sparsity means this drops almost all of the matrix on large grids.
    """
    if ptdf_active.size == 0:
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([], dtype=float),
        )
    line_idx, gen_idx = np.where(np.abs(ptdf_active) > 1e-12)
    coefs = ptdf_active[line_idx, gen_idx]
    rows = np.repeat(gen_idx, n_t) * n_t + np.tile(t_range, line_idx.size)
    cols = mu_offset + np.repeat(line_idx, n_t) * n_t + np.tile(t_range, line_idx.size)
    data = sign * np.repeat(coefs, n_t)
    return rows, cols, data


def bound_dual_kkt_block(
    *,
    gen_indices: np.ndarray,
    n_t: int,
    dual_offset: int,
    sign: float,
    t_range: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The ν or ξ contribution to a KKT row block.

    Each maxed / min-bound generator contributes ``sign · 1`` on its
    own row at every timestep — vectorised over ``(k, t)`` pairs.
    """
    n = gen_indices.size
    if n == 0:
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([], dtype=float),
        )
    rows = np.repeat(gen_indices, n_t) * n_t + np.tile(t_range, n)
    cols = dual_offset + np.repeat(np.arange(n, dtype=int), n_t) * n_t + np.tile(t_range, n)
    data = np.full(n * n_t, sign)
    return rows, cols, data


def cost_residual_norms(
    theta: dict[str, float],
    network_data: InvoptNetworkData,
    prices: np.ndarray,
) -> np.ndarray:
    """Per-timestep ``‖c − λ_obs‖`` residual norm.

    Shared by every formulation as its diagnostic ``residuals(...)``
    implementation.
    """
    n_t = prices.shape[0]
    n_gens = len(network_data.generators)
    if n_t == 0 or n_gens == 0:
        return np.zeros(n_t)

    cost_vec = np.array(
        [theta.get(f"gen:{g}:marginal_cost", 0.0) for g in network_data.generators],
        dtype=float,
    )
    bus_idx = np.array(
        [network_data.bus_index[network_data.gen_bus[g]] for g in network_data.generators],
        dtype=int,
    )
    # prices is (T, n_buses); pick the bus column for each generator → (T, n_gens).
    prices_at_gen = prices[:, bus_idx]
    residual = prices_at_gen - cost_vec  # broadcasts cost_vec over the timestep axis
    sse_per_t = np.einsum("tg,tg->t", residual, residual)
    return np.sqrt(sse_per_t / n_gens)


def solve_qp_or_raise(
    *,
    model: Any,
    solver_config: SolverConfig | None,
    formulation_name: str,
    convergence_hint: str = "",
):
    """Boilerplate every formulation's ``solve()`` shares.

    Reads ``verbose`` off ``solver_config``, dispatches to
    :func:`pypsa_invopt.solvers.qp.solve_qp` with the standard
    ``Q/q/A_eq/b_eq/lb/ub`` payload, and raises
    :class:`InvoptConvergenceError` on non-optimal status. Returns the
    raw ``solve_qp`` result.

    ``convergence_hint`` is an optional extra line appended to the
    error message when the solver doesn't converge — e.g. the
    noiseless formulation suggests trying ``noisy``.
    """
    from pypsa_invopt.exceptions import InvoptConvergenceError
    from pypsa_invopt.solvers.qp import solve_qp

    verbose = bool(solver_config.verbose) if solver_config else False
    qp = solve_qp(
        Q=model.Q, q=model.q,
        A_eq=model.A_eq, b_eq=model.b_eq,
        lb=model.lb, ub=model.ub,
        verbose=verbose,
    )
    if not qp.is_optimal:
        msg = f"{formulation_name} QP did not converge ({qp.status})."
        if convergence_hint:
            msg = f"{msg} {convergence_hint}"
        raise InvoptConvergenceError(msg)
    return qp


__all__ = [
    "BuildSpec",
    "InverseFormulation",
    "bound_dual_kkt_block",
    "cost_residual_norms",
    "maxed_or_min_gen_indices",
    "mu_kkt_block",
    "ptdf_projection",
    "solve_qp_or_raise",
]
