"""Noiseless inverse OPF (KKT equality LP).

Sister of :mod:`pypsa_invopt.formulations.noisy`: the KKT
stationarity identity is enforced as a *hard equality* instead of as
a penalty, so the model carries no ``r`` slack and the objective
collapses to the squared price-fit term.

Variable layout (HiGHS column order):

* ``c``   — generator costs                 (n_gens)
* ``λ``   — bus LMPs                        (n_buses · T)
* ``μ``   — line shadow prices              (|A_lines| · T)
* ``ν``   — generator upper-bound duals     (|A_max|  · T)
* ``ξ``   — generator lower-bound duals     (|A_min|  · T)

KKT row (one per ``(g, t)``):

    c[g]  −  λ_{bus(g), t}
    +  Σ_{l ∈ A_lines}  PTDF_{l, bus(g)} · μ_{l, t}
    +  ν_{g, t} · 1{g ∈ A_max}
    −  ξ_{g, t} · 1{g ∈ A_min}
    =  0

Reference: Ruiz C., Conejo A.J. (2009). Pool Strategy of a Producer
with Endogenous Formation of LMPs. *IEEE TPWRS* 24(4), eq. [5].
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

from pypsa_invopt.formulations.base import (
    BuildSpec,
    InverseFormulation,
    bound_dual_kkt_block,
    cost_residual_norms,
    maxed_or_min_gen_indices,
    mu_kkt_block,
    ptdf_projection,
    solve_qp_or_raise,
)
from pypsa_invopt.network import InvoptNetworkData
from pypsa_invopt.solvers import SolverConfig
from pypsa_invopt.utils.active_set import ActiveSetBatch


class NoiselessFormulation(InverseFormulation):
    """Noiseless inverse OPF on a direct sparse QP."""

    name = "noiseless"

    def build_model(
        self,
        network_data: InvoptNetworkData,
        batch: ActiveSetBatch,
        observations: dict[str, np.ndarray],
        **kwargs: Any,
    ) -> _NoiselessQP:
        return _NoiselessQP.assemble(
            network_data=network_data,
            batch=batch,
            prices=observations["prices"],
        )

    def solve(
        self,
        model: _NoiselessQP,
        solver_config: SolverConfig | None = None,
    ) -> dict[str, Any]:
        qp = solve_qp_or_raise(
            model=model,
            solver_config=solver_config,
            formulation_name="noiseless",
            convergence_hint=(
                "The active set may be inconsistent with the observed "
                "prices — try the 'noisy' formulation."
            ),
        )
        theta = model.layout.extract_costs(qp.x, model.generators)
        residuals = _price_residual_norms(
            x=qp.x,
            layout=model.layout,
            prices=model.prices,
        )
        return {
            "theta": theta,
            "residuals": residuals,
            "status": "optimal",
            "objective": qp.objective,
        }

    def residuals(
        self,
        theta: dict[str, float],
        network_data: InvoptNetworkData,
        observations: dict[str, np.ndarray],
        batch: ActiveSetBatch,
    ) -> np.ndarray:
        return cost_residual_norms(theta, network_data, observations["prices"])


# ---------------------------------------------------------------------------
# Variable layout (no ``r`` block compared to the noisy formulation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _NoiselessLayout:
    n_gens: int
    n_buses: int
    n_active_lines: int
    n_maxed: int
    n_min: int
    n_t: int

    c: int
    lam: int
    mu: int
    nu: int
    xi: int
    total: int

    @classmethod
    def for_shape(
        cls,
        *,
        n_gens: int,
        n_buses: int,
        n_active_lines: int,
        n_maxed: int,
        n_min: int,
        n_t: int,
    ) -> _NoiselessLayout:
        c = 0
        lam = c + n_gens
        mu = lam + n_buses * n_t
        nu = mu + n_active_lines * n_t
        xi = nu + n_maxed * n_t
        total = xi + n_min * n_t
        return cls(
            n_gens=n_gens, n_buses=n_buses,
            n_active_lines=n_active_lines, n_maxed=n_maxed, n_min=n_min,
            n_t=n_t, c=c, lam=lam, mu=mu, nu=nu, xi=xi, total=total,
        )

    def extract_costs(
        self,
        z: np.ndarray,
        generators: list[str],
    ) -> dict[str, float]:
        return {
            f"gen:{gen}:marginal_cost": float(z[self.c + i])
            for i, gen in enumerate(generators)
        }


# ---------------------------------------------------------------------------
# QP assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _NoiselessQP:
    Q: sp.csc_matrix
    q: np.ndarray
    A_eq: sp.csc_matrix
    b_eq: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    layout: _NoiselessLayout
    generators: list[str]
    prices: np.ndarray

    @classmethod
    def assemble(
        cls,
        *,
        network_data: InvoptNetworkData,
        batch: ActiveSetBatch,
        prices: np.ndarray,
    ) -> _NoiselessQP:
        spec = BuildSpec.from_inputs(network_data=network_data, batch=batch)
        layout = _NoiselessLayout.for_shape(
            n_gens=spec.n_gens,
            n_buses=spec.n_buses,
            n_active_lines=len(spec.active_lines),
            n_maxed=len(spec.maxed_gens),
            n_min=len(spec.min_gens),
            n_t=spec.n_t,
        )

        A_eq, b_eq = _build_kkt_equality(spec, layout)
        Q, q = _build_objective(spec, layout, prices=prices)
        lb, ub = _build_bounds(layout)

        return cls(
            Q=Q, q=q, A_eq=A_eq, b_eq=b_eq, lb=lb, ub=ub,
            layout=layout, generators=spec.generators, prices=prices,
        )


def _build_kkt_equality(
    spec: BuildSpec, layout: _NoiselessLayout,
) -> tuple[sp.csc_matrix, np.ndarray]:
    """Hard KKT equality without a residual slack — direct ``Az = 0`` form.

    For each ``(g, t)`` row the equation is

        +c[g]  −  λ_{bus(g), t}
        +  Σ_l  PTDF_{l, bus(g)} · μ_{l, t}
        +  ν_{g, t} · 1{g ∈ A_max}
        −  ξ_{g, t} · 1{g ∈ A_min}
        =  0

    Identical sparsity structure to the noisy formulation but with the
    opposite sign convention (no ``r`` slack). Constructed in pure
    numpy — no Python loops over generators / timesteps / active lines.
    """
    n_t = spec.n_t
    n_gens = spec.n_gens
    n_rows = n_gens * n_t
    if n_rows == 0:
        return sp.csc_matrix((0, layout.total)), np.zeros(0)

    ptdf_active = ptdf_projection(
        network_data=spec.network_data,
        active_lines=spec.active_lines,
        gen_bus_idx=spec.gen_bus_idx,
    )
    rows_flat = np.arange(n_rows, dtype=int)
    t_range = np.arange(n_t, dtype=int)

    # +c[g] on row (g, t).
    c_cols = layout.c + np.repeat(np.arange(n_gens, dtype=int), n_t)
    c_data = np.ones(n_rows)

    # −λ[bus(g), t] on row (g, t).
    lam_cols = layout.lam + np.repeat(spec.gen_bus_idx, n_t) * n_t + np.tile(t_range, n_gens)
    lam_data = -np.ones(n_rows)

    # PTDF·μ enters with ``+`` sign here (noiseless) vs ``−`` in the
    # noisy form; ``sign=+1`` flips the helper's default.
    mu_rows, mu_cols, mu_data = mu_kkt_block(
        n_t=n_t, ptdf_active=ptdf_active, mu_offset=layout.mu,
        t_range=t_range, sign=+1.0,
    )
    nu_rows, nu_cols, nu_data = bound_dual_kkt_block(
        gen_indices=maxed_or_min_gen_indices(spec, spec.maxed_gens),
        n_t=n_t, dual_offset=layout.nu, sign=+1.0, t_range=t_range,
    )
    xi_rows, xi_cols, xi_data = bound_dual_kkt_block(
        gen_indices=maxed_or_min_gen_indices(spec, spec.min_gens),
        n_t=n_t, dual_offset=layout.xi, sign=-1.0, t_range=t_range,
    )

    A_eq = sp.csc_matrix(
        (
            np.concatenate([c_data, lam_data, mu_data, nu_data, xi_data]),
            (
                np.concatenate([rows_flat, rows_flat, mu_rows, nu_rows, xi_rows]),
                np.concatenate([c_cols, lam_cols, mu_cols, nu_cols, xi_cols]),
            ),
        ),
        shape=(n_rows, layout.total),
    )
    return A_eq, np.zeros(n_rows)


def _build_objective(
    spec: BuildSpec,
    layout: _NoiselessLayout,
    *,
    prices: np.ndarray,
) -> tuple[sp.csc_matrix, np.ndarray]:
    """``min  ‖λ − λ_obs‖²`` — diagonal Hessian on the λ block only."""
    diag = np.zeros(layout.total)
    q = np.zeros(layout.total)

    n_lam = spec.n_buses * spec.n_t
    diag[layout.lam : layout.lam + n_lam] = 2.0
    if n_lam > 0:
        q[layout.lam : layout.lam + n_lam] = -2.0 * prices.T.ravel()
    return sp.diags(diag, format="csc"), q


def _build_bounds(layout: _NoiselessLayout) -> tuple[np.ndarray, np.ndarray]:
    """``c, μ, ν, ξ ≥ 0``; ``λ`` is free."""
    lb = np.full(layout.total, -np.inf)
    ub = np.full(layout.total, np.inf)
    lb[layout.c : layout.c + layout.n_gens] = 0.0
    lb[layout.mu : layout.mu + layout.n_active_lines * layout.n_t] = 0.0
    lb[layout.nu : layout.nu + layout.n_maxed * layout.n_t] = 0.0
    lb[layout.xi : layout.xi + layout.n_min * layout.n_t] = 0.0
    return lb, ub


def _price_residual_norms(
    *,
    x: np.ndarray,
    layout: _NoiselessLayout,
    prices: np.ndarray,
) -> np.ndarray:
    """Per-timestep RMS of ``λ − λ_obs`` — the noiseless reporting metric."""
    if layout.n_t == 0 or layout.n_buses == 0:
        return np.zeros(layout.n_t)
    lam_block = x[
        layout.lam : layout.lam + layout.n_buses * layout.n_t
    ].reshape(layout.n_buses, layout.n_t)
    diff = lam_block - prices.T  # prices is (T, n_buses)
    sse_per_t = np.einsum("bt,bt->t", diff, diff)
    return np.sqrt(sse_per_t / max(layout.n_buses, 1))


__all__ = ["NoiselessFormulation"]
