"""Noisy inverse OPF — KKT-with-slack formulation (Liang-Dvorkin 2023).

Variable layout for the flat primal vector ``z``:

    c   generator costs                       (n_gens)
    λ   bus LMPs                              (n_buses · T)
    μ   line shadow prices on the active set  (|A_lines| · T)
    ν   generator upper-bound duals           (|A_max|   · T)
    ξ   generator lower-bound duals           (|A_min|   · T)
    r   per-generator KKT residual            (n_gens · T)

Per-``(g, t)`` KKT stationarity row::

    r[g,t] − c[g] + λ[bus(g),t]
        − Σ_ℓ PTDF[ℓ, bus(g)] · μ[ℓ,t]
        − ν[g,t]·1{g ∈ A_max}  + ξ[g,t]·1{g ∈ A_min}
    = 0

Objective is the Aswani-Shen-Siddiq (2018) sum of three quadratics
(KKT residual + observation fit + Tikhonov on ``c``); solved as one
sparse QP via :func:`pypsa_invopt.solvers.qp.solve_qp` (HiGHS).

References:
    Liang Z., Dvorkin Y. (2023) eq. (11a) — single-level KKT-QP form.
    Aswani A., Shen Z.J., Siddiq A. (2018) *OR* 66(3) — noisy-KKT
    likelihood with slack ``r``.
    Ruiz C., Conejo A.J. (2009) *IEEE TPWRS* 24(4) — active-set
    complementary-slackness collapse.
"""
from __future__ import annotations

from dataclasses import dataclass, field
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
)
from pypsa_invopt.network import InvoptNetworkData
from pypsa_invopt.solvers import SolverConfig
from pypsa_invopt.solvers.qp import solve_qp
from pypsa_invopt.utils.active_set import ActiveSetBatch

# Storage / link / store dispatch is "at bound" when observed power is
# within ``max(_INTERTEMPORAL_BOUND_TOL_MW, _INTERTEMPORAL_BOUND_TOL_REL · p_nom)``
# of p_max or p_min. The absolute floor handles HiGHS round-off on
# small assets; the relative term scales with capacity so a 5000 MW
# unit doesn't flap on its 0.5 MW absolute tolerance.
_INTERTEMPORAL_BOUND_TOL_MW: float = 0.5
_INTERTEMPORAL_BOUND_TOL_REL: float = 0.001  # 0.1 % of p_nom

# Global-constraint binding threshold: emissions within this fraction
# of the cap count as "binding" for the active-set detector. Default
# 2 % handles HiGHS's primal-feasibility slack on the emission row.
_GLOBAL_CONSTRAINT_BINDING_TOL: float = 0.02


class NoisyFormulation(InverseFormulation):
    """Aswani noisy inverse OPF on a direct sparse QP."""

    name = "noisy"

    def build_model(
        self,
        network_data: InvoptNetworkData,
        batch: ActiveSetBatch,
        observations: dict[str, np.ndarray],
        *,
        lambda_reg: float = 0.1,
        obs_sigma: float = 5.0,
        prior_costs: dict[str, float] | None = None,
        storage_prior_costs: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> _NoisyQP:
        prior = prior_costs or {
            g: max(network_data.gen_marginal_cost[g], 0.1)
            for g in network_data.generators
        }
        s_prior = storage_prior_costs or {
            s: max(network_data.storage_marginal_cost.get(s, 0.0), 0.0)
            for s in network_data.storage_units
        }
        return _NoisyQP.assemble(
            network_data=network_data,
            batch=batch,
            prices=observations["prices"],
            prior_costs=prior,
            obs_sigma=obs_sigma,
            lambda_reg=lambda_reg,
            storage_obs=observations.get("storage"),
            storage_prior_costs=s_prior,
            link_obs=observations.get("link"),
            store_obs=observations.get("store"),
            dispatch_obs=observations.get("dispatch"),
        )

    def solve(
        self,
        model: _NoisyQP,
        solver_config: SolverConfig | None = None,
    ) -> dict[str, Any]:
        verbose = bool(solver_config.verbose) if solver_config else False
        qp = solve_qp(
            Q=model.Q,
            q=model.q,
            A_eq=model.A_eq,
            b_eq=model.b_eq,
            lb=model.lb,
            ub=model.ub,
            verbose=verbose,
        )
        if not qp.is_optimal:
            from pypsa_invopt.exceptions import InvoptConvergenceError
            raise InvoptConvergenceError(
                f"noisy QP did not converge ({qp.status})."
            )

        theta = model.layout.extract_costs(
            qp.x, model.generators, model.storage_units,
            model.links, model.stores, model.global_constraints,
        )
        residuals = model.layout.residual_norms(qp.x, model.n_gens)
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
# Variable layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Layout:
    """Slot offsets for every variable block of the QP.

    Indexing each block is a simple offset + flat index; building this
    once per QP keeps the matrix-construction code free of dimensional
    arithmetic.

    Storage support (Phase 2, 2026-05-14): when ``n_storage > 0`` the
    layout grows by ``n_storage · (1 + 4·n_t)`` slots — one cost slot
    per storage plus four per-snapshot dual blocks (SOC dual ``ν``,
    discharge upper-bound ``μ_d``, charge upper-bound ``μ_s``, SOC
    upper-bound ``μ_soc``). The corresponding KKT equality rows
    (discharge stationarity, store stationarity, cyclic-SOC link)
    live in ``_build_storage_kkt``.
    """

    n_gens: int
    n_buses: int
    n_active_lines: int
    n_maxed: int
    n_min: int
    n_t: int
    n_storage: int
    n_links: int
    n_stores: int
    n_global: int

    c: int
    c_q: int
    lam: int
    mu: int
    nu: int
    xi: int
    r: int
    c_s: int
    nu_soc: int
    mu_d: int
    mu_s: int
    mu_soc: int
    c_link: int
    mu_link: int
    c_store: int
    nu_store: int
    mu_store: int
    mu_global: int
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
        n_storage: int = 0,
        n_links: int = 0,
        n_stores: int = 0,
        n_global: int = 0,
    ) -> _Layout:
        c = 0
        c_q = c + n_gens                     # quadratic-cost block, one per gen
        lam = c_q + n_gens
        mu = lam + n_buses * n_t
        nu = mu + n_active_lines * n_t
        xi = nu + n_maxed * n_t
        r = xi + n_min * n_t
        # Storage blocks at the tail so the existing layout indices
        # remain numerically identical when n_storage == 0.
        c_s = r + n_gens * n_t
        nu_soc = c_s + n_storage
        mu_d = nu_soc + n_storage * n_t
        mu_s = mu_d + n_storage * n_t
        mu_soc = mu_s + n_storage * n_t
        # Link blocks at the tail after storage. Per-link variables:
        # one cost slot (c_link) plus one signed bound dual per (link, t)
        # (mu_link, signed: ≥0 at upper bound, ≤0 at lower).
        c_link = mu_soc + n_storage * n_t
        mu_link = c_link + n_links
        # Store blocks at the tail after links.
        c_store = mu_link + n_links * n_t
        nu_store = c_store + n_stores
        mu_store = nu_store + n_stores * n_t
        # Global-constraint duals (CO2 caps etc.) — one scalar dual per
        # constraint, free ≥ 0 when binding, pinned to 0 when slack.
        mu_global = mu_store + n_stores * n_t
        total = mu_global + n_global
        return cls(
            n_gens=n_gens, n_buses=n_buses, n_active_lines=n_active_lines,
            n_maxed=n_maxed, n_min=n_min, n_t=n_t,
            n_storage=n_storage, n_links=n_links, n_stores=n_stores,
            n_global=n_global,
            c=c, c_q=c_q, lam=lam, mu=mu, nu=nu, xi=xi, r=r,
            c_s=c_s, nu_soc=nu_soc, mu_d=mu_d, mu_s=mu_s, mu_soc=mu_soc,
            c_link=c_link, mu_link=mu_link,
            c_store=c_store, nu_store=nu_store, mu_store=mu_store,
            mu_global=mu_global,
            total=total,
        )

    def nu_soc_idx(self, s: int, t: int) -> int:
        return self.nu_soc + s * self.n_t + t

    def mu_d_idx(self, s: int, t: int) -> int:
        return self.mu_d + s * self.n_t + t

    def mu_s_idx(self, s: int, t: int) -> int:
        return self.mu_s + s * self.n_t + t

    def mu_link_idx(self, link_idx: int, t: int) -> int:
        return self.mu_link + link_idx * self.n_t + t

    def nu_store_idx(self, s: int, t: int) -> int:
        return self.nu_store + s * self.n_t + t

    def mu_store_idx(self, s: int, t: int) -> int:
        return self.mu_store + s * self.n_t + t

    def mu_soc_idx(self, s: int, t: int) -> int:
        return self.mu_soc + s * self.n_t + t

    def lam_idx(self, b: int, t: int) -> int:
        return self.lam + b * self.n_t + t

    def mu_idx(self, line_idx: int, t: int) -> int:
        return self.mu + line_idx * self.n_t + t

    def nu_idx(self, k: int, t: int) -> int:
        return self.nu + k * self.n_t + t

    def xi_idx(self, k: int, t: int) -> int:
        return self.xi + k * self.n_t + t

    def r_idx(self, g: int, t: int) -> int:
        return self.r + g * self.n_t + t

    def extract_costs(
        self,
        z: np.ndarray,
        generators: list[str],
        storage_units: list[str] | None = None,
        links: list[str] | None = None,
        stores: list[str] | None = None,
        global_constraints: list[str] | None = None,
    ) -> dict[str, float]:
        """Pull every cost block out of the solved vector.

        Returns ``gen:<name>:marginal_cost`` for generators and the
        analogous ``storage:``, ``link:``, ``store:`` keys for the
        other dispatchable components when present.
        ``global_constraint:<name>:mu`` carries the recovered CO2
        shadow price (EUR/tCO2).
        """
        out: dict[str, float] = {}
        for i, gen in enumerate(generators):
            out[f"gen:{gen}:marginal_cost"] = float(z[self.c + i])
            # Only emit ``marginal_cost_quadratic`` when it's non-zero;
            # keeps the result dict clean for legacy linear-cost users
            # while still surfacing recovered heat-rate slopes when the
            # caller's network declared them.
            cq_val = float(z[self.c_q + i])
            if abs(cq_val) > 1e-12:
                out[f"gen:{gen}:marginal_cost_quadratic"] = cq_val
        if storage_units:
            for i, s in enumerate(storage_units):
                out[f"storage:{s}:marginal_cost"] = float(z[self.c_s + i])
        if links:
            for i, ln in enumerate(links):
                out[f"link:{ln}:marginal_cost"] = float(z[self.c_link + i])
        if stores:
            for i, s in enumerate(stores):
                out[f"store:{s}:marginal_cost"] = float(z[self.c_store + i])
        if global_constraints:
            for i, gc in enumerate(global_constraints):
                out[f"global_constraint:{gc}:mu"] = float(z[self.mu_global + i])
        return out

    def residual_norms(self, z: np.ndarray, n_gens: int) -> np.ndarray:
        """Per-timestep RMS of the recovered KKT residual ``r``."""
        if self.n_t == 0:
            return np.zeros(0)
        r_block = z[self.r : self.r + n_gens * self.n_t].reshape(n_gens, self.n_t)
        sse_per_t = np.einsum("gt,gt->t", r_block, r_block)
        return np.sqrt(sse_per_t / max(n_gens, 1))


# ---------------------------------------------------------------------------
# QP container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _NoisyQP:
    """The fully-assembled QP for one ASTB batch."""

    Q: sp.csc_matrix
    q: np.ndarray
    A_eq: sp.csc_matrix
    b_eq: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    layout: _Layout
    generators: list[str]
    n_gens: int

    storage_units: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    stores: list[str] = field(default_factory=list)
    global_constraints: list[str] = field(default_factory=list)

    @classmethod
    def assemble(
        cls,
        *,
        network_data: InvoptNetworkData,
        batch: ActiveSetBatch,
        prices: np.ndarray,
        prior_costs: dict[str, float],
        obs_sigma: float,
        lambda_reg: float,
        storage_obs: dict[str, np.ndarray] | None = None,
        storage_prior_costs: dict[str, float] | None = None,
        link_obs: dict[str, np.ndarray] | None = None,
        store_obs: dict[str, np.ndarray] | None = None,
        dispatch_obs: np.ndarray | None = None,
    ) -> _NoisyQP:
        spec = BuildSpec.from_inputs(network_data=network_data, batch=batch)
        # Each component family is opt-in: when there are no observations
        # for the family, we omit its layout slots entirely and the
        # existing storage-less / link-less / store-less code paths stay
        # byte-identical numerically. This keeps legacy tests stable.
        storage_units = list(network_data.storage_units) if storage_obs else []
        links = list(network_data.links) if link_obs else []
        stores = list(network_data.stores) if store_obs else []
        # Global constraints (CO2 caps) are opt-in via the network's
        # ``global_constraints`` table — no explicit observation column
        # is required since the constraint is system-wide.
        global_constraints = list(network_data.global_constraints)
        layout = _Layout.for_shape(
            n_gens=spec.n_gens,
            n_buses=spec.n_buses,
            n_active_lines=len(spec.active_lines),
            n_maxed=len(spec.maxed_gens),
            n_min=len(spec.min_gens),
            n_t=spec.n_t,
            n_storage=len(storage_units),
            n_links=len(links),
            n_stores=len(stores),
            n_global=len(global_constraints),
        )

        A_eq, b_eq = _build_kkt_equality(spec, layout, dispatch_obs=dispatch_obs)
        if storage_units:
            A_storage, b_storage = _build_storage_kkt(
                spec=spec, layout=layout,
                network_data=network_data,
                storage_units=storage_units,
            )
            A_eq = sp.vstack([A_eq, A_storage], format="csc")
            b_eq = np.concatenate([b_eq, b_storage])
        if links:
            A_link, b_link = _build_link_kkt(
                spec=spec, layout=layout,
                network_data=network_data,
                links=links,
            )
            A_eq = sp.vstack([A_eq, A_link], format="csc")
            b_eq = np.concatenate([b_eq, b_link])
        if stores:
            A_store, b_store = _build_store_kkt(
                spec=spec, layout=layout,
                network_data=network_data,
                stores=stores,
            )
            A_eq = sp.vstack([A_eq, A_store], format="csc")
            b_eq = np.concatenate([b_eq, b_store])
        Q, q = _build_objective(
            spec, layout,
            prices=prices,
            obs_sigma=obs_sigma,
            lambda_reg=lambda_reg,
            prior_costs=prior_costs,
            storage_units=storage_units,
            storage_prior_costs=storage_prior_costs or {},
            links=links,
            stores=stores,
            network_data=network_data,
        )
        lb, ub = _build_bounds(
            layout=layout,
            storage_units=storage_units,
            storage_obs=storage_obs,
            network_data=network_data,
            links=links,
            link_obs=link_obs,
            stores=stores,
            store_obs=store_obs,
            global_constraints=global_constraints,
            dispatch_obs=dispatch_obs,
        )

        return cls(
            Q=Q, q=q, A_eq=A_eq, b_eq=b_eq, lb=lb, ub=ub,
            layout=layout,
            storage_units=storage_units,
            links=links,
            stores=stores,
            global_constraints=global_constraints,
            generators=spec.generators,
            n_gens=spec.n_gens,
        )


# ---------------------------------------------------------------------------
# Equality constraints  — the KKT row block
# ---------------------------------------------------------------------------


def _build_kkt_equality(
    spec: BuildSpec, layout: _Layout,
    dispatch_obs: np.ndarray | None = None,
) -> tuple[sp.csc_matrix, np.ndarray]:
    """Build the per-``(g, t)`` KKT-stationarity equality block.

    For each generator–timestep pair the row reads

        r[g, t]  −  c[g]  +  λ_{bus(g), t}
                  −  Σ_l  PTDF_{l, bus(g)}  μ_{l, t}
                  −  ν_{g, t} · 1{g ∈ A_max}
                  +  ξ_{g, t} · 1{g ∈ A_min}
                  −  Σ_gc  (e_g · w_t)  μ_{global, gc}      [CO2 cap term]
        =  0

    The last term is the carbon-price contribution: when a CO2 cap
    is binding (PyPSA's "primary_energy" global constraint), each
    generator's effective marginal cost gains ``e_g · w_t · μ_co2``
    per the standard KKT derivation. ``e_g`` is the per-MWh
    electrical emission factor (=carrier emission / efficiency),
    ``w_t`` the snapshot weight (1 h default), and ``μ_co2`` is the
    recovered shadow price (EUR/tCO2).

    Constructed in pure numpy — every block's ``(rows, cols, data)``
    vector is produced by ``np.repeat`` / ``np.tile`` and concatenated
    once at the end. Replaces the Python triple-loop that scaled
    badly on PyPSA-Eur-sized batches.
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

    # r — one diagonal entry per (g, t).
    r_cols = layout.r + rows_flat
    r_data = np.ones(n_rows)

    # −c[g] on row (g, t).
    c_cols = layout.c + np.repeat(np.arange(n_gens, dtype=int), n_t)
    c_data = -np.ones(n_rows)

    # Quadratic-cost coefficient: ``− 2 · p_obs[g, t] · c_q[g]`` on
    # row (g, t). When observed dispatch is supplied AND the network
    # declares non-zero ``marginal_cost_quadratic`` for at least one
    # generator, we inject this term. Otherwise — by far the common
    # case — we skip the c_q coefficient entirely so the QP is
    # numerically identical to the pre-Gap-1 implementation. The
    # ``c_q ≥ 0`` bound + zero-anchor regularisation still keeps the
    # variable pinned near 0 even when injection is on, but the data-
    # signal term must be present for the heat-rate inversion.
    quadratic_active = (
        dispatch_obs is not None
        and dispatch_obs.shape == (n_t, n_gens)
        and any(
            spec.network_data.gen_marginal_cost_quadratic.get(g, 0.0) > 0.0
            for g in spec.generators
        )
    )
    if quadratic_active:
        cq_cols = layout.c_q + np.repeat(np.arange(n_gens, dtype=int), n_t)
        # Row (g, t) → coefficient on c_q[g] is −2·p_obs[t, g].
        # dispatch_obs is laid out (t, g); transpose-then-ravel to (g, t)
        # row-major so it matches ``rows_flat = arange(g·n_t + t)``.
        cq_data = -2.0 * dispatch_obs.T.ravel()
    else:
        cq_cols = np.zeros(0, dtype=int)
        cq_data = np.zeros(0)

    # +λ[bus(g), t] on row (g, t).
    lam_cols = layout.lam + np.repeat(spec.gen_bus_idx, n_t) * n_t + np.tile(t_range, n_gens)
    lam_data = np.ones(n_rows)

    mu_rows, mu_cols, mu_data = mu_kkt_block(
        n_t=n_t, ptdf_active=ptdf_active, mu_offset=layout.mu,
        t_range=t_range, sign=-1.0,
    )
    nu_rows, nu_cols, nu_data = bound_dual_kkt_block(
        gen_indices=maxed_or_min_gen_indices(spec, spec.maxed_gens),
        n_t=n_t, dual_offset=layout.nu, sign=-1.0, t_range=t_range,
    )
    xi_rows, xi_cols, xi_data = bound_dual_kkt_block(
        gen_indices=maxed_or_min_gen_indices(spec, spec.min_gens),
        n_t=n_t, dual_offset=layout.xi, sign=+1.0, t_range=t_range,
    )

    # CO2 / global-constraint term: for every (g, t) row, add
    # ``− (e_g · w_t) · μ_global[gc]`` for each active global
    # constraint. The KKT sign matches the convention used for
    # generator-bound duals (``− ν``): the dual carries the cost
    # of the binding constraint, which the inverse problem will
    # recover as ``μ_co2`` (EUR/tCO2).
    gc_rows_list: list[np.ndarray] = []
    gc_cols_list: list[np.ndarray] = []
    gc_data_list: list[np.ndarray] = []
    if layout.n_global > 0 and spec.network_data.gen_emission_factor:
        gen_emissions = np.array(
            [spec.network_data.gen_emission_factor.get(g, 0.0)
             for g in spec.generators], dtype=float,
        )
        snap_w = spec.network_data.snapshot_weights
        if snap_w is None or len(snap_w) < n_t:
            snap_w = np.ones(n_t, dtype=float)
        snap_w_used = snap_w[:n_t]
        # Coefficient for (g, t): -e_g * w_t. Broadcast outer product.
        eg_wt = -np.outer(gen_emissions, snap_w_used).ravel()
        # Only emit non-zero coefficients to keep the matrix sparse.
        for gc_idx in range(layout.n_global):
            nz_mask = eg_wt != 0.0
            if not np.any(nz_mask):
                continue
            rows_local = np.arange(n_rows, dtype=int)[nz_mask]
            gc_rows_list.append(rows_local)
            gc_cols_list.append(
                np.full(int(nz_mask.sum()), layout.mu_global + gc_idx, dtype=int)
            )
            gc_data_list.append(eg_wt[nz_mask])

    all_data = [r_data, c_data, lam_data, mu_data, nu_data, xi_data]
    all_rows = [rows_flat, rows_flat, rows_flat, mu_rows, nu_rows, xi_rows]
    all_cols = [r_cols, c_cols, lam_cols, mu_cols, nu_cols, xi_cols]
    if cq_data.size > 0:
        # Each c_q entry sits in row ``g · n_t + t`` (= ``rows_flat[i]``)
        # and column ``layout.c_q + g``. The data was built with
        # ``dispatch_obs.T.ravel()`` which is g-major, t-minor — so
        # element i = (g · n_t + t) lines up with rows_flat[i] = i.
        all_data.append(cq_data)
        all_rows.append(rows_flat)
        all_cols.append(cq_cols)
    if gc_rows_list:
        all_data += gc_data_list
        all_rows += gc_rows_list
        all_cols += gc_cols_list

    A_eq = sp.csc_matrix(
        (
            np.concatenate(all_data),
            (np.concatenate(all_rows), np.concatenate(all_cols)),
        ),
        shape=(n_rows, layout.total),
    )
    return A_eq, np.zeros(n_rows)


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


def _build_objective(
    spec: BuildSpec,
    layout: _Layout,
    *,
    prices: np.ndarray,
    obs_sigma: float,
    lambda_reg: float,
    prior_costs: dict[str, float],
    storage_units: list[str] | None = None,
    storage_prior_costs: dict[str, float] | None = None,
    links: list[str] | None = None,
    stores: list[str] | None = None,
    network_data: InvoptNetworkData | None = None,
) -> tuple[sp.csc_matrix, np.ndarray]:
    """Build the diagonal Hessian ``Q`` and linear term ``q``.

    HiGHS minimises ``(1/2) zᵀ Q z + qᵀ z``, so the squared-residual
    ``α x²`` becomes ``Q[i,i] = 2α`` and the cross term ``−2α a · x``
    becomes ``q[i] = −2α a``. Constants are dropped.

    The objective decomposes into three diagonal blocks (no
    off-diagonal entries) — every coupling lives in the equality block.
    """
    n_t = max(spec.n_t, 1)
    inv_obs = 1.0 / (obs_sigma ** 2)
    weight_lam = inv_obs / n_t
    weight_r = 1.0 / n_t

    diag = np.zeros(layout.total)
    q = np.zeros(layout.total)

    # c block: λ_reg · ‖c − c_prior‖²
    prior_vec = np.array(
        [float(prior_costs.get(g, 0.0)) for g in spec.generators], dtype=float,
    )
    diag[layout.c : layout.c + spec.n_gens] = 2.0 * lambda_reg
    q[layout.c : layout.c + spec.n_gens] = -2.0 * lambda_reg * prior_vec

    # c_q (quadratic-cost) block: anchor at the PyPSA prior
    # ``marginal_cost_quadratic``. Default = 0 (pure linear cost),
    # which keeps the recovered c_q at 0 unless the data informs
    # otherwise. Same Tikhonov shape as the linear cost block.
    cq_prior = np.array(
        [float(network_data.gen_marginal_cost_quadratic.get(g, 0.0))
         for g in spec.generators]
        if network_data is not None else [0.0] * spec.n_gens,
        dtype=float,
    )
    diag[layout.c_q : layout.c_q + spec.n_gens] = 2.0 * lambda_reg
    q[layout.c_q : layout.c_q + spec.n_gens] = -2.0 * lambda_reg * cq_prior

    # λ block: weight_lam · ‖λ − λ_obs‖²
    # ``λ`` is laid out (bus, timestep) row-major, so ``prices.T.ravel()`` —
    # which iterates bus-major, timestep-minor — lines up with the slot order.
    n_lam = spec.n_buses * spec.n_t
    diag[layout.lam : layout.lam + n_lam] = 2.0 * weight_lam
    if n_lam > 0:
        q[layout.lam : layout.lam + n_lam] = -2.0 * weight_lam * prices.T.ravel()

    # r block: weight_r · ‖r‖²  (no linear part)
    diag[layout.r : layout.r + spec.n_gens * spec.n_t] = 2.0 * weight_r

    # Storage c_s block: λ_reg · ‖c_s − c_s_prior‖²
    if storage_units:
        priors = storage_prior_costs or {}
        s_prior_vec = np.array(
            [float(priors.get(s, 0.0)) for s in storage_units], dtype=float,
        )
        diag[layout.c_s : layout.c_s + len(storage_units)] = 2.0 * lambda_reg
        q[layout.c_s : layout.c_s + len(storage_units)] = (
            -2.0 * lambda_reg * s_prior_vec
        )

    # Link c_link block: λ_reg · ‖c_link − prior‖² (anchored at the
    # PyPSA network's existing ``marginal_cost`` field).
    if links and network_data is not None:
        link_prior = np.array(
            [float(network_data.link_marginal_cost.get(ln, 0.0))
             for ln in links], dtype=float,
        )
        diag[layout.c_link : layout.c_link + len(links)] = 2.0 * lambda_reg
        q[layout.c_link : layout.c_link + len(links)] = (
            -2.0 * lambda_reg * link_prior
        )

    # Store c_store block: same shape as storage.
    if stores and network_data is not None:
        store_prior = np.array(
            [float(network_data.store_marginal_cost.get(s, 0.0))
             for s in stores], dtype=float,
        )
        diag[layout.c_store : layout.c_store + len(stores)] = 2.0 * lambda_reg
        q[layout.c_store : layout.c_store + len(stores)] = (
            -2.0 * lambda_reg * store_prior
        )

    return sp.diags(diag, format="csc"), q


# ---------------------------------------------------------------------------
# Variable bounds
# ---------------------------------------------------------------------------


def _build_bounds(
    *,
    layout: _Layout,
    storage_units: list[str] | None = None,
    storage_obs: dict[str, np.ndarray] | None = None,
    network_data: InvoptNetworkData | None = None,
    links: list[str] | None = None,
    link_obs: dict[str, np.ndarray] | None = None,
    stores: list[str] | None = None,
    store_obs: dict[str, np.ndarray] | None = None,
    global_constraints: list[str] | None = None,
    dispatch_obs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Lower / upper bounds aligned with the variable layout.

    Non-negative blocks (``c``, ``μ``, ``ν``, ``ξ``) get ``[0, ∞)``;
    free blocks (``λ``, ``r``) get ``(−∞, ∞)``.

    Storage blocks (Phase 2):
    * ``c_s``: ``[0, ∞)`` (marginal costs non-negative).
    * ``ν_soc``: ``[0, ∞)`` (energy has non-negative value).
    * ``μ_d`` / ``μ_s`` / ``μ_soc``: complementary slackness — when
      the observed dispatch / storage / SOC is *not* at its bound at
      ``(s, t)``, the corresponding dual is pinned to ``0`` via
      ``lb = ub = 0``. When the bound is binding, the dual is free
      in ``[0, ∞)``.
    """
    lb = np.full(layout.total, -np.inf)
    ub = np.full(layout.total, np.inf)

    # Generator-side non-negative blocks: c, c_q, μ, ν, ξ.
    # ``c_q ≥ 0`` enforces convexity of the quadratic-cost curve
    # (a negative quadratic term would imply non-convex marginal cost,
    # which neither PyPSA's forward LOPF nor classical merit-order
    # admits; convexity is the canonical assumption in
    # Birge-Hortaçsu-Pavlin (2017) eq. (3) and Liang-Dvorkin (2023)).
    lb[layout.c : layout.c + layout.n_gens] = 0.0
    lb[layout.c_q : layout.c_q + layout.n_gens] = 0.0
    lb[layout.mu : layout.mu + layout.n_active_lines * layout.n_t] = 0.0
    lb[layout.nu : layout.nu + layout.n_maxed * layout.n_t] = 0.0
    lb[layout.xi : layout.xi + layout.n_min * layout.n_t] = 0.0

    # Storage blocks
    if storage_units and storage_obs is not None and network_data is not None:
        n_s = len(storage_units)
        n_t = layout.n_t
        lb[layout.c_s : layout.c_s + n_s] = 0.0
        lb[layout.nu_soc : layout.nu_soc + n_s * n_t] = 0.0

        p_disp = storage_obs["p_dispatch"]   # (n_t, n_s)
        p_store = storage_obs["p_store"]
        soc = storage_obs.get("soc")          # (n_t, n_s) or None
        # Caller can override; otherwise scale tolerance per-unit by p_nom.
        tol_abs = float(storage_obs.get(
            "active_set_tol", _INTERTEMPORAL_BOUND_TOL_MW,
        ))

        # Signed-dual convention (handles both p_dispatch ≥ 0 and
        # p_dispatch ≤ p_nom complementarity in one variable):
        #   * Interior (0 < p < p_nom): μ pinned to 0 (lb = ub = 0).
        #   * At upper bound (p = p_nom): μ free in [0, +∞).
        #   * At lower bound (p = 0):     μ free in (−∞, 0].
        # Same for store. For SOC we only model upper bound (SOC ≥ 0
        # is typically non-binding in practice).
        for s_idx, s in enumerate(storage_units):
            p_nom = float(network_data.storage_p_nom[s])
            max_hours = float(network_data.storage_max_hours.get(s, 1.0))
            soc_max = p_nom * max_hours
            tol = max(tol_abs, _INTERTEMPORAL_BOUND_TOL_REL * p_nom)
            for t in range(n_t):
                # μ_d: discharge dual
                idx_d = layout.mu_d_idx(s_idx, t)
                if p_disp[t, s_idx] >= p_nom - tol:
                    lb[idx_d], ub[idx_d] = 0.0, np.inf       # at upper
                elif p_disp[t, s_idx] <= tol:
                    lb[idx_d], ub[idx_d] = -np.inf, 0.0       # at lower
                else:
                    lb[idx_d], ub[idx_d] = 0.0, 0.0           # interior
                # μ_s: store dual
                idx_s = layout.mu_s_idx(s_idx, t)
                if p_store[t, s_idx] >= p_nom - tol:
                    lb[idx_s], ub[idx_s] = 0.0, np.inf
                elif p_store[t, s_idx] <= tol:
                    lb[idx_s], ub[idx_s] = -np.inf, 0.0
                else:
                    lb[idx_s], ub[idx_s] = 0.0, 0.0
                # μ_soc: SOC upper bound (skip lower bound for now)
                idx_soc = layout.mu_soc_idx(s_idx, t)
                if soc is not None and soc[t, s_idx] >= soc_max - tol:
                    lb[idx_soc], ub[idx_soc] = 0.0, np.inf
                else:
                    lb[idx_soc], ub[idx_soc] = 0.0, 0.0

    # Link blocks
    if links and link_obs is not None and network_data is not None:
        n_l = len(links)
        n_t = layout.n_t
        lb[layout.c_link : layout.c_link + n_l] = 0.0
        p_link = link_obs["p"]   # (n_t, n_links) — observed signed flow
        tol_abs = float(link_obs.get("active_set_tol", _INTERTEMPORAL_BOUND_TOL_MW))
        for l_idx, ln in enumerate(links):
            p_nom = float(network_data.link_p_nom[ln])
            p_min = float(network_data.link_p_min_pu.get(ln, 0.0)) * p_nom
            p_max = float(network_data.link_p_max_pu.get(ln, 1.0)) * p_nom
            tol = max(tol_abs, _INTERTEMPORAL_BOUND_TOL_REL * p_nom)
            for t in range(n_t):
                idx = layout.mu_link_idx(l_idx, t)
                if p_link[t, l_idx] >= p_max - tol:
                    lb[idx], ub[idx] = 0.0, np.inf       # at upper bound
                elif p_link[t, l_idx] <= p_min + tol:
                    lb[idx], ub[idx] = -np.inf, 0.0       # at lower bound
                else:
                    lb[idx], ub[idx] = 0.0, 0.0           # interior

    # Global constraints (CO2 caps) — detect whether the cap is
    # binding from the observed dispatch, then pin or free μ_global
    # accordingly.
    #
    # Active-set detection: compute
    #   total_emissions = Σ_t Σ_g  e_g · p_g[g, t] · w_t
    # and compare to the constraint's ``constant``. Binding when
    # total_emissions ≥ (1 − ε) · constant.
    if (
        global_constraints and network_data is not None
        and dispatch_obs is not None and dispatch_obs.size > 0
    ):
        # dispatch_obs has shape (n_t, n_gens). Build e_g vector in
        # the same generator order.
        n_t_loc = dispatch_obs.shape[0]
        gen_emissions = np.array(
            [network_data.gen_emission_factor.get(g, 0.0)
             for g in network_data.generators], dtype=float,
        )
        snap_w = network_data.snapshot_weights
        if snap_w is None or len(snap_w) < n_t_loc:
            snap_w = np.ones(n_t_loc, dtype=float)
        snap_w_used = snap_w[:n_t_loc]
        weighted_dispatch = dispatch_obs * snap_w_used[:, None]   # (n_t, n_gens)
        total_emissions = float((weighted_dispatch * gen_emissions[None, :]).sum())

        # uses module-level _GLOBAL_CONSTRAINT_BINDING_TOL
        for gc_idx, gc in enumerate(global_constraints):
            cap = float(network_data.global_constraint_constant.get(gc, 0.0))
            sense = network_data.global_constraint_sense.get(gc, "<=")
            idx = layout.mu_global + gc_idx
            # μ_global ≥ 0 always. Pin to 0 (lb = ub = 0) when slack.
            if cap <= 0.0 or sense not in {"<=", "==", "<"}:
                lb[idx], ub[idx] = 0.0, 0.0
                continue
            ratio = total_emissions / cap
            if ratio >= (1.0 - _GLOBAL_CONSTRAINT_BINDING_TOL):
                lb[idx], ub[idx] = 0.0, np.inf       # binding → recoverable μ
            else:
                lb[idx], ub[idx] = 0.0, 0.0           # slack → μ pinned to 0

    # Store blocks (mirror of storage_units but with single signed p variable)
    if stores and store_obs is not None and network_data is not None:
        n_s = len(stores)
        n_t = layout.n_t
        lb[layout.c_store : layout.c_store + n_s] = 0.0
        lb[layout.nu_store : layout.nu_store + n_s * n_t] = 0.0  # SOC value ≥ 0
        p_store_obs = store_obs["p"]   # (n_t, n_stores) signed power
        tol_abs = float(store_obs.get("active_set_tol", _INTERTEMPORAL_BOUND_TOL_MW))
        for s_idx, s in enumerate(stores):
            e_nom = float(network_data.store_e_nom[s])
            # Stores have no separate p_nom; treat as |p| ≤ e_nom / hr.
            # The signed-dual handles both bounds.
            p_cap = e_nom   # PyPSA's default: p unbounded; if e_nom finite use it
            tol = max(tol_abs, _INTERTEMPORAL_BOUND_TOL_REL * p_cap)
            for t in range(n_t):
                idx = layout.mu_store_idx(s_idx, t)
                if p_store_obs[t, s_idx] >= p_cap - tol:
                    lb[idx], ub[idx] = 0.0, np.inf
                elif p_store_obs[t, s_idx] <= -p_cap + tol:
                    lb[idx], ub[idx] = -np.inf, 0.0
                else:
                    lb[idx], ub[idx] = 0.0, 0.0

    return lb, ub


# ---------------------------------------------------------------------------
# Storage KKT block (Phase 2 — intertemporal SOC dynamics)
# ---------------------------------------------------------------------------


def _build_storage_kkt(
    *,
    spec: BuildSpec,
    layout: _Layout,
    network_data: InvoptNetworkData,
    storage_units: list[str],
) -> tuple[sp.csc_matrix, np.ndarray]:
    """Build the storage KKT-stationarity rows.

    Three rows per ``(storage s, timestep t)``:

    1. **Discharge stationarity**::

           c_s − λ[bus(s), t] + ν[s, t] / η_dispatch + μ_d[s, t] = 0

    2. **Store stationarity**::

           λ[bus(s), t] − η_store · ν[s, t] + μ_s[s, t] = 0

    3. **SOC link (cyclic)**::

           ν[s, t] − ν[s, (t+1) mod n_t] + μ_soc[s, t] = 0

    Cyclic SOC matches PyPSA's default ``cyclic_state_of_charge=True``.
    Standing loss is assumed zero (PyPSA default) — adding it is a
    multiplicative factor on the SOC-link `ν` term.
    """
    n_t = spec.n_t
    n_s = len(storage_units)
    n_rows = 3 * n_s * n_t
    if n_rows == 0:
        return sp.csc_matrix((0, layout.total)), np.zeros(0)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    row_idx = 0
    for s_idx, s in enumerate(storage_units):
        bus = network_data.storage_bus[s]
        bus_idx = network_data.bus_index[bus]
        eta_d = float(network_data.storage_efficiency_dispatch.get(s, 1.0))
        eta_s = float(network_data.storage_efficiency_store.get(s, 1.0))

        for t in range(n_t):
            # Row 1 — discharge stationarity
            #   c_s − λ[bus,t] + ν[s,t]/η_d + μ_d[s,t] = 0
            rows += [row_idx, row_idx, row_idx, row_idx]
            cols += [
                layout.c_s + s_idx,
                layout.lam + bus_idx * n_t + t,
                layout.nu_soc_idx(s_idx, t),
                layout.mu_d_idx(s_idx, t),
            ]
            data += [+1.0, -1.0, +1.0 / max(eta_d, 1e-9), +1.0]
            row_idx += 1

            # Row 2 — store stationarity
            #   λ[bus,t] − η_s · ν[s,t] + μ_s[s,t] = 0
            rows += [row_idx, row_idx, row_idx]
            cols += [
                layout.lam + bus_idx * n_t + t,
                layout.nu_soc_idx(s_idx, t),
                layout.mu_s_idx(s_idx, t),
            ]
            data += [+1.0, -eta_s, +1.0]
            row_idx += 1

            # Row 3 — SOC link (cyclic)
            #   ν[s,t] − ν[s,(t+1) mod n_t] + μ_soc[s,t] = 0
            t_next = (t + 1) % n_t
            rows += [row_idx, row_idx, row_idx]
            cols += [
                layout.nu_soc_idx(s_idx, t),
                layout.nu_soc_idx(s_idx, t_next),
                layout.mu_soc_idx(s_idx, t),
            ]
            data += [+1.0, -1.0, +1.0]
            row_idx += 1

    A_eq = sp.csc_matrix(
        (np.array(data, dtype=float),
         (np.array(rows, dtype=int), np.array(cols, dtype=int))),
        shape=(n_rows, layout.total),
    )
    return A_eq, np.zeros(n_rows)


# ---------------------------------------------------------------------------
# Link KKT block (HVDC, P2H, electrolysers, etc.)
# ---------------------------------------------------------------------------


def _build_link_kkt(
    *,
    spec: BuildSpec,
    layout: _Layout,
    network_data: InvoptNetworkData,
    links: list[str],
) -> tuple[sp.csc_matrix, np.ndarray]:
    """Build the link KKT-stationarity row block.

    PyPSA models a ``Link`` as a controllable power transfer from
    ``bus0`` to ``bus1`` with signed flow ``p`` and efficiency ``η``.
    Bus balance contributions: ``-p`` at bus0, ``+η · p`` at bus1.
    Objective contribution: ``c_link · p`` (cost on flow).

    KKT stationarity for ``p`` (one row per ``(link, t)``)::

        c_link + λ[bus0, t] − η · λ[bus1, t] + μ_link[link, t] = 0

    Derivation: the bus balance in PyPSA's convention reads
    ``load + outflows − supply − inflows = 0`` per bus, so the link
    contributes ``+p`` to ``b0``'s balance (outflow) and ``−η · p`` to
    ``b1``'s (inflow with efficiency loss). Differentiating the
    Lagrangian ``∑ c · p + λ_b · (...)`` w.r.t. ``p`` produces
    ``c_link + λ[b0] − η · λ[b1] + (bound duals) = 0``. At interior
    this gives the well-known arbitrage identity
    ``λ[b1] = (c_link + λ[b0]) / η`` — the price at the destination
    equals the source price plus the link's marginal cost, scaled by
    the inverse efficiency.

    The signed dual ``μ_link`` carries both upper- and lower-bound
    complementarity (positive at ``p_max``, negative at ``p_min``,
    zero in the interior — same pattern as the storage dispatch dual).
    """
    n_t = spec.n_t
    n_l = len(links)
    n_rows = n_l * n_t
    if n_rows == 0:
        return sp.csc_matrix((0, layout.total)), np.zeros(0)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    row_idx = 0
    for l_idx, ln in enumerate(links):
        bus0_idx = network_data.bus_index[network_data.link_bus0[ln]]
        bus1_idx = network_data.bus_index[network_data.link_bus1[ln]]
        eta = float(network_data.link_efficiency.get(ln, 1.0))
        for t in range(n_t):
            rows += [row_idx, row_idx, row_idx, row_idx]
            cols += [
                layout.c_link + l_idx,
                layout.lam + bus0_idx * n_t + t,
                layout.lam + bus1_idx * n_t + t,
                layout.mu_link_idx(l_idx, t),
            ]
            data += [+1.0, +1.0, -eta, +1.0]
            row_idx += 1

    A_eq = sp.csc_matrix(
        (np.array(data, dtype=float),
         (np.array(rows, dtype=int), np.array(cols, dtype=int))),
        shape=(n_rows, layout.total),
    )
    return A_eq, np.zeros(n_rows)


# ---------------------------------------------------------------------------
# Store KKT block (H₂, heat — single signed p variable)
# ---------------------------------------------------------------------------


def _build_store_kkt(
    *,
    spec: BuildSpec,
    layout: _Layout,
    network_data: InvoptNetworkData,
    stores: list[str],
) -> tuple[sp.csc_matrix, np.ndarray]:
    """Build the store KKT-stationarity row block.

    PyPSA's ``Store`` uses a single signed power variable ``p`` (positive
    = discharge into bus, negative = charge from bus) and an energy
    state ``e[t] = e[t−1] · (1 − loss) − p[t]``.

    KKT rows per ``(store, t)``:

    1. **Power stationarity** (∂L/∂p)::

           c_store − λ[bus, t] − ν[store, t] + μ_store[store, t] = 0

    2. **Energy / SOC link (cyclic)** (∂L/∂e)::

           ν[store, t] − ν[store, (t+1) mod n_t] · (1 − loss) = 0

    The signed dual ``μ_store`` handles both p≥|cap| and p≤−|cap|
    complementarity. The SOC bound shadow on ``e ≤ e_nom`` is
    implicit (absorbed into ``ν`` when SOC saturates).
    """
    n_t = spec.n_t
    n_s = len(stores)
    n_rows = 2 * n_s * n_t   # power + SOC link per (store, t)
    if n_rows == 0:
        return sp.csc_matrix((0, layout.total)), np.zeros(0)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    row_idx = 0
    for s_idx, s in enumerate(stores):
        bus_idx = network_data.bus_index[network_data.store_bus[s]]
        loss = float(network_data.store_standing_loss.get(s, 0.0))
        for t in range(n_t):
            # Row 1 — power stationarity
            #   c_store − λ[bus,t] − ν[s,t] + μ_store[s,t] = 0
            rows += [row_idx, row_idx, row_idx, row_idx]
            cols += [
                layout.c_store + s_idx,
                layout.lam + bus_idx * n_t + t,
                layout.nu_store_idx(s_idx, t),
                layout.mu_store_idx(s_idx, t),
            ]
            data += [+1.0, -1.0, -1.0, +1.0]
            row_idx += 1

            # Row 2 — SOC link (cyclic, with standing loss)
            t_next = (t + 1) % n_t
            rows += [row_idx, row_idx]
            cols += [
                layout.nu_store_idx(s_idx, t),
                layout.nu_store_idx(s_idx, t_next),
            ]
            data += [+1.0, -(1.0 - loss)]
            row_idx += 1

    A_eq = sp.csc_matrix(
        (np.array(data, dtype=float),
         (np.array(rows, dtype=int), np.array(cols, dtype=int))),
        shape=(n_rows, layout.total),
    )
    return A_eq, np.zeros(n_rows)


__all__ = ["NoisyFormulation"]
