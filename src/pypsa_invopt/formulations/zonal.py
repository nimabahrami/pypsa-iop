"""Zonal inverse OPF — zone-price duality with NTC shadow prices.

**Mathematical pedigree** (verified 2026-05-14):

* Zonal duality `λ_z = c_z + Σ sign·μ_p` follows **Bjørndal &
  Jørnsten (2001)**, *The Energy Journal* 22(1). Same shape as the
  Liang-Dvorkin (2023) LMP decomposition (eq. 14, 15) when the
  zonal model is the limit of a single-bus-per-zone DCOPF.
* Complementary slackness `μ_p = 0` off the active set and NTC
  lower bound `NTC_p ≥ max_{t ∈ congested} |f_obs[p, t]|` follow
  directly from the Lagrangian.
* `c_z ≥ 0` non-negativity is the marginal-cost interpretation.
* The 2026-05-14 sweep added a `c_prior` anchor — without it, the
  L₂ regulariser silently pulled `c_z` toward 0 (non-physical;
  see the project memory).

This implementation also corrects a column-order bug discovered in
the same sweep where per-bus observed prices were misaligned with
the QP's per-zone slots when many buses shared one zone (the
European multi-bus case). The fix lives in
``calibration._extract_observations``.

Variable layout (HiGHS column order):

* ``c_z``      — zone marginal costs           (n_zones)
* ``λ_z``      — zone LMPs                     (n_zones · T)
* ``μ_pair``   — signed NTC shadow prices      (n_pairs · T)
* ``ntc``      — recovered NTC magnitudes      (n_pairs)

Constraints — equalities only:

    λ_z[z, t]  −  c_z[z]  −  Σ_{p: z endpoint}  sign(z, p) · μ_pair[p, t]  =  0

Other model conditions are expressed as **variable bounds** (bounds
are cheaper than inequality rows in the HiGHS QP form):

* Uncongested ``(pair, timestep)`` cells: ``lb = ub = 0`` on
  ``μ_pair[pair, t]`` — complementary slackness pinning the dual.
* NTC lower bound: ``ntc[pair] ≥ max_{t ∈ congested}  |f_obs[pair, t]|``
  — the maximum is computed once before the QP is built, replacing
  what would otherwise be one inequality per congested cell.

Reference: Bjørndal & Jørnsten (2001), zonal-pricing duality.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

from pypsa_invopt._constants import (
    ZONAL_MU_REGULARISER,
)
from pypsa_invopt.formulations.base import InverseFormulation, solve_qp_or_raise
from pypsa_invopt.network import InvoptNetworkData
from pypsa_invopt.solvers import SolverConfig
from pypsa_invopt.utils.active_set import ActiveSetBatch


class ZonalFormulation(InverseFormulation):
    """Zonal inverse OPF on a direct sparse QP."""

    name = "zonal"

    def build_model(
        self,
        network_data: InvoptNetworkData,
        batch: ActiveSetBatch,
        observations: dict[str, np.ndarray],
        *,
        zones: list[str] | None = None,
        bus_to_zone: dict[str, str] | None = None,
        zone_pairs: list[tuple[str, str]] | None = None,
        flows_per_pair: dict[str, np.ndarray] | None = None,
        ntc_prior: dict[tuple[str, str], float] | None = None,
        c_prior: dict[str, float] | None = None,
        lambda_reg: float = 0.1,
        ntc_active_tol: float = 0.95,
        **kwargs: Any,
    ) -> _ZonalQP:
        ctx = _ZonalContext.from_inputs(
            prices=observations["prices"],
            n_t=len(batch.timestep_indices),
            zones=zones,
            bus_to_zone=bus_to_zone,
            zone_pairs=zone_pairs,
            flows_per_pair=flows_per_pair,
            ntc_prior=ntc_prior,
            ntc_active_tol=ntc_active_tol,
        )
        return _ZonalQP.assemble(
            ctx=ctx, lambda_reg=lambda_reg, c_prior=c_prior or {},
        )

    def solve(
        self,
        model: _ZonalQP,
        solver_config: SolverConfig | None = None,
    ) -> dict[str, Any]:
        qp = solve_qp_or_raise(
            model=model,
            solver_config=solver_config,
            formulation_name="zonal",
        )
        return _build_solve_result(qp.x, model.layout, model.ctx, qp.objective)

    def residuals(
        self,
        theta: dict[str, float],
        network_data: InvoptNetworkData,
        observations: dict[str, np.ndarray],
        batch: ActiveSetBatch,
    ) -> np.ndarray:
        """Diagnostic per-timestep residual ``‖λ_z − c_z‖``.

        Ignores zonal congestion rents (μ): they are not in ``theta``
        and would not be observable from clearing prices alone.
        """
        prices = observations["prices"]
        n_t = prices.shape[0]
        zones = sorted({
            key.split(":")[1] for key in theta if key.startswith("zone:")
        })
        if not zones:
            return np.zeros(n_t)
        n_zones = len(zones)
        norms = np.zeros(n_t)
        for t in range(n_t):
            sq = 0.0
            for i, z in enumerate(zones):
                if i >= prices.shape[1]:
                    break
                c_z = theta.get(f"zone:{z}:marginal_cost", 0.0)
                sq += (c_z - float(prices[t, i])) ** 2
            norms[t] = float(np.sqrt(sq / max(n_zones, 1)))
        return norms


# ---------------------------------------------------------------------------
# Build-time context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ZonalContext:
    """Pre-resolved zonal layout — pair endpoints, congestion mask, NTC LB."""

    prices: np.ndarray
    zones: list[str]
    pair_names: list[str]
    pair_endpoints: dict[str, tuple[str, str]]
    flows_per_pair: dict[str, np.ndarray]
    congested: dict[tuple[str, int], bool]
    pair_signs_by_zone: dict[str, list[tuple[int, int]]]
    ntc_min_per_pair: np.ndarray   # shape (n_pairs,)
    n_t: int
    n_zones: int
    n_pairs: int

    @classmethod
    def from_inputs(
        cls,
        *,
        prices: np.ndarray,
        n_t: int,
        zones: list[str] | None,
        bus_to_zone: dict[str, str] | None,
        zone_pairs: list[tuple[str, str]] | None,
        flows_per_pair: dict[str, np.ndarray] | None,
        ntc_prior: dict[tuple[str, str], float] | None,
        ntc_active_tol: float,
    ) -> _ZonalContext:
        zones_resolved = _resolve_zones(zones, bus_to_zone)
        pairs = zone_pairs or []
        flows_resolved = flows_per_pair or {}
        ntc_resolved = ntc_prior or dict.fromkeys(pairs, 1000.0)
        pair_names = [f"{a}_{b}" for a, b in pairs]
        pair_endpoints = dict(zip(pair_names, pairs, strict=True))

        congested = _detect_pair_congestion(
            pair_endpoints=pair_endpoints,
            flows_per_pair=flows_resolved,
            ntc_prior=ntc_resolved,
            ntc_active_tol=ntc_active_tol,
            n_t=n_t,
        )
        pair_signs = _pair_signs_per_zone(
            zones=zones_resolved, pair_endpoints=pair_endpoints,
        )
        ntc_min = _compute_ntc_lower_bounds(
            pair_names=pair_names,
            congested=congested,
            flows_per_pair=flows_resolved,
            n_t=n_t,
        )
        return cls(
            prices=prices,
            zones=zones_resolved,
            pair_names=pair_names,
            pair_endpoints=pair_endpoints,
            flows_per_pair=flows_resolved,
            congested=congested,
            pair_signs_by_zone=pair_signs,
            ntc_min_per_pair=ntc_min,
            n_t=n_t,
            n_zones=len(zones_resolved),
            n_pairs=len(pair_names),
        )


def _resolve_zones(
    zones: list[str] | None,
    bus_to_zone: dict[str, str] | None,
) -> list[str]:
    if zones is not None:
        return list(zones)
    if bus_to_zone:
        return sorted(set(bus_to_zone.values()))
    return ["Z1"]


def _detect_pair_congestion(
    *,
    pair_endpoints: dict[str, tuple[str, str]],
    flows_per_pair: dict[str, np.ndarray],
    ntc_prior: dict[tuple[str, str], float],
    ntc_active_tol: float,
    n_t: int,
) -> dict[tuple[str, int], bool]:
    """``(pair, t)`` → True when ``|f_obs| ≥ tol · ntc_prior``."""
    congested: dict[tuple[str, int], bool] = {}
    for pair_name, endpoints in pair_endpoints.items():
        ntc_value = float(ntc_prior.get(endpoints, 0.0))
        observed_flow = flows_per_pair.get(pair_name)
        for t in range(n_t):
            if observed_flow is None or ntc_value <= 0.0:
                congested[pair_name, t] = False
            else:
                congested[pair_name, t] = (
                    abs(float(observed_flow[t])) >= ntc_active_tol * ntc_value
                )
    return congested


def _pair_signs_per_zone(
    *,
    zones: list[str],
    pair_endpoints: dict[str, tuple[str, str]],
) -> dict[str, list[tuple[int, int]]]:
    """Zone → ``[(pair_index, sign), ...]`` for the price-equation row.

    Sign ``+1`` when the zone is the receiving endpoint (importer in
    a ``from → to`` flow), ``-1`` when it is the sending endpoint.
    """
    pair_index = {name: idx for idx, name in enumerate(pair_endpoints)}
    signs: dict[str, list[tuple[int, int]]] = {z: [] for z in zones}
    for pair_name, (z_from, z_to) in pair_endpoints.items():
        idx = pair_index[pair_name]
        if z_from in signs:
            signs[z_from].append((idx, -1))
        if z_to in signs:
            signs[z_to].append((idx, +1))
    return signs


def _compute_ntc_lower_bounds(
    *,
    pair_names: list[str],
    congested: dict[tuple[str, int], bool],
    flows_per_pair: dict[str, np.ndarray],
    n_t: int,
) -> np.ndarray:
    """Per-pair tightest binding observation — used as the ``ntc[pair]`` LB.

    Reduces the per-congested-timestep ``ntc ≥ |f*|`` inequality to a
    single variable bound on ``ntc[pair]``.
    """
    lower = np.zeros(len(pair_names))
    for p_idx, pair_name in enumerate(pair_names):
        flow_arr = flows_per_pair.get(pair_name)
        if flow_arr is None:
            continue
        max_magnitude = 0.0
        for t in range(n_t):
            if congested.get((pair_name, t), False):
                magnitude = abs(float(flow_arr[t]))
                if magnitude > max_magnitude:
                    max_magnitude = magnitude
        lower[p_idx] = max_magnitude
    return lower


# ---------------------------------------------------------------------------
# Variable layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ZonalLayout:
    n_zones: int
    n_pairs: int
    n_t: int

    c_z: int
    lam_z: int
    mu_pair: int
    ntc: int
    total: int

    @classmethod
    def for_shape(cls, *, n_zones: int, n_pairs: int, n_t: int) -> _ZonalLayout:
        c_z = 0
        lam_z = c_z + n_zones
        mu_pair = lam_z + n_zones * n_t
        ntc = mu_pair + n_pairs * n_t
        total = ntc + n_pairs
        return cls(
            n_zones=n_zones, n_pairs=n_pairs, n_t=n_t,
            c_z=c_z, lam_z=lam_z, mu_pair=mu_pair, ntc=ntc, total=total,
        )

    def mu_idx(self, p: int, t: int) -> int:
        return self.mu_pair + p * self.n_t + t


# ---------------------------------------------------------------------------
# QP assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ZonalQP:
    Q: sp.csc_matrix
    q: np.ndarray
    A_eq: sp.csc_matrix
    b_eq: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    layout: _ZonalLayout
    ctx: _ZonalContext

    @classmethod
    def assemble(
        cls,
        *,
        ctx: _ZonalContext,
        lambda_reg: float,
        c_prior: dict[str, float] | None = None,
    ) -> _ZonalQP:
        layout = _ZonalLayout.for_shape(
            n_zones=ctx.n_zones, n_pairs=ctx.n_pairs, n_t=ctx.n_t,
        )
        A_eq, b_eq = _build_price_equation(ctx, layout)
        Q, q = _build_objective(
            ctx, layout, lambda_reg=lambda_reg, c_prior=c_prior or {},
        )
        lb, ub = _build_bounds(ctx, layout)
        return cls(
            Q=Q, q=q, A_eq=A_eq, b_eq=b_eq, lb=lb, ub=ub,
            layout=layout, ctx=ctx,
        )


# ---------------------------------------------------------------------------
# Equality constraints
# ---------------------------------------------------------------------------


def _build_price_equation(
    ctx: _ZonalContext, layout: _ZonalLayout,
) -> tuple[sp.csc_matrix, np.ndarray]:
    """One row per ``(zone, timestep)``: ``λ_z − c_z − Σ sign · μ_pair = 0``.

    Vectorised across the ``(zone, timestep)`` grid; the per-pair
    contributions are flattened across the (pair_index, zone, t)
    triples once via numpy concatenation.
    """
    n_t = ctx.n_t
    n_zones = ctx.n_zones
    n_rows = n_zones * n_t
    if n_rows == 0:
        return sp.csc_matrix((0, layout.total)), np.zeros(0)

    rows_flat = np.arange(n_rows, dtype=int)
    t_range = np.arange(n_t, dtype=int)

    lam_cols = layout.lam_z + rows_flat
    lam_data = np.ones(n_rows)

    c_cols = layout.c_z + np.repeat(np.arange(n_zones, dtype=int), n_t)
    c_data = -np.ones(n_rows)

    mu_rows, mu_cols, mu_data = _zonal_mu_block(ctx, layout, t_range)

    A_eq = sp.csc_matrix(
        (
            np.concatenate([lam_data, c_data, mu_data]),
            (
                np.concatenate([rows_flat, rows_flat, mu_rows]),
                np.concatenate([lam_cols, c_cols, mu_cols]),
            ),
        ),
        shape=(n_rows, layout.total),
    )
    return A_eq, np.zeros(n_rows)


def _zonal_mu_block(
    ctx: _ZonalContext,
    layout: _ZonalLayout,
    t_range: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The ``−Σ sign · μ_pair`` contribution to the price-equation block.

    Each ``(pair → zone)`` incidence yields one entry per timestep,
    expanded by ``np.tile`` over ``t_range`` rather than a Python loop.
    """
    rows_chunks: list[np.ndarray] = []
    cols_chunks: list[np.ndarray] = []
    data_chunks: list[np.ndarray] = []
    for z_idx, zone in enumerate(ctx.zones):
        pair_terms = ctx.pair_signs_by_zone.get(zone, [])
        if not pair_terms:
            continue
        pair_idx = np.array([p for p, _ in pair_terms], dtype=int)
        signs = np.array([s for _, s in pair_terms], dtype=float)
        n_pairs_z = pair_idx.size
        rows_chunks.append(
            z_idx * ctx.n_t + np.tile(t_range, n_pairs_z),
        )
        cols_chunks.append(
            layout.mu_pair + np.repeat(pair_idx, ctx.n_t) * ctx.n_t + np.tile(t_range, n_pairs_z),
        )
        data_chunks.append(-np.repeat(signs, ctx.n_t))

    if not rows_chunks:
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([], dtype=float),
        )
    return (
        np.concatenate(rows_chunks),
        np.concatenate(cols_chunks),
        np.concatenate(data_chunks),
    )


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


def _build_objective(
    ctx: _ZonalContext,
    layout: _ZonalLayout,
    *,
    lambda_reg: float,
    c_prior: dict[str, float],
) -> tuple[sp.csc_matrix, np.ndarray]:
    """Build the QP objective.

    Mathematical form::

        min  (1/T) ‖λ_z − p_obs‖²
           + λ_reg ‖c_z − c_prior‖²
           + μ_REG · ‖μ_pair‖²

    Expanding the Tikhonov anchor:
    ``λ_reg ‖c − c_prior‖² = λ_reg c² − 2 λ_reg c_prior · c + const``.
    The constant ``λ_reg ‖c_prior‖²`` is dropped (doesn't affect
    argmin); the linear term lives in the ``q`` vector.

    Anchoring to a non-zero ``c_prior`` is important on under-
    determined problems (e.g. a zone with a single congested
    interface): the price equation ``λ = c + Σ sign·μ`` then has
    many feasible ``(c, μ)`` decompositions, and a 0-anchor pushes
    ``c`` toward zero — *not* the marginal-cost interpretation the
    Bjørndal & Jørnsten (2001) duality intends. The default
    ``c_prior=0`` preserves the old behaviour for callers that
    haven't supplied a prior.
    """
    n_t = max(ctx.n_t, 1)
    weight_lam = 1.0 / n_t

    diag = np.zeros(layout.total)
    q = np.zeros(layout.total)

    # c_z block — λ_reg · ‖c_z − c_prior‖²
    prior_vec = np.array(
        [float(c_prior.get(zone, 0.0)) for zone in ctx.zones], dtype=float,
    )
    diag[layout.c_z : layout.c_z + ctx.n_zones] = 2.0 * lambda_reg
    q[layout.c_z : layout.c_z + ctx.n_zones] = -2.0 * lambda_reg * prior_vec

    # λ_z block — weight_lam · ‖λ_z − p_obs‖². ``λ_z`` is laid out
    # (zone, t) row-major; we right-pad observed prices when the
    # caller's frame has fewer columns than the model has zones.
    n_lam = ctx.n_zones * ctx.n_t
    diag[layout.lam_z : layout.lam_z + n_lam] = 2.0 * weight_lam
    if n_lam > 0:
        observed = np.zeros((ctx.n_zones, ctx.n_t))
        n_cols = min(ctx.prices.shape[1], ctx.n_zones)
        observed[:n_cols] = ctx.prices.T[:n_cols] if ctx.prices.size else observed[:n_cols]
        q[layout.lam_z : layout.lam_z + n_lam] = -2.0 * weight_lam * observed.ravel()

    # μ_pair block — small Tikhonov anchor on every shadow-price slot.
    diag[layout.mu_pair : layout.mu_pair + ctx.n_pairs * ctx.n_t] = 2.0 * ZONAL_MU_REGULARISER

    return sp.diags(diag, format="csc"), q


# ---------------------------------------------------------------------------
# Variable bounds (carries the complementarity + NTC LB conditions)
# ---------------------------------------------------------------------------


def _build_bounds(
    ctx: _ZonalContext, layout: _ZonalLayout,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode complementarity (μ=0 off the active set) and NTC LB as bounds.

    The complementarity mask is built as a single ``(n_pairs, n_t)``
    bool array from the precomputed ``ctx.congested`` dict, then
    flattened into the ``μ_pair`` slot range.
    """
    lb = np.full(layout.total, -np.inf)
    ub = np.full(layout.total, np.inf)

    lb[layout.c_z : layout.c_z + ctx.n_zones] = 0.0

    # μ_pair: 0 on uncongested cells (complementarity), free elsewhere.
    if ctx.n_pairs and ctx.n_t:
        congested_mask = np.array(
            [
                [ctx.congested.get((pair_name, t), False) for t in range(ctx.n_t)]
                for pair_name in ctx.pair_names
            ],
            dtype=bool,
        ).ravel()
        n_mu = ctx.n_pairs * ctx.n_t
        mu_lb = np.where(congested_mask, -np.inf, 0.0)
        mu_ub = np.where(congested_mask,  np.inf, 0.0)
        lb[layout.mu_pair : layout.mu_pair + n_mu] = mu_lb
        ub[layout.mu_pair : layout.mu_pair + n_mu] = mu_ub

    # NTC ≥ max binding observation per pair.
    lb[layout.ntc : layout.ntc + ctx.n_pairs] = ctx.ntc_min_per_pair

    return lb, ub


# ---------------------------------------------------------------------------
# Result construction
# ---------------------------------------------------------------------------


def _build_solve_result(
    x: np.ndarray,
    layout: _ZonalLayout,
    ctx: _ZonalContext,
    objective: float,
) -> dict[str, Any]:
    """Pull recovered θ̂ + price residuals out of the optimised QP vector."""
    theta: dict[str, float] = {}
    for z_idx, zone in enumerate(ctx.zones):
        theta[f"zone:{zone}:marginal_cost"] = float(x[layout.c_z + z_idx])

    for p_idx, pair_name in enumerate(ctx.pair_names):
        theta[f"ntc:{pair_name}"] = float(x[layout.ntc + p_idx])
        mu_block = x[
            layout.mu_idx(p_idx, 0) : layout.mu_idx(p_idx, 0) + ctx.n_t
        ]
        non_trivial = mu_block[np.abs(mu_block) > 1e-9]
        if non_trivial.size > 0:
            theta[f"ntc_shadow_mean:{pair_name}"] = float(np.mean(non_trivial))

    lam_block = x[
        layout.lam_z : layout.lam_z + ctx.n_zones * ctx.n_t
    ].reshape(ctx.n_zones, ctx.n_t)
    p_obs_per_zone = (
        ctx.prices.T if ctx.prices.shape == (ctx.n_t, ctx.n_zones)
        else _pad_or_truncate(ctx.prices.T, ctx.n_zones, ctx.n_t)
    )
    diff = lam_block - p_obs_per_zone
    sse_per_t = np.einsum("zt,zt->t", diff, diff)
    residuals = (
        np.sqrt(sse_per_t / max(ctx.n_zones, 1))
        if ctx.n_t > 0
        else np.zeros(0)
    )

    return {
        "theta": theta,
        "residuals": residuals,
        "status": "optimal",
        "objective": objective,
    }


def _pad_or_truncate(arr: np.ndarray, n_zones: int, n_t: int) -> np.ndarray:
    """Right-pad / top-pad ``arr`` to ``(n_zones, n_t)`` with zeros."""
    out = np.zeros((n_zones, n_t))
    rows = min(arr.shape[0], n_zones)
    cols = min(arr.shape[1], n_t)
    out[:rows, :cols] = arr[:rows, :cols]
    return out


__all__ = ["ZonalFormulation"]
