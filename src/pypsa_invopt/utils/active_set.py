"""Active-set detection and clustering for ASTB.

The Active-Set Temporal Batching pipeline:

1. **Detect** which constraints bind at each timestep
   (:func:`detect_active_sets_temporal`).
2. **Cluster** timesteps with identical active-set patterns into a
   single :class:`ActiveSetBatch` (:func:`cluster_active_sets`).
3. **Solve** one model per batch (the inverse-OPF formulations) — the
   reduction from ``T`` independent solves to ``K ≪ T`` batched solves
   is the practical performance win.

For European national grids ``K`` is typically 20-150 over a full
year of hourly observations.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ActiveSet:
    """Snapshot of which constraints bind at a single timestep.

    Attributes:
        congested_lines: Lines whose flow magnitude meets the thermal
            limit (within tolerance).
        maxed_generators: Generators dispatched at their upper bound.
        min_bound_generators: Generators dispatched at their lower bound.
    """

    congested_lines: frozenset[str] = field(default_factory=frozenset)
    maxed_generators: frozenset[str] = field(default_factory=frozenset)
    min_bound_generators: frozenset[str] = field(default_factory=frozenset)

    def pattern_key(self) -> tuple:
        """Hashable, order-insensitive identifier of this active set."""
        return (
            tuple(sorted(self.congested_lines)),
            tuple(sorted(self.maxed_generators)),
            tuple(sorted(self.min_bound_generators)),
        )


@dataclass
class ActiveSetBatch:
    """Timesteps that share a single :class:`ActiveSet` pattern.

    Attributes:
        pattern: The shared active set.
        timestep_indices: Original observation indices in this batch.
        cluster_id: 0-based integer label assigned by the clusterer.
    """

    pattern: ActiveSet
    timestep_indices: list[int]
    cluster_id: int


def detect_active_set(
    *,
    flows: np.ndarray,
    flow_limits: np.ndarray,
    dispatch: np.ndarray,
    gen_max: np.ndarray,
    gen_min: np.ndarray,
    line_names: list[str],
    gen_names: list[str],
    eps: float = 1e-3,
) -> ActiveSet:
    """Identify the binding-constraint set for a single observation.

    Args:
        flows: Observed flows, shape ``(n_lines,)``.
        flow_limits: Line ``s_nom`` values, shape ``(n_lines,)``.
        dispatch: Observed generator dispatch, shape ``(n_gens,)``.
        gen_max: Generator upper bounds, shape ``(n_gens,)``.
        gen_min: Generator lower bounds, shape ``(n_gens,)``.
        line_names: Ordered line names.
        gen_names: Ordered generator names.
        eps: Tolerance for treating a constraint as binding.
    """
    return ActiveSet(
        congested_lines=frozenset(
            line_names[i]
            for i in range(len(line_names))
            if abs(flows[i]) >= flow_limits[i] - eps
        ),
        maxed_generators=frozenset(
            gen_names[i]
            for i in range(len(gen_names))
            if dispatch[i] >= gen_max[i] - eps
        ),
        min_bound_generators=frozenset(
            gen_names[i]
            for i in range(len(gen_names))
            if dispatch[i] <= gen_min[i] + eps
        ),
    )


def detect_active_sets_temporal(
    *,
    flows: np.ndarray,
    flow_limits: np.ndarray,
    dispatch: np.ndarray,
    gen_max: np.ndarray,
    gen_min: np.ndarray,
    line_names: list[str],
    gen_names: list[str],
    eps: float = 1e-3,
    mu_lines: np.ndarray | None = None,
    mu_gens_upper: np.ndarray | None = None,
    mu_gens_lower: np.ndarray | None = None,
    mu_tol: float = 1e-6,
) -> list[ActiveSet]:
    """Detect the active set at every timestep.

    Fully vectorised: the three ``(T, n_*)`` boolean masks are produced
    with one numpy comparison each, then each row is condensed into a
    :class:`frozenset` for the per-timestep :class:`ActiveSet` record.

    ``gen_max`` and ``gen_min`` can be supplied as 1-D arrays (constant
    bounds) or 2-D arrays ``(T, n_gens)`` (time-varying bounds).

    **Shadow-price-direct detection** (added 2026-05-14):
    when ``mu_lines`` / ``mu_gens_upper`` / ``mu_gens_lower`` are
    provided (shape ``(T, n_*)``), they take precedence over the
    tolerance-based detection. PyPSA's solved networks expose these
    directly as ``network.lines_t.mu_upper`` etc., so passing them
    in eliminates the tolerance heuristic — a line is congested iff
    its reported dual exceeds ``mu_tol``. This is the canonical
    KKT-correct definition of "active set" and removes the false
    positives that creep in when observed flow happens to be within
    ``eps`` of ``s_nom`` due to numerical noise.
    """
    n_t = flows.shape[0]
    if n_t == 0:
        return []

    flow_limits_2d = np.broadcast_to(flow_limits, flows.shape)
    gen_max_2d = np.broadcast_to(gen_max, dispatch.shape)
    gen_min_2d = np.broadcast_to(gen_min, dispatch.shape)

    # Shadow-price-direct detection wins when present; otherwise fall
    # back to the tolerance heuristic on observed flow / dispatch.
    if mu_lines is not None:
        congested_mask = np.abs(mu_lines) > mu_tol
    else:
        congested_mask = np.abs(flows) >= flow_limits_2d - eps
    if mu_gens_upper is not None:
        maxed_mask = mu_gens_upper > mu_tol
    else:
        maxed_mask = dispatch >= gen_max_2d - eps
    min_mask = mu_gens_lower > mu_tol if mu_gens_lower is not None else dispatch <= gen_min_2d + eps

    # Outage / availability filter: when a generator's per-snapshot
    # upper bound is effectively zero (``p_max ≤ eps``), it's offline
    # at that (g, t) — either an unplanned outage or a commitment
    # OFF-hour. Such snapshots carry no cost information for that
    # generator: the KKT row would just say ``0 - c + λ + ν = 0`` with
    # ν free, letting any cost satisfy the equation trivially. Mark
    # the (g, t) as *neither* maxed *nor* min-bound so the KKT loop
    # silently drops the row. This is the Liang-Dvorkin (2023)
    # Algorithm 1 "identify free generators" filter, generalised
    # per-snapshot.
    if gen_max_2d.ndim == 2:
        offline_mask = gen_max_2d <= eps
    else:
        offline_mask = np.broadcast_to(gen_max <= eps, dispatch.shape)
    maxed_mask = maxed_mask & ~offline_mask
    min_mask = min_mask & ~offline_mask

    line_arr = np.asarray(line_names, dtype=object)
    gen_arr = np.asarray(gen_names, dtype=object)

    return [
        ActiveSet(
            congested_lines=frozenset(line_arr[congested_mask[t]].tolist()),
            maxed_generators=frozenset(gen_arr[maxed_mask[t]].tolist()),
            min_bound_generators=frozenset(gen_arr[min_mask[t]].tolist()),
        )
        for t in range(n_t)
    ]


def cluster_active_sets(active_sets: list[ActiveSet]) -> list[ActiveSetBatch]:
    """Group timesteps that share the same :class:`ActiveSet` pattern.

    Active-Set Temporal Batching (ASTB). Returned batches are sorted
    largest first so the dominant active set is solved first.

    **Scope note.** ASTB is the right compression for *pure thermal*
    inverse-OPF problems where snapshots are independent (no storage,
    no global cap, no controllable link). For modern grids with any of
    those, :func:`pypsa_invopt.calibrate` collapses the batches to a
    single union pattern (see ``_maybe_collapse_for_intertemporal``);
    on such grids ASTB is effectively a no-op. The compression buys
    real wall-clock time on IEEE-test-system-style benchmarks and on
    sub-grids without intertemporal couplings.
    """
    pattern_indices: dict[tuple, list[int]] = {}
    pattern_seen: dict[tuple, ActiveSet] = {}

    for t, aset in enumerate(active_sets):
        key = aset.pattern_key()
        if key not in pattern_indices:
            pattern_indices[key] = []
            pattern_seen[key] = aset
        pattern_indices[key].append(t)

    batches = [
        ActiveSetBatch(
            pattern=pattern_seen[key],
            timestep_indices=indices,
            cluster_id=cluster_id,
        )
        for cluster_id, (key, indices) in enumerate(pattern_indices.items())
    ]
    batches.sort(key=lambda b: len(b.timestep_indices), reverse=True)
    return batches


__all__ = [
    "ActiveSet",
    "ActiveSetBatch",
    "cluster_active_sets",
    "detect_active_sets_temporal",
]
