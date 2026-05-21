"""Inverse-OPF calibration orchestrator.

Reads a ``pypsa.Network`` and an observations DataFrame, runs the
configured formulation through the Active-Set Temporal Batching (ASTB)
loop, optionally post-processes line parameters, and returns an
:class:`InverseResult`.

Batches are aggregated by the **inverse-variance (BLUE) estimator**:
each batch's per-parameter estimate ``θ_k`` is weighted by
``T_k / σ²_k`` (snapshot count over batch residual variance). This
reduces to the frequency-weighted mean when all batches have equal
residual variance and down-weights noisy batches otherwise.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from pypsa_invopt._constants import (
    BATCH_VARIANCE_FLOOR,
)
from pypsa_invopt.exceptions import InvoptInputError
from pypsa_invopt.network import InvoptNetworkData, read_network
from pypsa_invopt.results import InverseResult
from pypsa_invopt.solvers import SolverConfig
from pypsa_invopt.utils.active_set import (
    ActiveSet,
    ActiveSetBatch,
    cluster_active_sets,
    detect_active_sets_temporal,
)

if TYPE_CHECKING:
    import pypsa

    from pypsa_invopt.formulations.base import InverseFormulation

logger = logging.getLogger(__name__)

FormulationType = Literal["noiseless", "noisy", "zonal"]

_VALID_FORMULATIONS: frozenset[str] = frozenset(("noiseless", "noisy", "zonal"))

@dataclass(frozen=True)
class _BatchSolution:
    """One batch's contribution to the cross-batch aggregation."""

    pattern: ActiveSet
    theta: dict[str, float]
    weight: float  # T_k / σ²_k (used by the BLUE aggregator)
    status: str
    residuals: np.ndarray


def calibrate(
    network: pypsa.Network,
    observations: pd.DataFrame,
    *,
    formulation: FormulationType = "noisy",
    solver: str = "highs",
    active_set_tol: float = 1e-3,
    verbose: bool = False,
    recover_line_params: bool = False,
    **formulation_kwargs: Any,
) -> InverseResult:
    """Calibrate network parameters via inverse OPF.

    Args:
        network: A populated ``pypsa.Network``.
        observations: DataFrame with a DatetimeIndex and columns
            ``price_<bus>`` (or ``price_<zone>`` for zonal). May also
            include ``flow_<line>`` and ``dispatch_<gen>`` columns,
            which enable active-set detection and the optional
            line-parameter post-step.
        formulation: ``'noiseless'``, ``'noisy'`` (default; canonical
            Liang-Dvorkin 2023 single-level KKT-QP), or ``'zonal'``.
        solver: Solver name (``'highs'``, ``'gurobi'``, ``'ipopt'``).
        active_set_tol: Tolerance for declaring a constraint binding.
        verbose: If ``True``, print solver output.
        recover_line_params: If ``True``, after the main calibration
            run an additional post-step to recover ``s_nom`` (closed
            form) and ``x`` (NLP) for lines that were observed at their
            limit. Requires IPOPT for the susceptance NLP.
        **formulation_kwargs: Forwarded to the formulation's
            ``build_model`` (e.g. ``lambda_physics``, ``obs_sigma``,
            ``zones``, ``flows_per_pair``).

    Returns:
        :class:`InverseResult` with recovered parameters, KKT residuals
        and diagnostics.

    Raises:
        InvoptInputError: If ``formulation`` is unknown or the
            observations DataFrame lacks required columns.
    """
    if formulation not in _VALID_FORMULATIONS:
        raise InvoptInputError(
            f"Unknown formulation '{formulation}'. "
            f"Choose from: {sorted(_VALID_FORMULATIONS)}"
        )

    t_start = time.perf_counter()

    net_data = read_network(network)
    obs_arrays = _extract_observations(
        net_data, observations, formulation,
        bus_to_zone=formulation_kwargs.get("bus_to_zone"),
    )
    n_timesteps = obs_arrays["prices"].shape[0]

    flows = obs_arrays.get("flows")
    dispatch = obs_arrays.get("dispatch")
    active_sets = _detect_active_sets(
        net_data=net_data,
        flows=flows,
        dispatch=dispatch,
        n_timesteps=n_timesteps,
        tol=active_set_tol,
        mu_lines=obs_arrays.get("mu_lines"),
        mu_gens_upper=obs_arrays.get("mu_gens_upper"),
        mu_gens_lower=obs_arrays.get("mu_gens_lower"),
    )
    batches = cluster_active_sets(active_sets)
    batches = _maybe_collapse_for_intertemporal(
        batches, obs_arrays, net_data, n_timesteps,
    )

    solver_config = SolverConfig(solver=solver, verbose=verbose)
    form = _build_formulation(formulation)

    batch_solutions, kkt_residuals, statuses, batch_active_sets = _solve_batches(
        form=form,
        batches=batches,
        net_data=net_data,
        obs_arrays=obs_arrays,
        solver_config=solver_config,
        formulation_kwargs=formulation_kwargs,
    )

    theta_hat = _blue_aggregate(batch_solutions)
    rmse = (
        float(np.sqrt(np.mean(kkt_residuals ** 2)))
        if kkt_residuals.size > 0
        else float("inf")
    )

    if recover_line_params and flows is not None:
        _recover_line_parameters(
            theta_hat=theta_hat,
            network=network,
            net_data=net_data,
            batches=batches,
            flows=flows,
            dispatch=dispatch,
            n_timesteps=n_timesteps,
            verbose=verbose,
        )

    return InverseResult(
        theta_hat=theta_hat,
        rmse=rmse,
        kkt_residuals=kkt_residuals,
        active_set=batch_active_sets,
        solver_status=_aggregate_status(statuses),
        formulation=formulation,
        n_active_sets=len(batches),
        wall_time_s=time.perf_counter() - t_start,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_formulation(name: FormulationType) -> InverseFormulation:
    """Instantiate the formulation class for ``name``."""
    from pypsa_invopt.formulations import (
        NoiselessFormulation,
        NoisyFormulation,
        ZonalFormulation,
    )

    if name == "noiseless":
        return NoiselessFormulation()
    if name == "noisy":
        return NoisyFormulation()
    if name == "zonal":
        return ZonalFormulation()
    # _VALID_FORMULATIONS gate above guarantees this is unreachable
    raise ValueError(f"Unknown formulation: {name}")


def _maybe_collapse_for_intertemporal(
    batches: list[ActiveSetBatch],
    obs_arrays: dict[str, np.ndarray],
    net_data: InvoptNetworkData,
    n_timesteps: int,
) -> list[ActiveSetBatch]:
    """If the network has cross-snapshot couplings, fold every ASTB
    batch into one single-batch run.

    Triggered when the network includes:

    * **storage_units / stores** — cyclic SOC duals link snapshot ``t``
      to ``(t+1) mod n_t``, breaking ASTB's snapshot-independence
      assumption.
    * **links** — currently routed through the single-batch path.
    * **global_constraints** (CO₂ / RES caps) — a single dual sums
      emissions across all snapshots; solving per-batch would fragment
      ``μ_co2`` into K independent estimates and break the recovery.

    Pure generator + line networks fall through unchanged and keep
    multi-batch ASTB compression.
    """
    has_intertemporal = (
        ("storage" in obs_arrays and net_data.storage_units)
        or ("store" in obs_arrays and net_data.stores)
        or ("link" in obs_arrays and net_data.links)
        or bool(net_data.global_constraints)
    )
    if not has_intertemporal:
        return batches
    union_pattern = batches[0].pattern
    for b in batches[1:]:
        union_pattern = type(union_pattern)(
            congested_lines=frozenset(union_pattern.congested_lines)
                              | frozenset(b.pattern.congested_lines),
            maxed_generators=frozenset(union_pattern.maxed_generators)
                              | frozenset(b.pattern.maxed_generators),
            min_bound_generators=frozenset(union_pattern.min_bound_generators)
                              | frozenset(b.pattern.min_bound_generators),
        )
    return [ActiveSetBatch(
        pattern=union_pattern,
        timestep_indices=list(range(n_timesteps)),
        cluster_id=0,
    )]


def _detect_active_sets(
    *,
    net_data: InvoptNetworkData,
    flows: np.ndarray | None,
    dispatch: np.ndarray | None,
    n_timesteps: int,
    tol: float,
    mu_lines: np.ndarray | None = None,
    mu_gens_upper: np.ndarray | None = None,
    mu_gens_lower: np.ndarray | None = None,
) -> list[ActiveSet]:
    """Return the per-timestep active set; empty sets when no flow data."""
    if flows is None or dispatch is None:
        return [ActiveSet() for _ in range(n_timesteps)]

    flow_limits = np.array([net_data.line_s_nom[ln] for ln in net_data.lines])
    # Use time-varying p_max_pu/p_min_pu when present (renewables, run-of-river
    # hydro, demand-response — the dominant LMP drivers in European systems).
    # Falls back to the static p_nom × p_*_pu when there's no time-series.
    if net_data.gen_p_max_t is not None:
        gen_max = net_data.gen_p_max_t
    else:
        gen_max = np.array([net_data.gen_p_max[g] for g in net_data.generators])
    if net_data.gen_p_min_t is not None:
        gen_min = net_data.gen_p_min_t
    else:
        gen_min = np.array([net_data.gen_p_min[g] for g in net_data.generators])
    return detect_active_sets_temporal(
        flows=flows,
        flow_limits=flow_limits,
        dispatch=dispatch,
        gen_max=gen_max,
        gen_min=gen_min,
        line_names=net_data.lines,
        gen_names=net_data.generators,
        eps=tol,
        # When the caller exposes PyPSA's reported duals
        # (``lines_t.mu_upper`` etc.) the detector uses them directly
        # instead of the |flow| ≥ s_nom − ε heuristic — KKT-correct
        # active-set identification with no tolerance tuning needed.
        mu_lines=mu_lines,
        mu_gens_upper=mu_gens_upper,
        mu_gens_lower=mu_gens_lower,
    )


def _solve_batches(
    *,
    form: InverseFormulation,
    batches: list[ActiveSetBatch],
    net_data: InvoptNetworkData,
    obs_arrays: dict[str, np.ndarray],
    solver_config: SolverConfig,
    formulation_kwargs: dict[str, Any],
) -> tuple[
    list[_BatchSolution],
    np.ndarray,
    list[str],
    dict[int, dict[str, list[str]]],
]:
    """Solve each ASTB batch sequentially.

    Returns the per-batch :class:`_BatchSolution` records (each carrying
    its inverse-variance BLUE weight), the concatenated KKT residual
    series, the per-batch solver statuses, and a per-timestep
    active-set diagnostic.
    """
    solutions: list[_BatchSolution] = []
    residual_chunks: list[np.ndarray] = []
    statuses: list[str] = []
    batch_active_sets: dict[int, dict[str, list[str]]] = {}

    for batch in batches:
        outcome = _solve_one_batch(
            form=form,
            batch=batch,
            net_data=net_data,
            obs_arrays=obs_arrays,
            solver_config=solver_config,
            formulation_kwargs=formulation_kwargs,
        )
        statuses.append(outcome.status)
        if outcome.solution is None:
            continue

        solutions.append(outcome.solution)
        residual_chunks.append(outcome.solution.residuals)
        _record_active_set(batch_active_sets, batch)

    kkt_residuals = (
        np.concatenate(residual_chunks) if residual_chunks else np.array([])
    )
    return solutions, kkt_residuals, statuses, batch_active_sets


@dataclass(frozen=True)
class _BatchOutcome:
    """Result of attempting one batch solve.

    Carries the recovered ``_BatchSolution`` on success, or just the
    status string on build/solve failure so the caller can record it
    without branching on exception types.
    """

    solution: _BatchSolution | None
    status: str


def _solve_one_batch(
    *,
    form: InverseFormulation,
    batch: ActiveSetBatch,
    net_data: InvoptNetworkData,
    obs_arrays: dict[str, np.ndarray],
    solver_config: SolverConfig,
    formulation_kwargs: dict[str, Any],
) -> _BatchOutcome:
    """Build and solve a single ASTB batch.

    Failures during build or solve are caught and turned into a
    descriptive status string — they do not propagate.
    """
    batch_obs = _slice_observations(obs_arrays, batch)

    try:
        model = form.build_model(net_data, batch, batch_obs, **formulation_kwargs)
    except (RuntimeError, ValueError, KeyError) as exc:
        logger.warning("Batch %d build failed: %s", batch.cluster_id, exc)
        return _BatchOutcome(solution=None, status=f"error: {exc}")

    try:
        result = form.solve(model, solver_config)
    except (RuntimeError, ValueError, KeyError) as exc:
        logger.warning("Batch %d solve failed: %s", batch.cluster_id, exc)
        return _BatchOutcome(solution=None, status=f"error: {exc}")

    residuals = result["residuals"]
    solution = _BatchSolution(
        pattern=batch.pattern,
        theta=dict(result["theta"]),
        weight=_blue_weight(
            n_timesteps=len(batch.timestep_indices),
            residuals=residuals,
        ),
        status=result["status"],
        residuals=residuals,
    )
    return _BatchOutcome(solution=solution, status=result["status"])


def _record_active_set(
    bookkeeping: dict[int, dict[str, list[str]]],
    batch: ActiveSetBatch,
) -> None:
    """Copy the batch's pattern into the per-timestep diagnostic map."""
    pattern_record = {
        "congested_lines": list(batch.pattern.congested_lines),
        "maxed_generators": list(batch.pattern.maxed_generators),
    }
    for t in batch.timestep_indices:
        bookkeeping[t] = pattern_record


def _blue_weight(*, n_timesteps: int, residuals: np.ndarray) -> float:
    """BLUE aggregation weight for one batch.

    Two effects combine:

    * ``T_k`` (batch size) — larger batches carry more information.
    * ``1 / σ²_k`` (inverse residual variance) — tighter-fitting batches
      are more reliable estimates of θ.

    The variance is floored at :data:`BATCH_VARIANCE_FLOOR` so a
    near-perfect batch can not blow up the denominator.
    """
    if n_timesteps <= 0:
        return 0.0
    variance = float(np.mean(residuals ** 2)) if residuals.size > 0 else 0.0
    return n_timesteps / max(variance, BATCH_VARIANCE_FLOOR)


def _blue_aggregate(solutions: list[_BatchSolution]) -> dict[str, float]:
    """Combine per-batch ``θ_k`` by inverse-variance weighting.

    For each parameter key, returns ``Σ w_k θ_k / Σ w_k`` over the
    batches where that parameter is **identifiable** — i.e. where the
    associated generator is *interior* (neither at p_max nor at p_min)
    in that batch's active set. Bound-binding batches are gauge-
    degenerate for the bound generator's cost (any ``c ≤ λ`` satisfies
    KKT with the bound dual absorbing the slack), so including them
    dilutes the truth-recovery from the interior batches.

    Falls back to all-batches aggregation for parameters that no batch
    identifies (so callers always see a recovered value, just one with
    no data-driven information — the regulariser pull).
    """
    weighted_sum: dict[str, float] = {}
    weight_total: dict[str, float] = {}
    seen_in_any: dict[str, bool] = {}

    for sol in solutions:
        pattern = sol.pattern
        bound_gens = pattern.maxed_generators | pattern.min_bound_generators
        for key, val in sol.theta.items():
            seen_in_any[key] = True
            # gen:<name>:* keys: skip if <name> is bound-binding here
            if key.startswith("gen:"):
                gen_name = key.split(":", 2)[1]
                if gen_name in bound_gens:
                    continue
            weighted_sum[key] = weighted_sum.get(key, 0.0) + sol.weight * float(val)
            weight_total[key] = weight_total.get(key, 0.0) + sol.weight

    # Fallback for parameters that were bound-binding in every batch:
    # use the un-filtered BLUE average (gauge-degenerate, but better
    # than dropping the key entirely).
    fallback_sum: dict[str, float] = {}
    fallback_total: dict[str, float] = {}
    for sol in solutions:
        for key, val in sol.theta.items():
            if weight_total.get(key, 0.0) > 0.0:
                continue
            fallback_sum[key] = fallback_sum.get(key, 0.0) + sol.weight * float(val)
            fallback_total[key] = fallback_total.get(key, 0.0) + sol.weight

    result: dict[str, float] = {}
    for key in seen_in_any:
        if weight_total.get(key, 0.0) > 0.0:
            result[key] = weighted_sum[key] / weight_total[key]
        elif fallback_total.get(key, 0.0) > 0.0:
            result[key] = fallback_sum[key] / fallback_total[key]
    return result


def _aggregate_status(statuses: list[str]) -> str:
    """Reduce a list of per-batch solver statuses to a single label."""
    if not statuses:
        return "unknown"
    optimal = {"optimal", "globallyoptimal", "locallyoptimal"}
    if all(s in optimal for s in statuses):
        return "optimal"
    if any("error" in s for s in statuses):
        return "partial"
    return statuses[0]


def _recover_line_parameters(
    *,
    theta_hat: dict[str, float],
    network: pypsa.Network,
    net_data: InvoptNetworkData,
    batches: list[ActiveSetBatch],
    flows: np.ndarray,
    dispatch: np.ndarray | None,
    n_timesteps: int,
    verbose: bool,
) -> None:
    """Add ``line:<name>:s_nom`` (closed form) and ``line:<name>:x`` (NLP)
    entries to ``theta_hat`` in-place. Failures during the NLP are
    logged and silently skipped — line-parameter recovery is opt-in
    and is not allowed to derail the main calibration result.
    """
    from pypsa_invopt.utils.susceptance import (
        recover_flow_limits,
        recover_line_susceptances,
    )

    congested_per_line: dict[str, list[int]] = {ln: [] for ln in net_data.lines}
    for batch in batches:
        for ln in batch.pattern.congested_lines:
            if ln in congested_per_line:
                congested_per_line[ln].extend(batch.timestep_indices)

    for ln, s_val in recover_flow_limits(
        network_data=net_data,
        flows=flows,
        congested_per_line=congested_per_line,
    ).items():
        theta_hat[f"line:{ln}:s_nom"] = s_val

    try:
        loads_per_bus = _compute_loads_per_bus(network, net_data, n_timesteps)
    except (KeyError, AttributeError, ValueError):
        loads_per_bus = None

    try:
        x_hat = recover_line_susceptances(
            network_data=net_data,
            flows=flows,
            dispatch=dispatch,
            loads_per_bus=loads_per_bus,
            solver_config=SolverConfig(solver="ipopt", verbose=verbose),
        )
    except Exception as exc:
        logger.info("Susceptance recovery skipped: %s", exc)
        return

    for ln, x_val in x_hat.items():
        theta_hat[f"line:{ln}:x"] = x_val


def _compute_loads_per_bus(
    network: pypsa.Network,
    net_data: InvoptNetworkData,
    n_timesteps: int,
) -> np.ndarray:
    """Aggregate per-bus demand for the susceptance NLP.

    Time-varying ``network.loads_t.p_set`` is used where available, and
    static ``network.loads['p_set']`` fills in the rest.
    """
    n_buses = len(net_data.buses)
    bus_idx = net_data.bus_index
    loads_per_bus = np.zeros((n_timesteps, n_buses))

    loads_t = getattr(network, "loads_t", None)
    p_set_t = getattr(loads_t, "p_set", None) if loads_t is not None else None
    have_dynamic = p_set_t is not None and not p_set_t.empty

    if have_dynamic:
        for load_name in p_set_t.columns:
            bus = network.loads.at[load_name, "bus"]
            if bus not in bus_idx:
                continue
            series = p_set_t[load_name].to_numpy(dtype=float)
            n_avail = min(len(series), n_timesteps)
            loads_per_bus[:n_avail, bus_idx[bus]] += series[:n_avail]

    covered = set(p_set_t.columns) if have_dynamic else set()
    for load_name in network.loads.index:
        if load_name in covered:
            continue
        bus = network.loads.at[load_name, "bus"]
        if bus not in bus_idx:
            continue
        loads_per_bus[:, bus_idx[bus]] += float(network.loads.at[load_name, "p_set"])

    return loads_per_bus


def _extract_prices(
    net_data: InvoptNetworkData,
    observations: pd.DataFrame,
    formulation: str,
    *,
    bus_to_zone: dict[str, str] | None,
) -> np.ndarray:
    """Build the (T, n_buses) or (T, n_zones) price matrix.

    Zonal calibrations average constituent-bus LMPs into one column
    per zone (single-bus-per-zone default; European zonal calibrations
    pass ``bus_to_zone`` to do the rollup).
    """
    price_cols = [
        f"price_{bus}" for bus in net_data.buses
        if f"price_{bus}" in observations.columns
    ]
    if not price_cols and formulation == "zonal":
        # Back-compat: legacy zonal calls that pass price_<zone_label>
        # columns directly (e.g. when zone labels don't match bus names).
        price_cols = [
            c for c in observations.columns if c.startswith("price_")
        ]
    if not price_cols:
        raise InvoptInputError(
            "No price columns found in observations. Expected "
            "'price_<bus_name>' columns. Available: "
            f"{list(observations.columns)}"
        )
    bus_prices = observations[price_cols].to_numpy(dtype=float)
    if not (formulation == "zonal" and bus_to_zone):
        return bus_prices
    # Roll bus → zone by mean — zone order matches the QP's `zones` list.
    zones = sorted(set(bus_to_zone.values()))
    zone_prices = np.zeros((bus_prices.shape[0], len(zones)))
    for i, zone in enumerate(zones):
        cols = [
            idx for idx, bus in enumerate(net_data.buses)
            if bus_to_zone.get(bus) == zone
            and f"price_{bus}" in observations.columns
        ]
        if cols:
            zone_prices[:, i] = bus_prices[:, cols].mean(axis=1)
    return zone_prices


def _columns_complete(
    prefix: str, names: list[str], df: pd.DataFrame,
) -> list[str] | None:
    """Return the prefixed column list iff *every* name has a column.

    Returning ``None`` when even one is missing keeps the calibrator
    in "skip this block" mode for partial datasets — the storage /
    link / store KKT blocks never get built unless every component
    in that family was observed.
    """
    cols = [f"{prefix}{n}" for n in names if f"{prefix}{n}" in df.columns]
    return cols if len(cols) == len(names) else None


def _extract_observations(
    net_data: InvoptNetworkData,
    observations: pd.DataFrame,
    formulation: str,
    *,
    bus_to_zone: dict[str, str] | None = None,
) -> dict[str, np.ndarray]:
    """Pull the typed numpy arrays the formulation needs out of the
    user-supplied DataFrame.

    Delegates to per-component helpers — see :func:`_extract_prices`,
    :func:`_columns_complete`, and the per-block extractors below.
    """
    result: dict[str, np.ndarray] = {
        "prices": _extract_prices(
            net_data, observations, formulation, bus_to_zone=bus_to_zone,
        ),
    }

    # Generators: flows + dispatch (and optional PyPSA-reported μ duals).
    flow_cols = _columns_complete("flow_", list(net_data.lines), observations)
    if flow_cols:
        result["flows"] = observations[flow_cols].to_numpy(dtype=float)
    dispatch_cols = _columns_complete("dispatch_", list(net_data.generators), observations)
    if dispatch_cols:
        result["dispatch"] = observations[dispatch_cols].to_numpy(dtype=float)
    mu_lines = _columns_complete("mu_line_", list(net_data.lines), observations)
    if mu_lines:
        result["mu_lines"] = observations[mu_lines].to_numpy(dtype=float)
    mu_gmax = _columns_complete("mu_gen_max_", list(net_data.generators), observations)
    if mu_gmax:
        result["mu_gens_upper"] = observations[mu_gmax].to_numpy(dtype=float)
    mu_gmin = _columns_complete("mu_gen_min_", list(net_data.generators), observations)
    if mu_gmin:
        result["mu_gens_lower"] = observations[mu_gmin].to_numpy(dtype=float)

    # Storage: dispatch + store columns required; SOC optional.
    storage = list(net_data.storage_units)
    if storage:
        disp = _columns_complete("storage_dispatch_", storage, observations)
        store = _columns_complete("storage_store_", storage, observations)
        if disp and store:
            bundle: dict[str, np.ndarray] = {
                "p_dispatch": observations[disp].to_numpy(dtype=float),
                "p_store":    observations[store].to_numpy(dtype=float),
            }
            soc = _columns_complete("storage_soc_", storage, observations)
            if soc:
                bundle["soc"] = observations[soc].to_numpy(dtype=float)
            result["storage"] = bundle  # type: ignore[assignment]

    # Link (HVDC, P2H, P2G).
    if net_data.links:
        link_cols = _columns_complete(
            "link_dispatch_", list(net_data.links), observations,
        )
        if link_cols:
            result["link"] = {  # type: ignore[assignment]
                "p": observations[link_cols].to_numpy(dtype=float),
            }

    # Store (H₂, heat).
    if net_data.stores:
        store_cols = _columns_complete(
            "store_dispatch_", list(net_data.stores), observations,
        )
        if store_cols:
            result["store"] = {  # type: ignore[assignment]
                "p": observations[store_cols].to_numpy(dtype=float),
            }

    return result


def _slice_observations(
    obs_arrays: dict[str, np.ndarray],
    batch: ActiveSetBatch,
) -> dict[str, np.ndarray]:
    """Return ``obs_arrays`` row-sliced to the batch's timesteps.

    The ``storage`` entry is a nested dict of (n_t, n_storage) arrays
    rather than a flat array; it must be sliced per-key. We forward
    it as-is and let the formulation index the rows it needs — when
    storage is present we collapse ASTB to a single full-T batch so
    no row-subsetting happens anyway.
    """
    indices = batch.timestep_indices
    out: dict[str, Any] = {}
    nested_keys = {"storage", "link", "store"}
    for key, arr in obs_arrays.items():
        if key in nested_keys:
            out[key] = arr   # dict of (n_t, n_*) — forward unchanged
        else:
            out[key] = arr[indices]
    return out


__all__ = ["FormulationType", "calibrate"]
