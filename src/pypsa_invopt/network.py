"""PyPSA-network adapter: read topology + parameters into the internal
schema, and write recovered values back.

This module contains no optimisation logic. It is the single point
where ``pypsa.Network`` objects are translated to / from the typed
dataclass that the formulations consume.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from pypsa_invopt.exceptions import InvoptInputError
from pypsa_invopt.utils.ptdf import compute_ptdf

if TYPE_CHECKING:
    import pandas as pd
    import pypsa

    from pypsa_invopt.results import InverseResult


@dataclass
class InvoptNetworkData:
    """Typed view of a ``pypsa.Network`` used by every formulation.

    Built once by :func:`read_network` and shared across timesteps and
    formulations. All collections are ordered consistently so that
    ``bus_index``/``line_index``/``gen_index`` line up with the PTDF
    matrix rows and columns.

    Attributes:
        buses: Ordered list of bus names.
        generators: Ordered list of generator names.
        lines: Ordered list of line names.
        gen_bus: Generator → its bus.
        gen_p_nom: Generator → nominal capacity (MW).
        gen_p_min: Generator → lower dispatch bound (MW) = ``p_nom * p_min_pu``.
            For static models this is the only field. For time-varying
            availability (renewables), this is the per-generator *minimum*
            of the time-series; the per-snapshot value lives in
            ``gen_p_min_t``.
        gen_p_max: Generator → upper dispatch bound (MW) = ``p_nom * p_max_pu``.
            For static models this is the only field. For time-varying
            availability (renewables, run-of-river hydro, demand-side
            resources), the per-snapshot value lives in ``gen_p_max_t``;
            this attribute holds the snapshot-wise *maximum*.
        gen_p_max_t: Optional ``(n_t, n_gens)`` array of per-snapshot
            upper dispatch bounds (MW). Built from
            ``network.generators_t.p_max_pu`` when present, otherwise
            ``None`` and downstream code falls back to ``gen_p_max``.
        gen_p_min_t: Optional ``(n_t, n_gens)`` array of per-snapshot
            lower dispatch bounds (MW).
        gen_marginal_cost: Generator → marginal cost (EUR/MWh).
        line_bus0: Line → "from" bus.
        line_bus1: Line → "to" bus.
        line_s_nom: Line → thermal limit (MW).
        line_x: Line → series reactance (pu).
        bus_generators: Bus → generators sited at it.
        bus_lines: Bus → incident lines.
        ptdf: PTDF matrix of shape ``(n_lines, n_buses)``; ``PTDF[l, b]``
            is the sensitivity of flow on line ``l`` to a unit injection
            at bus ``b``.
        bus_index, line_index, gen_index: Name → row/column index for
            ``ptdf`` and other dense arrays.
    """

    buses: list[str]
    generators: list[str]
    lines: list[str]

    gen_bus: dict[str, str]
    gen_p_nom: dict[str, float]
    gen_p_min: dict[str, float]
    gen_p_max: dict[str, float]
    gen_marginal_cost: dict[str, float]

    line_bus0: dict[str, str]
    line_bus1: dict[str, str]
    line_s_nom: dict[str, float]
    line_x: dict[str, float]

    bus_generators: dict[str, list[str]]
    bus_lines: dict[str, list[str]]

    ptdf: np.ndarray

    bus_index: dict[str, int] = field(default_factory=dict)
    line_index: dict[str, int] = field(default_factory=dict)
    gen_index: dict[str, int] = field(default_factory=dict)

    # Time-varying dispatch bounds (renewables availability profiles
    # etc.). ``None`` when the network is static, in which case
    # downstream code uses ``gen_p_max`` / ``gen_p_min`` directly.
    gen_p_max_t: np.ndarray | None = None
    gen_p_min_t: np.ndarray | None = None

    # Quadratic (heat-rate) coefficient — PyPSA's
    # ``marginal_cost_quadratic`` column. The forward LOPF objective
    # then includes ``Σ_t (c_0 · p + c_1 · p²)`` per generator, and
    # the KKT stationarity row gains a ``+ 2 · p_obs[t] · c_1`` term.
    # When all values are zero, the package's existing linear-cost
    # code path is numerically identical to the pre-quadratic version.
    # Birge-Hortaçsu-Pavlin (2017) §3.2 motivates this as the basic
    # offer-curve recovery extension; standard inverse OPF generalises
    # cleanly because the KKT row stays linear in the cost parameters
    # given observed dispatch.
    gen_marginal_cost_quadratic: dict[str, float] = field(default_factory=dict)

    # Storage support — Phase 2 fully integrated (2026-05-14).
    # The inverse OPF recovers ``storage:<name>:marginal_cost`` via the
    # KKT system with intertemporal cyclic SOC dynamics.
    storage_units: list[str] = field(default_factory=list)
    storage_bus: dict[str, str] = field(default_factory=dict)
    storage_p_nom: dict[str, float] = field(default_factory=dict)
    storage_marginal_cost: dict[str, float] = field(default_factory=dict)
    storage_max_hours: dict[str, float] = field(default_factory=dict)
    storage_efficiency_store: dict[str, float] = field(default_factory=dict)
    storage_efficiency_dispatch: dict[str, float] = field(default_factory=dict)
    storage_index: dict[str, int] = field(default_factory=dict)
    bus_storage: dict[str, list[str]] = field(default_factory=dict)

    # Links support — 2026-05-14 audit. PyPSA's ``Link`` component
    # models HVDC, P2H, electrolysers, and similar controllable
    # bidirectional interconnectors. The KKT row is similar in shape
    # to a generator's: ``c_link − λ[bus0] + η · λ[bus1] + μ_link = 0``
    # where ``μ_link`` is a signed bound dual (positive at p_max,
    # negative at p_min, zero in interior). Recoverable parameter:
    # ``link:<name>:marginal_cost``.
    links: list[str] = field(default_factory=list)
    link_bus0: dict[str, str] = field(default_factory=dict)
    link_bus1: dict[str, str] = field(default_factory=dict)
    link_p_nom: dict[str, float] = field(default_factory=dict)
    link_p_min_pu: dict[str, float] = field(default_factory=dict)
    link_p_max_pu: dict[str, float] = field(default_factory=dict)
    link_marginal_cost: dict[str, float] = field(default_factory=dict)
    link_efficiency: dict[str, float] = field(default_factory=dict)
    link_index: dict[str, int] = field(default_factory=dict)

    # Stores support — energy stores (H₂, heat). Simpler than
    # storage_units because there's a single signed power variable
    # ``p`` (positive = discharge, negative = charge) and a separate
    # energy state ``e`` linked by ``e[t] = e[t−1] · (1 − loss) − p[t]``.
    # Recoverable parameter: ``store:<name>:marginal_cost``.
    stores: list[str] = field(default_factory=list)
    store_bus: dict[str, str] = field(default_factory=dict)
    store_e_nom: dict[str, float] = field(default_factory=dict)
    store_marginal_cost: dict[str, float] = field(default_factory=dict)
    store_standing_loss: dict[str, float] = field(default_factory=dict)
    store_index: dict[str, int] = field(default_factory=dict)

    # Global constraints — typically CO2 caps in PyPSA's
    # ``primary_energy`` form: ``Σ_t Σ_g (p_g[t]/η_g) · e_g · w_t ≤ E_max``.
    # The dual ``μ_co2 ≥ 0`` is the recoverable shadow price (EUR/tCO2).
    # Per-generator emission factor ``gen_emission_factor`` is the
    # product of carrier emission factor and 1/efficiency (the
    # "per-MWh electrical output" emission rate). Snapshot weight
    # ``snapshot_weights`` defaults to 1.0 h.
    global_constraints: list[str] = field(default_factory=list)
    global_constraint_constant: dict[str, float] = field(default_factory=dict)
    global_constraint_sense: dict[str, str] = field(default_factory=dict)
    global_constraint_carrier_attribute: dict[str, str] = field(default_factory=dict)
    gen_emission_factor: dict[str, float] = field(default_factory=dict)
    snapshot_weights: np.ndarray | None = None


def read_network(network: pypsa.Network) -> InvoptNetworkData:
    """Translate a populated ``pypsa.Network`` into :class:`InvoptNetworkData`.

    Args:
        network: A ``pypsa.Network`` with buses, generators and lines.

    Returns:
        The internal data view, with PTDF pre-computed.

    Raises:
        InvoptInputError: If the network is missing buses, generators
            or lines (all three are required for inverse OPF).
    """
    buses = list(network.buses.index)
    generators = list(network.generators.index)
    lines = list(network.lines.index)
    _require_nonempty(buses, generators, lines)

    (
        gen_bus, gen_p_nom, gen_p_min, gen_p_max, gen_marginal_cost,
        gen_marginal_cost_quadratic, gen_p_max_t, gen_p_min_t,
    ) = _read_generators(network, generators)
    (
        storage_units, storage_bus, storage_p_nom, storage_marginal_cost,
        storage_max_hours, storage_eff_store, storage_eff_dispatch,
    ) = _read_storage(network)
    (
        link_names, link_bus0_map, link_bus1_map, link_p_nom_map,
        link_pmin_map, link_pmax_map, link_cost_map, link_eff_map,
    ) = _read_links(network)
    (
        store_names, store_bus_map, store_e_nom_map,
        store_cost_map, store_loss_map,
    ) = _read_stores(network)
    (
        gc_names, gc_constant, gc_sense, gc_carrier_attr,
        gen_emission, snap_weights,
    ) = _read_global_constraints(network, generators)
    line_bus0, line_bus1, line_s_nom, line_x = _read_lines(network, lines)
    bus_generators, bus_lines = _build_topology(
        buses=buses,
        generators=generators,
        lines=lines,
        gen_bus=gen_bus,
        line_bus0=line_bus0,
        line_bus1=line_bus1,
    )

    ptdf = compute_ptdf(
        buses=buses,
        lines=lines,
        line_bus0=line_bus0,
        line_bus1=line_bus1,
        line_x=line_x,
    )

    bus_storage: dict[str, list[str]] = {b: [] for b in buses}
    for s, b in storage_bus.items():
        if b in bus_storage:
            bus_storage[b].append(s)

    return InvoptNetworkData(
        buses=buses,
        generators=generators,
        lines=lines,
        gen_bus=gen_bus,
        gen_p_nom=gen_p_nom,
        gen_p_min=gen_p_min,
        gen_p_max=gen_p_max,
        gen_marginal_cost=gen_marginal_cost,
        gen_marginal_cost_quadratic=gen_marginal_cost_quadratic,
        gen_p_max_t=gen_p_max_t,
        gen_p_min_t=gen_p_min_t,
        line_bus0=line_bus0,
        line_bus1=line_bus1,
        line_s_nom=line_s_nom,
        line_x=line_x,
        bus_generators=bus_generators,
        bus_lines=bus_lines,
        ptdf=ptdf,
        bus_index={b: i for i, b in enumerate(buses)},
        line_index={ln: i for i, ln in enumerate(lines)},
        gen_index={g: i for i, g in enumerate(generators)},
        storage_units=storage_units,
        storage_bus=storage_bus,
        storage_p_nom=storage_p_nom,
        storage_marginal_cost=storage_marginal_cost,
        storage_max_hours=storage_max_hours,
        storage_efficiency_store=storage_eff_store,
        storage_efficiency_dispatch=storage_eff_dispatch,
        storage_index={s: i for i, s in enumerate(storage_units)},
        bus_storage=bus_storage,
        links=link_names,
        link_bus0=link_bus0_map,
        link_bus1=link_bus1_map,
        link_p_nom=link_p_nom_map,
        link_p_min_pu=link_pmin_map,
        link_p_max_pu=link_pmax_map,
        link_marginal_cost=link_cost_map,
        link_efficiency=link_eff_map,
        link_index={ln: i for i, ln in enumerate(link_names)},
        stores=store_names,
        store_bus=store_bus_map,
        store_e_nom=store_e_nom_map,
        store_marginal_cost=store_cost_map,
        store_standing_loss=store_loss_map,
        store_index={s: i for i, s in enumerate(store_names)},
        global_constraints=gc_names,
        global_constraint_constant=gc_constant,
        global_constraint_sense=gc_sense,
        global_constraint_carrier_attribute=gc_carrier_attr,
        gen_emission_factor=gen_emission,
        snapshot_weights=snap_weights,
    )


def observations_from_pypsa(network: pypsa.Network) -> pd.DataFrame:
    """Build the observations DataFrame expected by :func:`calibrate`.

    Walks the *solved* ``pypsa.Network`` and emits one column per
    consumable quantity, using the column conventions the package
    expects:

    * ``price_<bus>`` — LMP from ``buses_t.marginal_price`` (required)
    * ``flow_<line>`` — signed line flow from ``lines_t.p0``
    * ``dispatch_<gen>`` — generator output from ``generators_t.p``
    * ``storage_dispatch_<s>`` / ``storage_store_<s>`` /
      ``storage_soc_<s>`` — storage_unit trajectory
    * ``link_dispatch_<l>`` — HVDC / controllable link flow
    * ``store_dispatch_<s>`` — energy-store power
    * ``mu_line_<l>`` — line shadow-price (if PyPSA reported it)

    This is the observation-construction step from the AI workflow
    ("you observe LMPs and dispatch, but not the bids") collapsed into
    one call. The caller can drop / overwrite any columns they don't
    want the inverse OPF to use.

    Args:
        network: A ``pypsa.Network`` that has already been solved
            (``network.optimize(...)``). The function reads only the
            ``*_t`` attributes; it does not touch the solver model.

    Returns:
        ``pandas.DataFrame`` indexed by ``network.snapshots``, ready
        to pass into :func:`pypsa_invopt.calibrate` as
        ``observations``.

    Raises:
        ValueError: If ``buses_t.marginal_price`` is empty (network
            not solved yet, or solve infeasible).
    """
    import pandas as pd

    if network.buses_t.marginal_price.empty:
        raise ValueError(
            "network has no buses_t.marginal_price — solve it first via "
            "network.optimize(solver_name='highs').",
        )
    obs = pd.DataFrame(index=network.snapshots.copy())
    for bus in network.buses.index:
        obs[f"price_{bus}"] = network.buses_t.marginal_price[bus]
    for line in network.lines.index:
        if line in network.lines_t.p0.columns:
            obs[f"flow_{line}"] = network.lines_t.p0[line]
        if (hasattr(network.lines_t, "mu_upper")
                and line in network.lines_t.mu_upper.columns):
            obs[f"mu_line_{line}"] = network.lines_t.mu_upper[line]
    for gen in network.generators.index:
        if gen in network.generators_t.p.columns:
            obs[f"dispatch_{gen}"] = network.generators_t.p[gen]
    for s in network.storage_units.index:
        if s in network.storage_units_t.p_dispatch.columns:
            obs[f"storage_dispatch_{s}"] = network.storage_units_t.p_dispatch[s]
            obs[f"storage_store_{s}"]    = network.storage_units_t.p_store[s]
            obs[f"storage_soc_{s}"]      = network.storage_units_t.state_of_charge[s]
    for ln in network.links.index:
        if ln in network.links_t.p0.columns:
            obs[f"link_dispatch_{ln}"] = network.links_t.p0[ln]
    for s in network.stores.index:
        if s in network.stores_t.p.columns:
            obs[f"store_dispatch_{s}"] = network.stores_t.p[s]
    return obs


def _write_simple_cost(
    network: pypsa.Network,
    df_attr: str,
    name: str,
    val: float,
    *,
    field: str = "marginal_cost",
    clamp_nonneg: bool = True,
) -> None:
    """Write ``max(0, val)`` (or just ``val``) into ``network.<df_attr>.at[name, field]``."""
    df = getattr(network, df_attr, None)
    if df is None or name not in df.index:
        return
    df.at[name, field] = max(0.0, float(val)) if clamp_nonneg else float(val)


def _write_zone_cost(
    network: pypsa.Network, zone: str, val: float,
    bus_to_zone: dict[str, str] | None,
) -> None:
    """Zonal writeback fans out one cost to all generators in the zone."""
    target_buses = _resolve_zone_target_buses(network, zone, bus_to_zone)
    if not target_buses:
        return
    new_cost = max(0.0, float(val))
    for gen in network.generators.index:
        if str(network.generators.at[gen, "bus"]) in target_buses:
            network.generators.at[gen, "marginal_cost"] = new_cost


def _write_line_param(
    network: pypsa.Network, name: str, attr: str, val: float,
) -> None:
    """Line params: `x`, `susceptance` (legacy), or `s_nom`."""
    if name not in network.lines.index:
        return
    if attr == "x" and val > 1e-12:
        network.lines.at[name, "x"] = float(val)
    elif attr == "susceptance" and val > 1e-12:
        network.lines.at[name, "x"] = 1.0 / float(val)
    elif attr == "s_nom" and val > 0:
        network.lines.at[name, "s_nom"] = float(val)


def apply_result(
    result: InverseResult,
    network: pypsa.Network,
    *,
    bus_to_zone: dict[str, str] | None = None,
) -> None:
    """Write recovered parameters from ``result`` back to ``network`` in place.

    Recognised ``theta_hat`` keys (format ``"<comp>:<name>:<attr>"``):

    * ``gen:<name>:marginal_cost``           — generator linear cost
    * ``gen:<name>:marginal_cost_quadratic`` — generator heat-rate slope
    * ``storage:<name>:marginal_cost``       — storage_unit cycling cost
    * ``link:<name>:marginal_cost``          — HVDC / link transit cost
    * ``store:<name>:marginal_cost``         — energy-store cost
    * ``global_constraint:<name>:mu``        — emissions / RES-cap shadow price
    * ``zone:<name>:marginal_cost``          — zonal cost fanned out to every
      generator in the matching bidding zone (slug matched as bus name first,
      then via ``bus_to_zone``)
    * ``line:<name>:x | susceptance | s_nom`` — line parameters

    Unrecognised keys are silently ignored so callers can pass through
    diagnostic entries (e.g. ``ntc:*``, ``ntc_shadow_mean:*``) without
    filtering.
    """
    for key, val in result.theta_hat.items():
        parts = key.split(":")
        if len(parts) != 3:
            continue
        comp, name, attr = parts

        if comp == "gen" and attr == "marginal_cost":
            _write_simple_cost(network, "generators", name, val, clamp_nonneg=False)
        elif comp == "gen" and attr == "marginal_cost_quadratic":
            _write_simple_cost(network, "generators", name, val,
                               field="marginal_cost_quadratic")
        elif comp == "storage" and attr == "marginal_cost":
            _write_simple_cost(network, "storage_units", name, val)
        elif comp == "link" and attr == "marginal_cost":
            _write_simple_cost(network, "links", name, val)
        elif comp == "store" and attr == "marginal_cost":
            _write_simple_cost(network, "stores", name, val)
        elif comp == "global_constraint" and attr == "mu":
            # μ ≥ 0 by complementary slackness on a ≤ cap.
            _write_simple_cost(network, "global_constraints", name, val, field="mu")
        elif comp == "zone" and attr == "marginal_cost":
            _write_zone_cost(network, name, val, bus_to_zone)
        elif comp == "line":
            _write_line_param(network, name, attr, val)


def _resolve_zone_target_buses(
    network: pypsa.Network,
    zone_name: str,
    bus_to_zone: dict[str, str] | None,
) -> set[str]:
    """Resolve a zonal ``name`` to the set of buses it covers.

    Single-bus-per-zone (default): the zone label is itself a bus
    name. Multi-bus-per-zone: pass ``bus_to_zone`` mapping each bus
    to its zone. Returns an empty set when neither lookup
    succeeds — caller silently skips.
    """
    if zone_name in network.buses.index:
        return {zone_name}
    if bus_to_zone is None:
        return set()
    return {bus for bus, zone in bus_to_zone.items() if zone == zone_name}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_nonempty(
    buses: list[str],
    generators: list[str],
    lines: list[str],
) -> None:
    if not buses:
        raise InvoptInputError(
            "Network has no buses. Add some with network.add('Bus', ...)."
        )
    if not generators:
        raise InvoptInputError(
            "Network has no generators. Add some with "
            "network.add('Generator', ...)."
        )
    if not lines:
        raise InvoptInputError(
            "Network has no lines. Inverse OPF requires at least one "
            "transmission line."
        )


def _read_generators(
    network: pypsa.Network,
    generators: list[str],
) -> tuple[
    dict[str, str],         # gen -> bus
    dict[str, float],       # gen -> p_nom
    dict[str, float],       # gen -> p_min (scalar)
    dict[str, float],       # gen -> p_max (scalar)
    dict[str, float],       # gen -> marginal_cost (linear)
    dict[str, float],       # gen -> marginal_cost_quadratic (heat-rate slope)
    np.ndarray | None,      # (n_t, n_gens) per-snapshot p_max
    np.ndarray | None,      # (n_t, n_gens) per-snapshot p_min
]:
    """Pull every generator attribute we need in a single pass.

    Reads both the static ``network.generators[p_min_pu, p_max_pu]``
    columns AND the time-varying ``network.generators_t.p_max_pu`` /
    ``p_min_pu`` frames if they are populated. The time-varying form
    is critical for European calibrations: renewables, run-of-river
    hydro, and demand-side resources all have per-snapshot
    availability profiles that drive much of the LMP signal.
    """
    has_pmin = "p_min_pu" in network.generators.columns
    has_pmax = "p_max_pu" in network.generators.columns

    bus_map: dict[str, str] = {}
    p_nom: dict[str, float] = {}
    p_min: dict[str, float] = {}
    p_max: dict[str, float] = {}
    marginal_cost: dict[str, float] = {}

    has_quad = "marginal_cost_quadratic" in network.generators.columns
    marginal_cost_quadratic: dict[str, float] = {}

    for g in generators:
        row = network.generators.loc[g]
        nom = float(row["p_nom"])
        bus_map[g] = str(row["bus"])
        p_nom[g] = nom
        p_min[g] = nom * (float(row["p_min_pu"]) if has_pmin else 0.0)
        p_max[g] = nom * (float(row["p_max_pu"]) if has_pmax else 1.0)
        marginal_cost[g] = float(row["marginal_cost"])
        marginal_cost_quadratic[g] = (
            float(row["marginal_cost_quadratic"]) if has_quad else 0.0
        )

    p_max_t = _read_time_varying_bound(
        network, generators, attr="p_max_pu",
        fallback=p_max, p_nom=p_nom,
    )
    p_min_t = _read_time_varying_bound(
        network, generators, attr="p_min_pu",
        fallback=p_min, p_nom=p_nom,
    )
    return bus_map, p_nom, p_min, p_max, marginal_cost, marginal_cost_quadratic, p_max_t, p_min_t


def _read_storage(
    network: pypsa.Network,
) -> tuple[
    list[str],          # storage names
    dict[str, str],     # storage -> bus
    dict[str, float],   # storage -> p_nom (MW, both charge and discharge)
    dict[str, float],   # storage -> marginal_cost (EUR/MWh)
    dict[str, float],   # storage -> max_hours (energy = p_nom * max_hours)
    dict[str, float],   # storage -> efficiency_store
    dict[str, float],   # storage -> efficiency_dispatch
]:
    """Pull every storage_unit attribute the inverse OPF needs.

    Returns empty containers when the network has no storage_units —
    the inverse OPF then runs the legacy generator-only path
    unchanged.
    """
    units = getattr(network, "storage_units", None)
    if units is None or len(units) == 0:
        empty: dict[str, float] = {}
        return [], {}, empty, empty, empty, empty, empty

    names = list(units.index)
    bus_map: dict[str, str] = {}
    p_nom: dict[str, float] = {}
    marginal_cost: dict[str, float] = {}
    max_hours: dict[str, float] = {}
    eff_store: dict[str, float] = {}
    eff_dispatch: dict[str, float] = {}

    for s in names:
        row = units.loc[s]
        bus_map[s] = str(row["bus"])
        p_nom[s] = float(row["p_nom"])
        marginal_cost[s] = float(row.get("marginal_cost", 0.0))
        max_hours[s] = float(row.get("max_hours", 1.0))
        eff_store[s] = float(row.get("efficiency_store", 1.0))
        eff_dispatch[s] = float(row.get("efficiency_dispatch", 1.0))

    return names, bus_map, p_nom, marginal_cost, max_hours, eff_store, eff_dispatch


def _read_links(
    network: pypsa.Network,
) -> tuple[
    list[str],
    dict[str, str],
    dict[str, str],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
]:
    """Read the network's ``Link`` components.

    Returns empty containers when no links are present so legacy
    line-only networks remain numerically identical.
    """
    links = getattr(network, "links", None)
    if links is None or len(links) == 0:
        empty: dict[str, float] = {}
        return [], {}, {}, empty, empty, empty, empty, empty

    names = list(links.index)
    bus0: dict[str, str] = {}
    bus1: dict[str, str] = {}
    p_nom: dict[str, float] = {}
    p_min_pu: dict[str, float] = {}
    p_max_pu: dict[str, float] = {}
    marginal_cost: dict[str, float] = {}
    efficiency: dict[str, float] = {}
    for ln in names:
        row = links.loc[ln]
        bus0[ln] = str(row["bus0"])
        bus1[ln] = str(row["bus1"])
        p_nom[ln] = float(row["p_nom"])
        p_min_pu[ln] = float(row.get("p_min_pu", 0.0))
        p_max_pu[ln] = float(row.get("p_max_pu", 1.0))
        marginal_cost[ln] = float(row.get("marginal_cost", 0.0))
        efficiency[ln] = float(row.get("efficiency", 1.0))
    return names, bus0, bus1, p_nom, p_min_pu, p_max_pu, marginal_cost, efficiency


def _read_global_constraints(
    network: pypsa.Network,
    generators: list[str],
) -> tuple[
    list[str],                    # constraint names
    dict[str, float],             # constraint → constant (cap value)
    dict[str, str],               # constraint → sense (<=, ==, >=)
    dict[str, str],               # constraint → carrier_attribute (e.g. "co2_emissions")
    dict[str, float],             # generator → per-MWh-electrical emission factor (tCO2/MWh)
    np.ndarray | None,            # snapshot weightings (1.0 h default)
]:
    """Read PyPSA's ``global_constraints`` (typically CO2/RES caps).

    Returns the per-constraint metadata plus a flat
    ``gen_emission_factor`` mapping each generator to its
    effective per-MWh-electrical CO2 emission (= carrier
    ``co2_emissions`` / generator ``efficiency``). Empty when no
    constraints are defined — the legacy CO2-free path runs
    unchanged.

    Mathematical reference: PyPSA's "primary_energy" constraint reads
    ``Σ_t Σ_g (p_g[t] / η_g) · e_carrier(g) · w_t ≤ E_max`` with dual
    ``μ_co2 ≥ 0``. The dual contributes
    ``+ (e_g / η_g) · w_t · μ_co2`` to every generator-snapshot KKT
    stationarity row. Verified against Korpås & Botterud (2020) MIT
    CEEPR equilibrium analysis.
    """
    gcs = getattr(network, "global_constraints", None)
    snaps = getattr(network, "snapshots", None)
    n_t = len(snaps) if snaps is not None else 0
    snap_weights_arr: np.ndarray | None = None
    if hasattr(network, "snapshot_weightings"):
        sw = network.snapshot_weightings
        if hasattr(sw, "generators"):
            snap_weights_arr = sw.generators.to_numpy(dtype=float)
        elif hasattr(sw, "objective"):
            snap_weights_arr = sw.objective.to_numpy(dtype=float)
        elif hasattr(sw, "values"):
            snap_weights_arr = np.asarray(sw, dtype=float)
    if snap_weights_arr is None and n_t > 0:
        snap_weights_arr = np.ones(n_t, dtype=float)

    if gcs is None or len(gcs) == 0:
        return [], {}, {}, {}, {}, snap_weights_arr

    names: list[str] = []
    constant: dict[str, float] = {}
    sense: dict[str, str] = {}
    carrier_attr: dict[str, str] = {}
    for gc in gcs.index:
        row = gcs.loc[gc]
        # We support the primary_energy CO2 cap shape. Skip anything
        # that's not on a recognised carrier_attribute — the user
        # can extend the list as needed.
        gc_type = str(row.get("type", "primary_energy"))
        if gc_type != "primary_energy":
            continue
        names.append(str(gc))
        constant[str(gc)] = float(row["constant"])
        sense[str(gc)] = str(row.get("sense", "<="))
        carrier_attr[str(gc)] = str(row.get("carrier_attribute", "co2_emissions"))

    # Per-generator emission factor: e_g = e_carrier(g) / efficiency_g
    gen_emission: dict[str, float] = {}
    carriers_df = getattr(network, "carriers", None)
    for g in generators:
        gen_row = network.generators.loc[g]
        carrier = str(gen_row.get("carrier", ""))
        eff = float(gen_row.get("efficiency", 1.0)) or 1.0
        if carrier and carriers_df is not None and carrier in carriers_df.index:
            # Take the first global constraint's carrier_attribute as
            # the lookup key. PyPSA conventionally uses "co2_emissions".
            attr = next(iter(carrier_attr.values())) if carrier_attr else "co2_emissions"
            e_carrier = float(carriers_df.at[carrier, attr]) if attr in carriers_df.columns else 0.0
        else:
            e_carrier = 0.0
        gen_emission[g] = e_carrier / eff

    return names, constant, sense, carrier_attr, gen_emission, snap_weights_arr


def _read_stores(
    network: pypsa.Network,
) -> tuple[
    list[str],
    dict[str, str],
    dict[str, float],
    dict[str, float],
    dict[str, float],
]:
    """Read the network's ``Store`` components."""
    stores = getattr(network, "stores", None)
    if stores is None or len(stores) == 0:
        empty: dict[str, float] = {}
        return [], {}, empty, empty, empty
    names = list(stores.index)
    bus_map: dict[str, str] = {}
    e_nom: dict[str, float] = {}
    marginal_cost: dict[str, float] = {}
    standing_loss: dict[str, float] = {}
    for s in names:
        row = stores.loc[s]
        bus_map[s] = str(row["bus"])
        e_nom[s] = float(row["e_nom"])
        marginal_cost[s] = float(row.get("marginal_cost", 0.0))
        standing_loss[s] = float(row.get("standing_loss", 0.0))
    return names, bus_map, e_nom, marginal_cost, standing_loss


def _read_time_varying_bound(
    network: pypsa.Network,
    generators: list[str],
    *,
    attr: str,
    fallback: dict[str, float],
    p_nom: dict[str, float],
) -> np.ndarray | None:
    """Build a ``(n_t, n_gens)`` per-snapshot dispatch-bound array.

    Returns ``None`` when no generator has a time-varying ``attr``
    profile (i.e. the static value is correct for every snapshot) —
    downstream code uses ``gen_p_max`` / ``gen_p_min`` directly in
    that case to keep the fast path unchanged.
    """
    gens_t = getattr(network, "generators_t", None)
    if gens_t is None:
        return None
    frame = getattr(gens_t, attr, None)
    if frame is None or frame.empty:
        return None

    snaps = network.snapshots
    n_t = len(snaps)
    n_gens = len(generators)
    out = np.zeros((n_t, n_gens), dtype=float)
    has_time_varying = False
    for j, g in enumerate(generators):
        if g in frame.columns:
            series = frame[g].reindex(snaps).astype(float).to_numpy()
            out[:, j] = series * p_nom[g]
            has_time_varying = True
        else:
            out[:, j] = float(fallback[g])
    return out if has_time_varying else None


def _read_lines(
    network: pypsa.Network,
    lines: list[str],
) -> tuple[
    dict[str, str],     # line -> bus0
    dict[str, str],     # line -> bus1
    dict[str, float],   # line -> s_nom
    dict[str, float],   # line -> x
]:
    """Pull every line attribute we need in a single pass."""
    bus0: dict[str, str] = {}
    bus1: dict[str, str] = {}
    s_nom: dict[str, float] = {}
    x: dict[str, float] = {}
    for ln in lines:
        row = network.lines.loc[ln]
        bus0[ln] = str(row["bus0"])
        bus1[ln] = str(row["bus1"])
        s_nom[ln] = float(row["s_nom"])
        x[ln] = float(row["x"])
    return bus0, bus1, s_nom, x


def _build_topology(
    *,
    buses: list[str],
    generators: list[str],
    lines: list[str],
    gen_bus: dict[str, str],
    line_bus0: dict[str, str],
    line_bus1: dict[str, str],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Compute ``bus -> generators`` and ``bus -> incident lines``."""
    bus_generators: dict[str, list[str]] = {b: [] for b in buses}
    for g in generators:
        bus = gen_bus[g]
        if bus in bus_generators:
            bus_generators[bus].append(g)

    bus_lines: dict[str, list[str]] = {b: [] for b in buses}
    for ln in lines:
        for bus in (line_bus0[ln], line_bus1[ln]):
            if bus in bus_lines:
                bus_lines[bus].append(ln)

    return bus_generators, bus_lines


__all__ = ["InvoptNetworkData", "apply_result", "read_network"]
