"""Engineering reference costs and strategic-withholding scorer.

For each generator, compute::

    c_ref_g = fuel_price[carrier(g)] · heat_rate_g
            + co2_price             · emission_factor[carrier(g)]
            + variable_om_g

then flag those whose recovered ``c_g`` lies more than ``z`` σ above
(``withholding``) or below (``distressed``) ``c_ref_g``. Fuel /
heat-rate / emission / O&M dicts are caller-supplied; small defaults
ship for NREL ATB 2024 / IEA WEO 2024 baseline values.

Reference: Birge, Hortaçsu, Pavlin (2017) *OR* 65(4) — the MISO
market-monitoring use case this module operationalises.
"""
from __future__ import annotations

from dataclasses import dataclass

from pypsa_invopt.identifiability import ParameterIdentifiability

# Carriers whose marginal cost is structurally ≈ 0 and which are
# almost always bound-binding (at p_max during sun/wind, at 0 otherwise).
# The inverse problem cannot pin their bid: ν absorbs any deviation.
# We exclude them from withholding scoring by default to suppress the
# textbook false-positive ("wind flagged at €50 because the prior
# pulled it there with no data signal"). Caller can override via
# ``include_carriers``.
_NON_DISPATCHABLE_CARRIERS: frozenset[str] = frozenset({
    "wind", "solar", "hydro", "renewables",
})

# --- Engineering defaults (overridable per call) -----------------------------

# NREL ATB 2024 heat rates (MMBtu/MWh — divide by ~3.41 to get thermal MWh / MWh).
# Quoted in "MWh-thermal per MWh-electrical" for clean unit composition
# with fuel prices in EUR/MWh-thermal.
_DEFAULT_HEAT_RATE: dict[str, float] = {
    "nuclear":     3.05,   # high heat rate, low fuel cost
    "coal":        2.78,
    "lignite":     3.30,
    "ccgt":        1.85,   # combined-cycle gas
    "gas":         2.10,   # generic gas / OCGT
    "ocgt":        2.50,   # open-cycle gas turbine peaker
    "oil":         2.50,
    "biomass":     3.00,
    "hydro":       0.0,
    "wind":        0.0,
    "solar":       0.0,
    "renewables":  0.0,
    "peaker":      2.50,
    "AC":          0.0,    # placeholder carrier
}

# IEA WEO 2024 ballpark fuel prices (EUR/MWh-thermal). Sensible
# starting point; users will typically inject real market prices.
_DEFAULT_FUEL_PRICE: dict[str, float] = {
    "nuclear":     2.5,
    "coal":        12.0,
    "lignite":     6.0,
    "ccgt":        35.0,
    "gas":         35.0,
    "ocgt":        35.0,
    "oil":         55.0,
    "biomass":     25.0,
    "hydro":       0.0,
    "wind":        0.0,
    "solar":       0.0,
    "renewables":  0.0,
    "peaker":      45.0,
    "AC":          0.0,
}

# tCO2 per MWh-thermal (IPCC / EU ETS reference factors).
_DEFAULT_EMISSION_FACTOR: dict[str, float] = {
    "nuclear":     0.0,
    "coal":        0.34,
    "lignite":     0.40,
    "ccgt":        0.20,
    "gas":         0.20,
    "ocgt":        0.20,
    "oil":         0.27,
    "biomass":     0.0,    # carbon-neutral under standard EU accounting
    "hydro":       0.0,
    "wind":        0.0,
    "solar":       0.0,
    "renewables":  0.0,
    "peaker":      0.20,
    "AC":          0.0,
}

# EU variable O&M (EUR/MWh-electrical) — NREL ATB 2024 / IEA WEO 2024.
_DEFAULT_VARIABLE_OM: dict[str, float] = {
    "nuclear":     2.5,
    "coal":        4.0,
    "lignite":     4.0,
    "ccgt":        2.0,
    "gas":         2.0,
    "ocgt":        4.0,
    "oil":         5.0,
    "biomass":     3.5,
    "hydro":       0.5,
    "wind":        0.0,
    "solar":       0.0,
    "renewables":  0.0,
    "peaker":      6.0,
    "AC":          0.0,
}


# --- Result types -----------------------------------------------------------


@dataclass(frozen=True)
class ReferenceCost:
    """The reference-cost breakdown for one generator."""

    carrier: str
    fuel_cost: float            # EUR/MWh from fuel × heat rate
    co2_cost: float             # EUR/MWh from CO2 price × emission × heat rate
    variable_om: float          # EUR/MWh fixed O&M
    total: float                # EUR/MWh sum


@dataclass(frozen=True)
class WithholdingFlag:
    """One generator's recovered-vs-reference comparison.

    Attributes:
        recovered: Recovered marginal cost from inverse OPF.
        reference: Engineering reference (fuel + CO2 + O&M).
        deviation: ``recovered − reference`` (EUR/MWh).
        deviation_sigma: ``deviation / σ_post`` — standardised, the
            "is this statistically significant" signal.
        flag: ``'withholding'`` if recovered > reference by more than
            ``z`` σ; ``'distressed'`` if below by more than ``z`` σ;
            ``'normal'`` otherwise; ``'unidentifiable'`` if the
            posterior σ is too wide to detect anything.
        reason: Human-readable explanation.
    """

    recovered: float
    reference: float
    deviation: float
    deviation_sigma: float
    flag: str
    reason: str


# --- Public API -------------------------------------------------------------


def compute_reference_cost(
    carrier: str,
    *,
    fuel_price: float | None = None,
    co2_price: float = 0.0,
    heat_rate: float | None = None,
    emission_factor: float | None = None,
    variable_om: float | None = None,
) -> ReferenceCost:
    """Build the engineering reference cost for a single carrier.

    Any argument left as ``None`` is looked up from the module's
    NREL-ATB / IEA-WEO defaults. ``co2_price`` defaults to ``0.0``
    because the package doesn't make a policy assumption — the user
    supplies the prevailing EUA price for their market.
    """
    c = carrier or "AC"
    fp = _DEFAULT_FUEL_PRICE.get(c, 0.0) if fuel_price is None else fuel_price
    hr = _DEFAULT_HEAT_RATE.get(c, 0.0) if heat_rate is None else heat_rate
    ef = (
        _DEFAULT_EMISSION_FACTOR.get(c, 0.0)
        if emission_factor is None else emission_factor
    )
    vom = (
        _DEFAULT_VARIABLE_OM.get(c, 0.0)
        if variable_om is None else variable_om
    )
    fuel_cost = fp * hr
    co2_cost = co2_price * ef * hr
    total = fuel_cost + co2_cost + vom
    return ReferenceCost(
        carrier=c,
        fuel_cost=fuel_cost,
        co2_cost=co2_cost,
        variable_om=vom,
        total=total,
    )


def flag_withholding(
    *,
    theta_hat: dict[str, float],
    generator_carriers: dict[str, str],
    posterior_identifiability: dict[str, ParameterIdentifiability] | None = None,
    fuel_prices: dict[str, float] | None = None,
    co2_price: float = 0.0,
    heat_rates: dict[str, float] | None = None,
    emission_factors: dict[str, float] | None = None,
    variable_oms: dict[str, float] | None = None,
    z_threshold: float = 2.0,
    absolute_threshold: float = 5.0,
    include_carriers: set[str] | None = None,
    skip_non_dispatchable: bool = True,
) -> dict[str, WithholdingFlag]:
    """Compare every recovered cost against its engineering reference.

    Args:
        theta_hat: Recovered parameters keyed ``gen:<name>:marginal_cost``.
        generator_carriers: Map ``gen_name → carrier`` (from
            ``network.generators['carrier']``).
        posterior_identifiability: Optional output of
            :func:`pypsa_invopt.identifiability.compute_identifiability`
            for the same posterior. When supplied, the function uses
            the per-generator ``σ_post`` to compute the standardised
            deviation; otherwise falls back to the ``absolute_threshold``.
        fuel_prices: Per-carrier override of the default fuel price
            (EUR/MWh-thermal).
        co2_price: EUA price (EUR/tCO2). Default 0.
        heat_rates: Per-generator override of the carrier default.
        emission_factors: Per-carrier override.
        variable_oms: Per-generator override.
        z_threshold: Standardised-deviation threshold for flagging
            withholding / distressed. Default 2.0 (≈ 95 % confidence).
        absolute_threshold: Fallback EUR/MWh deviation threshold when
            no posterior σ is available. Default 5.0.
        include_carriers: If supplied, score only generators on these
            carriers; everything else is skipped. Mutually exclusive
            with ``skip_non_dispatchable=False`` semantics.
        skip_non_dispatchable: When ``True`` (default), skip generators
            on wind / solar / hydro / renewables carriers. These are
            structurally bound-binding and the inverse problem cannot
            pin their bid — flagging them would produce textbook false
            positives ("wind withholding at €50" when the prior just
            pulled it there with no data signal). Set ``False`` only if
            you know your fleet model genuinely has interior renewables.

    Returns:
        Dict mapping ``gen_name → WithholdingFlag``.
    """
    out: dict[str, WithholdingFlag] = {}
    for key, recovered in theta_hat.items():
        # We only flag generator marginal costs. Skip storage / link /
        # global-constraint keys — those have their own interpretation.
        if not key.startswith("gen:") or not key.endswith(":marginal_cost"):
            continue
        gen = key.split(":")[1]
        carrier = generator_carriers.get(gen, "")
        if include_carriers is not None and carrier not in include_carriers:
            continue
        if skip_non_dispatchable and carrier in _NON_DISPATCHABLE_CARRIERS:
            continue
        fp = (fuel_prices or {}).get(carrier)
        hr = (heat_rates or {}).get(gen)
        ef = (emission_factors or {}).get(carrier)
        vom = (variable_oms or {}).get(gen)
        ref = compute_reference_cost(
            carrier,
            fuel_price=fp, co2_price=co2_price,
            heat_rate=hr, emission_factor=ef, variable_om=vom,
        )
        deviation = recovered - ref.total
        # Standardise by σ_post when we have it; otherwise use the
        # absolute threshold and treat ``σ`` as a placeholder.
        sigma_post: float | None = None
        if posterior_identifiability is not None:
            pid = posterior_identifiability.get(key)
            if pid is not None and pid.identifiable:
                sigma_post = pid.sigma_post

        if sigma_post is None or sigma_post <= 0:
            flag, reason = _flag_absolute(
                deviation=deviation,
                threshold=absolute_threshold,
                identifiable=(
                    posterior_identifiability is None
                    or posterior_identifiability.get(key, None) is None
                    or posterior_identifiability[key].identifiable
                ),
            )
            deviation_sigma = 0.0
        else:
            deviation_sigma = deviation / sigma_post
            flag, reason = _flag_standardised(
                deviation_sigma=deviation_sigma, z=z_threshold,
                deviation_abs=deviation,
            )

        out[gen] = WithholdingFlag(
            recovered=float(recovered),
            reference=float(ref.total),
            deviation=float(deviation),
            deviation_sigma=float(deviation_sigma),
            flag=flag,
            reason=reason,
        )
    return out


def _flag_standardised(
    *, deviation_sigma: float, z: float, deviation_abs: float,
) -> tuple[str, str]:
    if deviation_sigma > z:
        return "withholding", (
            f"recovered exceeds reference by {deviation_abs:+.2f} "
            f"EUR/MWh ({deviation_sigma:+.2f}σ > {z}σ) — possible "
            "strategic withholding"
        )
    if deviation_sigma < -z:
        return "distressed", (
            f"recovered is below reference by {deviation_abs:+.2f} "
            f"EUR/MWh ({deviation_sigma:+.2f}σ < −{z}σ) — possible "
            "mis-bid or distressed dispatch"
        )
    return "normal", (
        f"deviation {deviation_abs:+.2f} EUR/MWh "
        f"({deviation_sigma:+.2f}σ) within ±{z}σ tolerance"
    )


def _flag_absolute(
    *, deviation: float, threshold: float, identifiable: bool,
) -> tuple[str, str]:
    if not identifiable:
        return "unidentifiable", (
            "posterior σ too wide to detect deviation reliably; "
            "recovered cost likely reflects the prior, not the data"
        )
    if deviation > threshold:
        return "withholding", (
            f"recovered exceeds reference by {deviation:+.2f} EUR/MWh "
            f"(> {threshold} threshold; no posterior σ available)"
        )
    if deviation < -threshold:
        return "distressed", (
            f"recovered is below reference by {deviation:+.2f} EUR/MWh "
            f"(< −{threshold} threshold; no posterior σ available)"
        )
    return "normal", f"deviation {deviation:+.2f} EUR/MWh within ±{threshold}"


__all__ = [
    "ReferenceCost",
    "WithholdingFlag",
    "compute_reference_cost",
    "flag_withholding",
]
