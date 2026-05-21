"""Reference-cost validation + strategic-withholding detection — Gap 3.

Implements the Birge-Hortaçsu-Pavlin (2017) MISO market-monitoring
use case as unit tests.
"""
from __future__ import annotations

import pytest

from pypsa_invopt import flag_withholding
from pypsa_invopt.identifiability import ParameterIdentifiability
from pypsa_invopt.reference_costs import (
    compute_reference_cost,
)


def test_compute_reference_cost_known_carrier():
    """ccgt with fuel_price=$35/MWh-th, heat rate 1.85, no CO2."""
    ref = compute_reference_cost(
        "ccgt", fuel_price=35.0, co2_price=0.0,
        heat_rate=1.85, emission_factor=0.20, variable_om=2.0,
    )
    assert ref.fuel_cost == pytest.approx(35.0 * 1.85, abs=1e-6)
    assert ref.co2_cost == pytest.approx(0.0, abs=1e-6)
    assert ref.total == pytest.approx(35.0 * 1.85 + 2.0, abs=1e-6)


def test_compute_reference_cost_with_co2_price():
    """Adding a €80/tCO2 EUA price for a gas generator."""
    ref = compute_reference_cost(
        "ccgt", fuel_price=35.0, co2_price=80.0,
        heat_rate=1.85, emission_factor=0.20, variable_om=2.0,
    )
    # CO2 cost = co2_price × emission × heat_rate = 80 × 0.20 × 1.85 = 29.60
    assert ref.co2_cost == pytest.approx(80.0 * 0.20 * 1.85, abs=1e-6)
    assert ref.total == pytest.approx(
        35.0 * 1.85 + 80.0 * 0.20 * 1.85 + 2.0, abs=1e-6,
    )


def test_flag_withholding_clear_strategic_bid():
    """A generator recovered at $200 with fuel reference ~$65 →
    flagged as strategic withholding."""
    theta_hat = {"gen:coal_plant_A:marginal_cost": 200.0}
    carriers = {"coal_plant_A": "coal"}
    # No posterior identifiability → falls back to absolute threshold
    flags = flag_withholding(
        theta_hat=theta_hat,
        generator_carriers=carriers,
        fuel_prices={"coal": 12.0},
        co2_price=80.0,
        absolute_threshold=5.0,
    )
    assert "coal_plant_A" in flags
    f = flags["coal_plant_A"]
    assert f.flag == "withholding"
    assert f.recovered == 200.0
    # Reference: 12 × 2.78 + 80 × 0.34 × 2.78 + 4 = 33.36 + 75.6 + 4 = 112.96
    assert f.reference == pytest.approx(33.36 + 75.616 + 4.0, abs=0.1)
    assert f.deviation > 80.0   # ~$87 above reference


def test_flag_withholding_normal_dispatch():
    """A generator priced near its engineering reference → normal."""
    # Reference for ccgt: 35 × 1.85 + 80 × 0.20 × 1.85 + 2 = 64.75 + 29.6 + 2 = 96.35
    theta_hat = {"gen:gas_plant:marginal_cost": 95.0}
    carriers = {"gas_plant": "ccgt"}
    flags = flag_withholding(
        theta_hat=theta_hat,
        generator_carriers=carriers,
        fuel_prices={"ccgt": 35.0},
        co2_price=80.0,
        absolute_threshold=5.0,
    )
    assert flags["gas_plant"].flag == "normal"


def test_flag_withholding_uses_posterior_sigma_when_available():
    """When identifiability is supplied, the flag uses standardised σ."""
    theta_hat = {"gen:plant:marginal_cost": 100.0}
    carriers = {"plant": "ccgt"}
    # Plant identifiable with σ_post=1.0 (tight). Reference ≈ 64.75 + 2 = 66.75
    pid = {
        "gen:plant:marginal_cost": ParameterIdentifiability(
            sigma_post=1.0, sigma_prior=5.0, information_gain=0.8,
            ci_low=98.04, ci_high=101.96, identifiable=True, reason="",
        ),
    }
    flags = flag_withholding(
        theta_hat=theta_hat,
        generator_carriers=carriers,
        posterior_identifiability=pid,
        fuel_prices={"ccgt": 35.0},
        co2_price=0.0,   # no CO2 to isolate the fuel test
        z_threshold=2.0,
    )
    # Reference ≈ 64.75 + 2 = 66.75; recovered 100; deviation ≈ +33.25;
    # standardised ≈ 33.25 (way above 2σ) → withholding
    assert flags["plant"].flag == "withholding"
    assert flags["plant"].deviation_sigma > 2.0


def test_flag_withholding_unidentifiable_marked():
    """A non-identifiable parameter is flagged as such, not as withholding."""
    theta_hat = {"gen:peaker:marginal_cost": 200.0}
    carriers = {"peaker": "ocgt"}
    pid = {
        "gen:peaker:marginal_cost": ParameterIdentifiability(
            sigma_post=50.0, sigma_prior=5.0, information_gain=0.0,
            ci_low=100.0, ci_high=300.0, identifiable=False,
            reason="σ_post=50 > threshold 2",
        ),
    }
    flags = flag_withholding(
        theta_hat=theta_hat,
        generator_carriers=carriers,
        posterior_identifiability=pid,
        fuel_prices={"ocgt": 35.0},
        co2_price=0.0,
    )
    assert flags["peaker"].flag == "unidentifiable"


def test_flag_withholding_skips_renewables_by_default():
    """Wind / solar / hydro are bound-binding and not identifiable —
    they must not appear in the withholding scorer's output by default
    (otherwise the user sees textbook false positives)."""
    theta_hat = {
        "gen:offshore_wind:marginal_cost": 45.0,    # spurious recovery
        "gen:pv_park:marginal_cost":       38.0,    # spurious recovery
        "gen:run_river:marginal_cost":     22.0,    # spurious recovery
        "gen:ccgt_main:marginal_cost":     72.0,    # legitimate
    }
    carriers = {
        "offshore_wind": "wind", "pv_park": "solar",
        "run_river": "hydro",    "ccgt_main": "gas",
    }
    flags = flag_withholding(
        theta_hat=theta_hat,
        generator_carriers=carriers,
        fuel_prices={"gas": 35.0},
        co2_price=75.0,
    )
    # Only the thermal gen survives the default filter.
    assert set(flags.keys()) == {"ccgt_main"}


def test_flag_withholding_skip_disable_lets_renewables_through():
    """``skip_non_dispatchable=False`` returns the old behaviour for
    callers that have a genuinely interior-renewable fleet model."""
    theta_hat = {"gen:offshore_wind:marginal_cost": 45.0}
    carriers = {"offshore_wind": "wind"}
    flags = flag_withholding(
        theta_hat=theta_hat,
        generator_carriers=carriers,
        skip_non_dispatchable=False,
    )
    assert "offshore_wind" in flags


def test_flag_withholding_include_carriers_filter():
    """``include_carriers`` restricts scoring to an explicit set."""
    theta_hat = {
        "gen:nuc_a:marginal_cost":  9.0,
        "gen:coal_b:marginal_cost": 40.0,
        "gen:ccgt_c:marginal_cost": 70.0,
        "gen:ocgt_d:marginal_cost": 110.0,
    }
    carriers = {"nuc_a": "nuclear", "coal_b": "coal",
                "ccgt_c": "gas",    "ocgt_d": "ocgt"}
    flags = flag_withholding(
        theta_hat=theta_hat,
        generator_carriers=carriers,
        include_carriers={"gas", "ocgt"},
        fuel_prices={"gas": 35.0, "ocgt": 35.0},
        co2_price=75.0,
    )
    assert set(flags.keys()) == {"ccgt_c", "ocgt_d"}
