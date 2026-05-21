"""Shared test fixtures for pypsa-invopt."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pypsa
import pytest


@pytest.fixture
def two_bus_network() -> pypsa.Network:
    """Minimal 2-bus network for unit tests."""
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2025-01-01", periods=24, freq="h"))
    n.add("Carrier", "AC")
    n.add("Bus", "A", v_nom=110.0, carrier="AC")
    n.add("Bus", "B", v_nom=110.0, carrier="AC")
    n.add("Line", "A-B", bus0="A", bus1="B", s_nom=100.0, x=0.1, r=0.0)
    n.add("Generator", "gA", bus="A", p_nom=100.0, marginal_cost=20.0)
    n.add("Generator", "gB", bus="B", p_nom=100.0, marginal_cost=50.0)
    n.add("Load", "load_B", bus="B", p_set=50.0)
    return n


@pytest.fixture
def three_bus_network() -> pypsa.Network:
    """3-bus triangular network for PTDF validation."""
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2025-01-01", periods=24, freq="h"))
    n.add("Carrier", "AC")
    n.add("Bus", "A", v_nom=110.0, carrier="AC")
    n.add("Bus", "B", v_nom=110.0, carrier="AC")
    n.add("Bus", "C", v_nom=110.0, carrier="AC")
    n.add("Line", "A-B", bus0="A", bus1="B", s_nom=100.0, x=0.1, r=0.0)
    n.add("Line", "B-C", bus0="B", bus1="C", s_nom=100.0, x=0.2, r=0.0)
    n.add("Line", "A-C", bus0="A", bus1="C", s_nom=100.0, x=0.15, r=0.0)
    n.add("Generator", "gA", bus="A", p_nom=200.0, marginal_cost=15.0)
    n.add("Generator", "gB", bus="B", p_nom=100.0, marginal_cost=30.0)
    n.add("Generator", "gC", bus="C", p_nom=100.0, marginal_cost=60.0)
    n.add("Load", "load_B", bus="B", p_set=80.0)
    n.add("Load", "load_C", bus="C", p_set=40.0)
    return n


@pytest.fixture
def five_zone_network() -> pypsa.Network:
    """5-zone network for zonal formulation tests."""
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2025-01-01", periods=24, freq="h"))
    n.add("Carrier", "AC")
    zones = ["Z1", "Z2", "Z3", "Z4", "Z5"]
    for z in zones:
        n.add("Bus", z, v_nom=380.0, carrier="AC")
        n.add("Generator", f"gen_{z}", bus=z, p_nom=500.0,
               marginal_cost=20.0 + zones.index(z) * 10)
        n.add("Load", f"load_{z}", bus=z, p_set=200.0)

    # Line connections
    connections = [("Z1", "Z2"), ("Z2", "Z3"), ("Z3", "Z4"),
                   ("Z4", "Z5"), ("Z1", "Z5")]
    for z1, z2 in connections:
        n.add("Line", f"{z1}-{z2}", bus0=z1, bus1=z2,
               s_nom=300.0, x=0.05, r=0.0)
    return n


@pytest.fixture
def synthetic_observations(two_bus_network) -> pd.DataFrame:
    """Synthetic observations matching the 2-bus network."""
    rng = np.random.default_rng(42)
    idx = two_bus_network.snapshots
    T = len(idx)

    return pd.DataFrame(
        {
            "price_A": 20.0 + rng.normal(0, 2.0, T),
            "price_B": 50.0 + rng.normal(0, 3.0, T),
            "flow_A-B": 30.0 + rng.normal(0, 5.0, T),
            "dispatch_gA": 80.0 + rng.normal(0, 5.0, T),
            "dispatch_gB": 20.0 + rng.normal(0, 3.0, T),
        },
        index=idx,
    )
