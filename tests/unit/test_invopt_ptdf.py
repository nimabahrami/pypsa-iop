"""PTDF computation tests."""
import numpy as np
import pytest

from pypsa_invopt.utils.ptdf import compute_ptdf, validate_ptdf


def test_ptdf_3bus_triangle():
    """Validate PTDF against analytical solution for a 3-bus triangle."""
    buses = ["A", "B", "C"]
    lines = ["A-B", "B-C", "A-C"]
    line_bus0 = {"A-B": "A", "B-C": "B", "A-C": "A"}
    line_bus1 = {"A-B": "B", "B-C": "C", "A-C": "C"}
    line_x = {"A-B": 0.1, "B-C": 0.2, "A-C": 0.15}

    ptdf = compute_ptdf(
        buses=buses, lines=lines,
        line_bus0=line_bus0, line_bus1=line_bus1, line_x=line_x,
    )

    assert ptdf.shape == (3, 3)
    # Slack column (bus A = index 0) should be zero
    np.testing.assert_allclose(ptdf[:, 0], 0.0, atol=1e-10)
    # Validate row sums ~ 0
    assert validate_ptdf(ptdf)


def test_ptdf_2bus_simple():
    """Simple 2-bus: PTDF should be [0, 1] (injection at B causes full flow)."""
    ptdf = compute_ptdf(
        buses=["A", "B"], lines=["L1"],
        line_bus0={"L1": "A"}, line_bus1={"L1": "B"},
        line_x={"L1": 0.1},
    )
    assert ptdf.shape == (1, 2)
    np.testing.assert_allclose(ptdf[0, 0], 0.0, atol=1e-10)
    # Injection at B (non-slack) should give PTDF = -1 (flow from A to B)
    assert abs(ptdf[0, 1]) > 0.5


def test_ptdf_zero_reactance_raises():
    """Zero reactance should raise ValueError."""
    with pytest.raises(ValueError, match="non-positive reactance"):
        compute_ptdf(
            buses=["A", "B"], lines=["L1"],
            line_bus0={"L1": "A"}, line_bus1={"L1": "B"},
            line_x={"L1": 0.0},
        )


def test_ptdf_single_bus():
    """Single bus network returns zero PTDF."""
    ptdf = compute_ptdf(
        buses=["A"], lines=[], line_bus0={}, line_bus1={}, line_x={},
    )
    assert ptdf.shape == (0, 1)
