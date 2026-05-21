"""Tests for the BLUE cross-batch aggregator.

The ASTB pipeline solves K active-set batches independently and
combines their per-batch ``θ_k`` estimates via inverse-variance (BLUE)
weighting. These cover the aggregator and its weight helper.
"""
from __future__ import annotations

import numpy as np

from pypsa_invopt.calibration import (
    _BatchSolution,
    _blue_aggregate,
    _blue_weight,
)
from pypsa_invopt.utils.active_set import ActiveSet


def _make_solution(
    *,
    theta: dict[str, float],
    weight: float,
    pattern: ActiveSet | None = None,
) -> _BatchSolution:
    return _BatchSolution(
        pattern=pattern or ActiveSet(),
        theta=theta,
        weight=weight,
        status="optimal",
        residuals=np.array([0.0]),
    )


def test_blue_aggregate_reduces_to_frequency_mean_when_equal_variance():
    """Equal weights → BLUE matches the frequency-weighted average."""
    sols = [
        _make_solution(theta={"gen:gA:marginal_cost": 20.0}, weight=10.0),
        _make_solution(theta={"gen:gA:marginal_cost": 30.0}, weight=10.0),
        _make_solution(theta={"gen:gA:marginal_cost": 40.0}, weight=10.0),
    ]
    result = _blue_aggregate(sols)
    assert result["gen:gA:marginal_cost"] == 30.0


def test_blue_aggregate_downweights_noisy_batches():
    """Higher BLUE weight pulls the average toward that batch's estimate."""
    sols = [
        _make_solution(theta={"gen:gA:marginal_cost": 20.0}, weight=1.0),
        _make_solution(theta={"gen:gA:marginal_cost": 50.0}, weight=99.0),
    ]
    result = _blue_aggregate(sols)
    assert result["gen:gA:marginal_cost"] > 49.0


def test_blue_aggregate_skips_keys_with_zero_total_weight():
    """A key whose every contributing batch has zero weight is dropped."""
    sols = [_make_solution(theta={"gen:gA:marginal_cost": 20.0}, weight=0.0)]
    assert _blue_aggregate(sols) == {}


def test_blue_weight_floor_prevents_division_blowup():
    """Zero residuals do not blow the BLUE weight up to infinity."""
    weight = _blue_weight(n_timesteps=24, residuals=np.zeros(24))
    assert np.isfinite(weight)
    assert weight > 0


def test_blue_weight_zero_for_empty_batch():
    """A batch with no timesteps contributes nothing to the aggregator."""
    assert _blue_weight(n_timesteps=0, residuals=np.array([])) == 0.0
