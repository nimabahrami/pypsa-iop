"""Unit tests for pio.validate_observations.

Covers the user-facing pre-flight check: must error on missing
columns / non-temporal index / empty frame; must return a warnings
list on borderline-but-acceptable data.
"""
from __future__ import annotations

import pandas as pd
import pytest

from pypsa_invopt import InvoptInputError, validate_observations


def _good_obs() -> pd.DataFrame:
    idx = pd.date_range("2025-06-15", periods=24, freq="h")
    return pd.DataFrame({
        "price_b1": [30.0 + i for i in range(24)],
        "price_b2": [32.0 + i for i in range(24)],
        "flow_l12": [50.0] * 24,
        "dispatch_g1": [80.0] * 24,
        "dispatch_g2": [40.0] * 24,
    }, index=idx)


def test_validate_clean_dataframe_returns_empty_warnings():
    obs = _good_obs()
    warnings = validate_observations(
        obs,
        required_price_columns=["price_b1", "price_b2"],
        required_flow_columns=["flow_l12"],
        required_dispatch_columns=["dispatch_g1", "dispatch_g2"],
    )
    assert warnings == []


def test_validate_raises_on_empty_dataframe():
    with pytest.raises(InvoptInputError, match="empty"):
        validate_observations(pd.DataFrame())


def test_validate_raises_on_non_temporal_index():
    obs = _good_obs().reset_index(drop=True)
    with pytest.raises(InvoptInputError, match="DatetimeIndex"):
        validate_observations(obs)


def test_validate_raises_on_missing_required_columns():
    obs = _good_obs().drop(columns=["price_b2"])
    with pytest.raises(InvoptInputError, match="price"):
        validate_observations(
            obs, required_price_columns=["price_b1", "price_b2"],
        )


def test_validate_raises_on_nan_in_required_when_not_allowed():
    obs = _good_obs()
    obs.iloc[5, 0] = float("nan")
    with pytest.raises(InvoptInputError, match=r"[Mm]issing"):
        validate_observations(
            obs,
            required_price_columns=["price_b1"],
            allow_missing=False,
        )


def test_validate_warns_but_does_not_raise_when_allow_missing():
    obs = _good_obs()
    obs.iloc[5, 0] = float("nan")
    warnings = validate_observations(
        obs,
        required_price_columns=["price_b1"],
        allow_missing=True,
    )
    assert any("missing" in w.lower() for w in warnings)


def test_validate_handles_no_required_columns_argument():
    # No required_*_columns args → only structural checks
    warnings = validate_observations(_good_obs())
    assert warnings == []
