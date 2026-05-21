"""Smoke tests for the ENTSO-E loader.

Only exercises the ``synthesize=True`` path so no API key (and no
``entsoe-py`` install) is required. Confirms the loader produces a
DataFrame in the shape downstream code expects.
"""
from __future__ import annotations

import pandas as pd
import pytest

from pypsa_invopt.data.entso_e import load_entso_e


def test_load_entso_e_synthesize_returns_price_column():
    df = load_entso_e(
        bidding_zone="NL",
        start="2025-06-15", end="2025-06-16",
        synthesize=True,
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "price_NL" in df.columns
    assert df["price_NL"].dtype.kind == "f"
    assert df["price_NL"].notna().all()


def test_load_entso_e_synthesize_includes_flow_when_requested():
    df = load_entso_e(
        bidding_zone="DE_LU",
        start="2025-06-15", end="2025-06-16",
        include_flows=True,
        synthesize=True,
    )
    flow_cols = [c for c in df.columns if c.startswith("flow_")]
    assert len(flow_cols) >= 1


def test_load_entso_e_synthesize_hourly_index():
    df = load_entso_e(
        bidding_zone="FR",
        start="2025-06-15", end="2025-06-17",
        synthesize=True,
    )
    assert isinstance(df.index, pd.DatetimeIndex)
    # Hourly frequency
    deltas = df.index.to_series().diff().dropna().unique()
    assert len(deltas) == 1
    assert deltas[0] == pd.Timedelta(hours=1)


def test_load_entso_e_live_raises_without_key_or_entsoe_py(monkeypatch):
    """Live mode requires either an API key or entsoe-py installed."""
    monkeypatch.delenv("ENTSOE_API_KEY", raising=False)
    # We can't safely call this in CI without the dep, so just confirm
    # that calling with ``synthesize=False`` does NOT silently return.
    with pytest.raises((ImportError, Exception)):
        load_entso_e(
            bidding_zone="NL",
            start="2025-06-15", end="2025-06-16",
            synthesize=False,
            api_key=None,
        )
