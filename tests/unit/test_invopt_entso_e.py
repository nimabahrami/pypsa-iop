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


def test_load_entso_e_live_path_via_mocked_client(monkeypatch):
    """Exercise the live (non-synthesize) code path with a mocked
    entsoe-py client. Covers ``_build_client`` → ``_fetch_prices`` →
    ``_attach_optional`` end-to-end, including the resample/rename.
    """
    import numpy as np
    import pandas as pd

    from pypsa_invopt.data import entso_e as e

    class _MockClient:
        def query_day_ahead_prices(self, zone, start, end):
            idx = pd.date_range(start, end, freq="h", tz="UTC")
            return pd.Series(50.0 + np.arange(len(idx), dtype=float), index=idx)
        def query_crossborder_flows(self, zone, start, end):
            idx = pd.date_range(start, end, freq="h", tz="UTC")
            return pd.DataFrame({"BE": np.arange(len(idx), dtype=float)}, index=idx)
        def query_generation(self, zone, start, end):
            # Simulate the optional-fetch swallow path
            raise ValueError("not available for this zone")

    monkeypatch.setattr(
        e, "_build_client",
        lambda api_key: _MockClient(),
    )
    df = e.load_entso_e(
        bidding_zone="NL",
        start="2025-06-15", end="2025-06-16",
        api_key="dummy",
        include_flows=True, include_generation=True,
        synthesize=False,
    )
    assert "price_NL" in df.columns
    assert any(c.startswith("flow_") for c in df.columns)
    # Generation fetch raised — should not have crashed the whole load.
    assert not any(c.startswith("gen_") for c in df.columns)


def test_build_client_raises_without_token(monkeypatch):
    """``_build_client`` must raise when neither api_key nor env var is set."""
    from pypsa_invopt.data.entso_e import _build_client
    from pypsa_invopt.exceptions import InvoptInputError

    monkeypatch.delenv("ENTSOE_API_KEY", raising=False)
    with pytest.raises(InvoptInputError, match=r"API key"):
        _build_client(api_key=None)


def test_fetch_prices_wraps_api_errors(monkeypatch):
    """``_fetch_prices`` must convert raw entsoe-py errors into
    ``InvoptInputError`` with a helpful zone-named message."""
    import pandas as pd

    from pypsa_invopt.data.entso_e import _fetch_prices
    from pypsa_invopt.exceptions import InvoptInputError

    class _BadClient:
        def query_day_ahead_prices(self, zone, start, end):
            raise RuntimeError("internal entsoe-py error")

    with pytest.raises(InvoptInputError, match=r"day-ahead prices for NL"):
        _fetch_prices(
            _BadClient(), "NL",
            pd.Timestamp("2025-06-15", tz="UTC"),
            pd.Timestamp("2025-06-16", tz="UTC"),
        )
