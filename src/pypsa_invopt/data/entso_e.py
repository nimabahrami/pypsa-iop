"""ENTSO-E Transparency Platform data integration.

Thin wrapper around ``entsoe-py`` that fetches day-ahead prices,
cross-border flows and actual generation, all aligned to a common
hourly time index. A synthetic generator with realistic-looking
daily/weekly/seasonal patterns is included for tests and demos so the
rest of the package can be exercised without an ENTSO-E API key.

Requires the optional dep group: ``pip install pypsa-invopt[entso_e]``.
"""
from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd

from pypsa_invopt.exceptions import InvoptInputError

_DEFAULT_BASE_PRICE: float = 50.0
_PEAK_PRICE_BUMP: float = 10.0
_OFF_PEAK_PRICE_DIP: float = -5.0
_WEEKEND_PRICE_DIP: float = -8.0
_WINTER_PRICE_BUMP: float = 15.0
_SUMMER_PRICE_DIP: float = -5.0
_PRICE_NOISE_STD: float = 5.0
_PEAK_HOUR_LO: int = 8
_PEAK_HOUR_HI: int = 20


def load_entso_e(
    bidding_zone: str,
    start: str | datetime,
    end: str | datetime,
    *,
    api_key: str | None = None,
    include_flows: bool = True,
    include_generation: bool = True,
    synthesize: bool = False,
) -> pd.DataFrame:
    """Load market data for one bidding zone.

    Args:
        bidding_zone: ENTSO-E code (``"DE_LU"``, ``"NL"``, ``"FR"``, …).
        start: Inclusive start (anything ``pd.Timestamp`` accepts).
        end: Inclusive end.
        api_key: ENTSO-E token. Falls back to the ``ENTSOE_API_KEY``
            environment variable.
        include_flows: Fetch commercial cross-border flows.
        include_generation: Fetch actual generation per type.
        synthesize: If ``True``, skip the API and produce synthetic
            but plausible time series.

    Returns:
        DataFrame with a tz-aware DatetimeIndex and at least one
        ``price_<zone>`` column.

    Raises:
        ImportError: If ``synthesize=False`` and ``entsoe-py`` is not
            installed.
        InvoptInputError: If no API key is available (and synthesise is off).
    """
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    if synthesize:
        return _synthesise(bidding_zone, start_ts, end_ts)

    client = _build_client(api_key)
    prices = _fetch_prices(client, bidding_zone, start_ts, end_ts)

    result = pd.DataFrame(prices)
    if include_flows:
        _attach_optional(
            result, client.query_crossborder_flows,
            bidding_zone, start_ts, end_ts, prefix="flow_",
        )
    if include_generation:
        _attach_optional(
            result, client.query_generation,
            bidding_zone, start_ts, end_ts, prefix="gen_",
        )

    result.index = pd.DatetimeIndex(result.index, name="timestamp")
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_client(api_key: str | None):
    """Lazy-import entsoe-py and return an authenticated client."""
    try:
        from entsoe import EntsoePandasClient
    except ImportError as exc:
        raise ImportError(
            "ENTSO-E data integration requires entsoe-py. Install with "
            "'pip install pypsa-invopt[entso_e]'."
        ) from exc

    token = api_key or os.environ.get("ENTSOE_API_KEY")
    if token is None:
        raise InvoptInputError(
            "ENTSO-E API key not found. Pass api_key= or set the "
            "ENTSOE_API_KEY environment variable. Free key at "
            "https://transparency.entsoe.eu/."
        )
    return EntsoePandasClient(api_key=token)


def _fetch_prices(client, zone: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Pull day-ahead prices, resample to hourly, return a Series."""
    try:
        prices = client.query_day_ahead_prices(zone, start=start, end=end)
    except Exception as exc:  # entsoe-py raises a wide variety of errors
        raise InvoptInputError(
            f"Failed to fetch day-ahead prices for {zone}: {exc}"
        ) from exc
    prices = prices.resample("h").mean()
    prices.name = f"price_{zone}"
    return prices


def _attach_optional(
    frame: pd.DataFrame,
    fetcher,
    zone: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    prefix: str,
) -> None:
    """Add columns to ``frame`` from ``fetcher``; swallow API errors.

    Optional series (cross-border flows, generation per type) should
    never fail the whole load — we add what we can.
    """
    try:
        series = fetcher(zone, start=start, end=end)
    except (AttributeError, KeyError, ValueError, RuntimeError):
        return
    if not isinstance(series, pd.DataFrame):
        return
    series = series.resample("h").mean()
    for col in series.columns:
        frame[f"{prefix}{col}"] = series[col]


def _synthesise(
    bidding_zone: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Generate plausible hourly market data without the API.

    Adds a daily peak/off-peak shape, a weekend dip, and a winter
    bump, plus Gaussian noise.
    """
    rng = np.random.default_rng(42)
    # PyPSA requires tz-naive timestamps; strip tz for compatibility.
    idx = pd.date_range(
        start.tz_localize(None) if start.tzinfo else start,
        end.tz_localize(None) if end.tzinfo else end,
        freq="h",
    )
    n = len(idx)

    hour = np.array([t.hour for t in idx])
    weekday = np.array([t.weekday() for t in idx])
    month = np.array([t.month for t in idx])

    daily = np.where(
        (hour >= _PEAK_HOUR_LO) & (hour <= _PEAK_HOUR_HI),
        _PEAK_PRICE_BUMP,
        _OFF_PEAK_PRICE_DIP,
    )
    weekly = np.where(weekday < 5, 0.0, _WEEKEND_PRICE_DIP)
    seasonal = np.where(
        (month >= 11) | (month <= 2),
        _WINTER_PRICE_BUMP,
        _SUMMER_PRICE_DIP,
    )
    noise = rng.normal(0.0, _PRICE_NOISE_STD, n)

    prices = np.clip(
        _DEFAULT_BASE_PRICE + daily + weekly + seasonal + noise,
        0.1,
        500.0,
    )

    return pd.DataFrame(
        {
            f"price_{bidding_zone}": prices,
            f"flow_{bidding_zone}_export": rng.normal(0.0, 200.0, n),
        },
        index=idx,
    )


__all__ = ["load_entso_e"]
