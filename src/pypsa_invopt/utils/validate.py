"""Input validation for pypsa-invopt.

Validates that observation DataFrames have the required columns,
consistent time index, and no missing values in critical columns.
"""
from __future__ import annotations

import pandas as pd

from pypsa_invopt.exceptions import InvoptInputError


def validate_observations(
    observations: pd.DataFrame,
    *,
    required_price_columns: list[str] | None = None,
    required_flow_columns: list[str] | None = None,
    required_dispatch_columns: list[str] | None = None,
    allow_missing: bool = False,
) -> list[str]:
    """Validate the observations DataFrame.

    Args:
        observations: Market observation data with a DatetimeIndex.
        required_price_columns: Column names that must exist for price
            data (e.g., ``['price_bus_A', 'price_bus_B']``).
        required_flow_columns: Column names for flow data.
        required_dispatch_columns: Column names for dispatch data.
        allow_missing: If ``False``, raise on any NaN in required
            columns. If ``True``, just warn.

    Returns:
        List of warning strings (empty if no issues).

    Raises:
        InvoptInputError: If required columns are missing or data
            is fundamentally invalid.
    """
    warnings: list[str] = []

    if observations.empty:
        raise InvoptInputError(
            "Observations DataFrame is empty. Provide at least one "
            "timestep of market data."
        )

    # Check index is temporal
    if not isinstance(observations.index, pd.DatetimeIndex):
        raise InvoptInputError(
            f"Observations must have a DatetimeIndex, got "
            f"{type(observations.index).__name__}. Use "
            f"observations.index = pd.to_datetime(observations.index)"
        )

    # Check required columns exist
    all_required = []
    for label, cols in [
        ("price", required_price_columns),
        ("flow", required_flow_columns),
        ("dispatch", required_dispatch_columns),
    ]:
        if cols is None:
            continue
        missing = [c for c in cols if c not in observations.columns]
        if missing:
            raise InvoptInputError(
                f"Missing {label} columns in observations: "
                f"{missing}. Available columns: "
                f"{list(observations.columns)}"
            )
        all_required.extend(cols)

    # Check for missing values
    if all_required:
        missing_mask = observations[all_required].isna()
        if missing_mask.any().any():
            n_missing = int(missing_mask.sum().sum())
            cols_with_missing = list(
                missing_mask.columns[missing_mask.any()]
            )
            msg = (
                f"{n_missing} missing values found in columns: "
                f"{cols_with_missing}"
            )
            if not allow_missing:
                raise InvoptInputError(
                    f"{msg}. Set allow_missing=True to proceed with "
                    f"interpolation, or fill missing values manually."
                )
            warnings.append(msg)

    # Check for duplicate timestamps
    if observations.index.duplicated().any():
        n_dups = int(observations.index.duplicated().sum())
        warnings.append(
            f"{n_dups} duplicate timestamps found. These will be "
            f"averaged during calibration."
        )

    # Check time series is sorted
    if not observations.index.is_monotonic_increasing:
        warnings.append(
            "Observations are not sorted by time. Sorting automatically."
        )

    # Check for suspicious values
    if required_price_columns:
        for col in required_price_columns:
            if col not in observations.columns:
                continue
            prices = observations[col].dropna()
            if len(prices) > 10:
                mean_p = prices.mean()
                std_p = prices.std()
                if std_p > 0:
                    spikes = (prices - mean_p).abs() > 3 * std_p
                    n_spikes = int(spikes.sum())
                    if n_spikes > 0:
                        warnings.append(
                            f"Column '{col}': {n_spikes} price spikes "
                            f"detected (|price - mean| > 3*std). "
                            f"Consider flagging these periods."
                        )

    return warnings


__all__ = ["validate_observations"]
