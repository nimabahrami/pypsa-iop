"""Data quality assessment.

Flags potential issues in market data that could compromise
calibration quality. Returns a structured report.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class DataQualityReport:
    """Quality assessment of market data for inverse OPF.

    Attributes:
        n_observations: Total number of timesteps.
        n_missing: Number of missing values (across all columns).
        n_price_spikes: Number of price spike events detected.
        n_negative_prices: Number of negative price observations.
        coverage_pct: Percentage of expected hourly observations present.
        issues: List of human-readable issue descriptions.
        quality_score: Overall quality score in [0, 1].
    """

    n_observations: int = 0
    n_missing: int = 0
    n_price_spikes: int = 0
    n_negative_prices: int = 0
    coverage_pct: float = 100.0
    issues: list[str] = field(default_factory=list)
    quality_score: float = 1.0


def assess_quality(
    data: pd.DataFrame,
    *,
    expected_freq: str = "h",
    spike_threshold: float = 3.0,
) -> DataQualityReport:
    """Assess data quality and return a structured report.

    Args:
        data: Market data DataFrame with DatetimeIndex.
        expected_freq: Expected time resolution (default hourly).
        spike_threshold: Number of std deviations for spike detection.

    Returns:
        DataQualityReport with quality metrics and issues.
    """
    report = DataQualityReport()
    report.n_observations = len(data)

    if data.empty:
        report.quality_score = 0.0
        report.issues.append("Empty dataset")
        return report

    # Missing values
    total_missing = int(data.isna().sum().sum())
    report.n_missing = total_missing
    if total_missing > 0:
        pct = 100 * total_missing / data.size
        report.issues.append(
            f"{total_missing} missing values ({pct:.1f}% of all cells)"
        )

    # Coverage
    if isinstance(data.index, pd.DatetimeIndex) and len(data) > 1:
        expected_periods = pd.date_range(
            data.index.min(), data.index.max(), freq=expected_freq
        )
        report.coverage_pct = 100 * len(data) / max(len(expected_periods), 1)
        if report.coverage_pct < 95:
            report.issues.append(
                f"Coverage is {report.coverage_pct:.1f}% "
                f"(expected ~100% at {expected_freq} frequency)"
            )

    # Price columns analysis
    price_cols = [c for c in data.columns if c.startswith("price_")]
    for col in price_cols:
        series = data[col].dropna()
        if len(series) < 2:
            continue

        # Negative prices
        n_neg = int((series < 0).sum())
        report.n_negative_prices += n_neg
        if n_neg > 0:
            report.issues.append(
                f"Column '{col}': {n_neg} negative price observations"
            )

        # Spikes
        mean = series.mean()
        std = series.std()
        if std > 0:
            spikes = (series - mean).abs() > spike_threshold * std
            n_spikes = int(spikes.sum())
            report.n_price_spikes += n_spikes
            if n_spikes > 0:
                report.issues.append(
                    f"Column '{col}': {n_spikes} price spikes "
                    f"(|price - mean| > {spike_threshold}σ)"
                )

    # Quality score: 1.0 = perfect, deductions for each issue
    score = 1.0
    if report.n_missing > 0:
        score -= min(0.3, report.n_missing / max(data.size, 1))
    if report.coverage_pct < 100:
        score -= min(0.2, (100 - report.coverage_pct) / 100)
    if report.n_price_spikes > 0:
        score -= min(0.2, report.n_price_spikes / max(report.n_observations, 1))
    if report.n_negative_prices > 0:
        score -= min(0.1, report.n_negative_prices / max(report.n_observations, 1))
    report.quality_score = max(0.0, score)

    return report


__all__ = ["DataQualityReport", "assess_quality"]
