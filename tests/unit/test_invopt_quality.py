"""Unit tests for pio.assess_data_quality.

Covers the user-facing quality report: missing values, negative prices,
coverage gaps, spike detection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pypsa_invopt import assess_data_quality
from pypsa_invopt.data.quality import DataQualityReport


def _clean_obs() -> pd.DataFrame:
    idx = pd.date_range("2025-06-15", periods=24, freq="h")
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "price_b1": 40.0 + rng.normal(0, 2.0, 24),
        "price_b2": 45.0 + rng.normal(0, 2.0, 24),
    }, index=idx)


def test_assess_clean_data_returns_empty_issues():
    rep = assess_data_quality(_clean_obs())
    assert isinstance(rep, DataQualityReport)
    assert rep.n_missing == 0
    assert rep.n_negative_prices == 0
    assert rep.coverage_pct == 100.0
    assert rep.issues == []
    assert rep.quality_score == 1.0


def test_assess_flags_missing_values():
    obs = _clean_obs()
    obs.iloc[5, 0] = float("nan")
    obs.iloc[10, 1] = float("nan")
    rep = assess_data_quality(obs)
    assert rep.n_missing == 2
    assert any("missing" in issue.lower() for issue in rep.issues)


def test_assess_flags_negative_prices():
    obs = _clean_obs()
    obs.iloc[0, 0] = -15.0   # negative price (real on windy nights)
    obs.iloc[12, 1] = -5.0
    rep = assess_data_quality(obs)
    assert rep.n_negative_prices == 2


def test_assess_flags_coverage_gap():
    # Skip several hours to drop coverage below the 95% threshold.
    skipped_hours = {3, 6, 9, 12, 15, 18}
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2025-06-15 00:00") + pd.Timedelta(hours=h)
         for h in range(24) if h not in skipped_hours],
    )
    obs = pd.DataFrame(
        {"price_b1": np.arange(len(idx), dtype=float)}, index=idx,
    )
    rep = assess_data_quality(obs, expected_freq="h")
    assert rep.coverage_pct < 100.0
    # 18/24 = 75% coverage → must trigger the < 95% issue path
    assert rep.coverage_pct < 95.0
    assert any("coverage" in issue.lower() for issue in rep.issues)


def test_assess_handles_empty_dataframe():
    rep = assess_data_quality(pd.DataFrame())
    assert rep.n_observations == 0
    assert rep.quality_score == 0.0
    assert any("empty" in issue.lower() for issue in rep.issues)
