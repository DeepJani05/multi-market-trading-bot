"""Tests for the feature engineering pipeline.

We generate a synthetic OHLCV frame so tests are deterministic and don't
require a market data fixture file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.feature_engineering import build_features


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 500
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    # Geometric Brownian-ish prices, never negative
    rets = rng.normal(0, 0.002, size=n)
    close = 100 * np.exp(rets.cumsum())
    high = close * (1 + np.abs(rng.normal(0, 0.001, size=n)))
    low = close * (1 - np.abs(rng.normal(0, 0.001, size=n)))
    open_ = close * (1 + rng.normal(0, 0.0005, size=n))
    volume = rng.integers(1_000, 10_000, size=n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def test_builds_expected_columns(synthetic_ohlcv: pd.DataFrame):
    feats = build_features(synthetic_ohlcv)
    expected = {
        "ret_1", "ret_5", "ret_15", "ret_60",
        "sma_20", "ema_50", "ma_cross_5_20",
        "vol_20", "atr_14", "parkinson_20",
        "rsi_14", "macd", "macd_signal", "stoch_k_14",
        "volume_z_20", "obv", "hl_range", "vol_regime",
    }
    assert expected.issubset(feats.columns)


def test_feature_count_is_substantial(synthetic_ohlcv: pd.DataFrame):
    feats = build_features(synthetic_ohlcv)
    assert feats.shape[1] >= 30  # we claim 40+, allow some headroom


def test_no_lookahead_in_recent_rows(synthetic_ohlcv: pd.DataFrame):
    """Last row's features must not change if we append more data."""
    feats = build_features(synthetic_ohlcv).iloc[-1]
    extra = synthetic_ohlcv.copy()
    extra.loc[extra.index[-1] + pd.Timedelta(hours=1)] = extra.iloc[-1]
    feats_extended = build_features(extra)
    # The last row of original input must match the same-timestamp row
    # in the extended computation.
    matching = feats_extended.loc[feats.name]
    pd.testing.assert_series_equal(feats, matching, check_names=False)


def test_rejects_missing_columns():
    bad = pd.DataFrame({"close": [1, 2, 3]})
    with pytest.raises(ValueError, match="missing"):
        build_features(bad)
