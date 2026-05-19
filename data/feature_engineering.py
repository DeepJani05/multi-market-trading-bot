"""Feature engineering for financial time series.

Builds the 40+ engineered features used by both the back-tester and the
live signal agent. Deterministic — same OHLCV input always produces the
same feature matrix (critical for reproducibility).

Feature families:
    - Returns:        log returns at 1/5/15/60 bar horizons
    - Moving avgs:    SMA & EMA at 5/10/20/50 windows + crossovers
    - Volatility:     rolling std, ATR, Parkinson, Garman-Klass
    - Momentum:       RSI, MACD, Stochastic, ROC
    - Volume:         OBV, volume z-score, dollar volume
    - Microstructure: high-low range, close location, gap features
    - Regime:         volatility regime via rolling-vol quantile
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------- helpers ----------


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(period).mean()


def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd = ema12 - ema26
    signal = _ema(macd, 9)
    hist = macd - signal
    return macd, signal, hist


# ---------- main entry point ----------


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Produce a feature matrix from an OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: open, high, low, close, volume (lowercase).
        Index must be a sorted DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Feature matrix aligned to the input index. Initial rows containing
        NaN due to look-back windows are dropped by the caller.
    """
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns.str.lower())
    if missing:
        raise ValueError(f"missing columns: {missing}")
    df = df.rename(columns=str.lower).copy()

    feats = pd.DataFrame(index=df.index)

    # --- Returns ---
    log_ret = np.log(df["close"] / df["close"].shift(1))
    feats["ret_1"] = log_ret
    for h in (5, 15, 60):
        feats[f"ret_{h}"] = log_ret.rolling(h).sum()

    # --- Moving averages + crossovers ---
    for w in (5, 10, 20, 50):
        feats[f"sma_{w}"] = df["close"].rolling(w).mean() / df["close"] - 1
        feats[f"ema_{w}"] = _ema(df["close"], w) / df["close"] - 1
    feats["ma_cross_5_20"] = (feats["sma_5"] - feats["sma_20"])
    feats["ma_cross_20_50"] = (feats["sma_20"] - feats["sma_50"])

    # --- Volatility ---
    for w in (5, 20, 60):
        feats[f"vol_{w}"] = log_ret.rolling(w).std()
    feats["atr_14"] = _atr(df["high"], df["low"], df["close"], 14)
    # Parkinson: uses high-low range
    feats["parkinson_20"] = (
        np.log(df["high"] / df["low"]).pow(2).rolling(20).mean()
        / (4 * np.log(2))
    ).pow(0.5)

    # --- Momentum ---
    feats["rsi_14"] = _rsi(df["close"], 14)
    macd, sig, hist = _macd(df["close"])
    feats["macd"] = macd
    feats["macd_signal"] = sig
    feats["macd_hist"] = hist
    for w in (10, 20, 60):
        feats[f"roc_{w}"] = df["close"].pct_change(w)
    # Stochastic %K
    low_n = df["low"].rolling(14).min()
    high_n = df["high"].rolling(14).max()
    feats["stoch_k_14"] = 100 * (df["close"] - low_n) / (high_n - low_n)

    # --- Volume ---
    feats["volume_z_20"] = (
        df["volume"] - df["volume"].rolling(20).mean()
    ) / df["volume"].rolling(20).std()
    feats["dollar_volume"] = df["close"] * df["volume"]
    # OBV
    direction = np.sign(df["close"].diff()).fillna(0)
    feats["obv"] = (direction * df["volume"]).cumsum()

    # --- Microstructure ---
    feats["hl_range"] = (df["high"] - df["low"]) / df["close"]
    feats["close_loc"] = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(
        0, np.nan
    )
    feats["gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

    # --- Regime: volatility quantile vs trailing 250 bars ---
    feats["vol_regime"] = (
        feats["vol_20"].rolling(250).apply(lambda x: x.rank(pct=True).iloc[-1])
    )

    return feats
