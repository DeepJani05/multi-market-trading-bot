"""Walk-forward back-testing engine.

Why walk-forward instead of k-fold?
    Financial time series have temporal structure — yesterday's price
    informs today's. A random shuffle leaks future info into the
    training set and gives you a Sharpe ratio that's a lie. Walk-forward
    trains on [t-N, t], tests on (t, t+H], then rolls. Honest, slow,
    irreplaceable.

Outputs:
    - per-fold metrics (sharpe, hit rate, max drawdown, total return)
    - aggregated equity curve
    - feature importance, averaged across folds
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from data.feature_engineering import build_features

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    fold_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    sharpe: float
    hit_rate: float
    max_drawdown: float
    total_return: float
    n_trades: int


@dataclass
class BacktestResult:
    folds: list[FoldResult] = field(default_factory=list)
    equity_curve: pd.Series | None = None
    feature_importance: pd.Series | None = None

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame([f.__dict__ for f in self.folds])


# ---------- metrics ----------


def sharpe(returns: pd.Series, bars_per_year: int = 252) -> float:
    if returns.std() == 0 or returns.empty:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(bars_per_year))


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())


# ---------- walk-forward driver ----------


def walk_forward_backtest(
    ohlcv: pd.DataFrame,
    label_fn: Callable[[pd.Series], pd.Series],
    train_model_fn: Callable[[pd.DataFrame, pd.Series], object],
    predict_fn: Callable[[object, pd.DataFrame], np.ndarray],
    train_window: int = 252 * 2,
    test_window: int = 21,
    step: int = 21,
    cost_per_trade_bps: float = 5.0,
) -> BacktestResult:
    """Run a walk-forward back-test.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        OHLCV data with DatetimeIndex.
    label_fn : Callable
        Function that produces classification/regression labels from
        the OHLCV close series. Typically the sign of forward returns.
    train_model_fn : Callable
        Trains and returns a model given (X_train, y_train).
    predict_fn : Callable
        Returns predictions given (model, X_test).
    train_window : int
        Bars used for each training set.
    test_window : int
        Bars used for each test set.
    step : int
        Rolling-window step size.
    cost_per_trade_bps : float
        Round-trip transaction cost in basis points.

    Returns
    -------
    BacktestResult
    """
    features = build_features(ohlcv).dropna()
    labels = label_fn(ohlcv["close"]).reindex(features.index).dropna()
    common = features.index.intersection(labels.index)
    X_all = features.loc[common]
    y_all = labels.loc[common]
    px = ohlcv["close"].loc[common]

    results: list[FoldResult] = []
    equity_pieces: list[pd.Series] = []
    importance_accum = pd.Series(0.0, index=X_all.columns)
    importance_count = 0

    start = 0
    fold_idx = 0
    cost = cost_per_trade_bps / 10_000.0
    while start + train_window + test_window <= len(X_all):
        train_idx = X_all.index[start : start + train_window]
        test_idx = X_all.index[start + train_window : start + train_window + test_window]

        X_train, y_train = X_all.loc[train_idx], y_all.loc[train_idx]
        X_test = X_all.loc[test_idx]

        model = train_model_fn(X_train, y_train)
        preds = predict_fn(model, X_test)

        # Translate predictions to positions in {-1, 0, +1}
        positions = pd.Series(np.sign(preds), index=test_idx).clip(-1, 1)
        fwd_ret = np.log(px.loc[test_idx] / px.loc[test_idx].shift(1)).fillna(0)
        # Apply position from t-1 to return at t (no look-ahead)
        strat_ret = positions.shift(1).fillna(0) * fwd_ret
        # Subtract transaction cost on position changes
        turnover = positions.diff().abs().fillna(0)
        strat_ret = strat_ret - turnover * cost

        equity = (1 + strat_ret).cumprod()
        equity_pieces.append(equity * (equity_pieces[-1].iloc[-1] if equity_pieces else 1))

        results.append(
            FoldResult(
                fold_idx=fold_idx,
                train_start=train_idx[0],
                train_end=train_idx[-1],
                test_start=test_idx[0],
                test_end=test_idx[-1],
                sharpe=sharpe(strat_ret),
                hit_rate=float((strat_ret > 0).mean()),
                max_drawdown=max_drawdown(equity),
                total_return=float(equity.iloc[-1] - 1),
                n_trades=int(turnover.sum()),
            )
        )

        # Accumulate feature importance if the model exposes it
        if hasattr(model, "feature_importances_"):
            importance_accum += pd.Series(model.feature_importances_, index=X_all.columns)
            importance_count += 1

        fold_idx += 1
        start += step

    equity_curve = pd.concat(equity_pieces) if equity_pieces else None
    importance = (
        importance_accum / importance_count if importance_count else None
    )
    return BacktestResult(
        folds=results, equity_curve=equity_curve, feature_importance=importance
    )


# ---------- CLI ----------


def _default_label_fn(close: pd.Series, horizon: int = 5) -> pd.Series:
    return np.sign(close.shift(-horizon) / close - 1)


def _default_train_xgb(X: pd.DataFrame, y: pd.Series):
    from xgboost import XGBClassifier

    # Map {-1, 0, 1} -> {0, 1, 2} for XGBoost
    mapping = {-1.0: 0, 0.0: 1, 1.0: 2}
    y_enc = y.map(mapping).fillna(1).astype(int)
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        objective="multi:softmax",
        num_class=3,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X, y_enc)
    return model


def _default_predict_xgb(model, X: pd.DataFrame) -> np.ndarray:
    raw = model.predict(X)
    return np.where(raw == 0, -1, np.where(raw == 2, 1, 0))


def main():  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to OHLCV CSV")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--train-window", type=int, default=504)
    parser.add_argument("--test-window", type=int, default=21)
    args = parser.parse_args()

    df = pd.read_csv(args.csv, parse_dates=["timestamp"], index_col="timestamp").sort_index()
    if args.start:
        df = df.loc[args.start:]
    if args.end:
        df = df.loc[:args.end]

    result = walk_forward_backtest(
        df,
        label_fn=_default_label_fn,
        train_model_fn=_default_train_xgb,
        predict_fn=_default_predict_xgb,
        train_window=args.train_window,
        test_window=args.test_window,
    )
    print(result.summary())
    print("\nFeature importance (top 15):")
    if result.feature_importance is not None:
        print(result.feature_importance.sort_values(ascending=False).head(15))


if __name__ == "__main__":
    main()
