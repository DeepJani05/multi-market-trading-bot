"""Signal Agent.

Ensembles two signal sources:
    - XGBoost classifier on the current feature snapshot (fast, non-linear).
    - LSTM on the last N feature snapshots (captures temporal structure).

The two predictions are blended with learned weights (default 0.6 XGB
+ 0.4 LSTM) into a single direction-class probability. The agent emits
`Signal` events for downstream risk + execution.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

Direction = Literal["long", "short", "flat"]


@dataclass
class Signal:
    """A single trading signal for one symbol at one bar."""

    symbol: str
    timestamp: pd.Timestamp
    direction: Direction
    probability: float
    score: float  # signed strength in [-1, 1]
    features_used: int


class SignalAgent:
    """Loads the trained XGBoost + LSTM models and produces blended signals.

    Both models are expected to be trained offline (see `data/backtester.py`
    for the XGBoost training loop). In production, model files live in
    blob storage and are pulled at agent startup.
    """

    def __init__(
        self,
        xgb_path: Path,
        lstm_path: Path | None = None,
        sequence_length: int = 30,
        xgb_weight: float = 0.6,
        threshold: float = 0.55,
    ) -> None:
        self.sequence_length = sequence_length
        self.xgb_weight = xgb_weight
        self.lstm_weight = 1.0 - xgb_weight
        self.threshold = threshold

        self.xgb_model = joblib.load(xgb_path)
        self.lstm_model = None
        if lstm_path and lstm_path.exists():
            try:
                import tensorflow as tf  # lazy import — heavy

                self.lstm_model = tf.keras.models.load_model(lstm_path)
            except Exception:  # pragma: no cover - tolerate missing TF
                logger.warning("lstm load failed; running XGB-only")

    # ------------------------------------------------------------------ infer

    def _xgb_probs(self, X: pd.DataFrame) -> np.ndarray:
        """3-class probability matrix shape (n, 3) -> [short, flat, long]."""
        return self.xgb_model.predict_proba(X)

    def _lstm_probs(self, X: pd.DataFrame) -> np.ndarray | None:
        if self.lstm_model is None or len(X) < self.sequence_length:
            return None
        seq = X.iloc[-self.sequence_length:].values
        seq = seq.reshape(1, self.sequence_length, X.shape[1])
        # Model output is expected to be softmax over [short, flat, long]
        out = self.lstm_model.predict(seq, verbose=0)
        # Broadcast last-step prediction to all rows for shape compat
        return np.tile(out, (len(X), 1))

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Return blended class probabilities of shape (n, 3)."""
        xgb_p = self._xgb_probs(features)
        lstm_p = self._lstm_probs(features)
        if lstm_p is None:
            return xgb_p
        return self.xgb_weight * xgb_p + self.lstm_weight * lstm_p

    # ------------------------------------------------------------------ signal

    def make_signal(self, symbol: str, features: pd.DataFrame) -> Signal:
        """Produce the most recent signal for a symbol."""
        probs = self.predict(features)
        last = probs[-1]
        p_short, p_flat, p_long = last
        score = float(p_long - p_short)

        if p_long >= self.threshold and p_long > p_short:
            direction: Direction = "long"
            probability = float(p_long)
        elif p_short >= self.threshold and p_short > p_long:
            direction = "short"
            probability = float(p_short)
        else:
            direction = "flat"
            probability = float(p_flat)

        sig = Signal(
            symbol=symbol,
            timestamp=features.index[-1],
            direction=direction,
            probability=probability,
            score=score,
            features_used=features.shape[1],
        )
        logger.info(
            "signal.emitted",
            extra={
                "symbol": symbol,
                "direction": direction,
                "probability": probability,
                "score": score,
            },
        )
        return sig
