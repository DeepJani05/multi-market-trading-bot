"""Risk Agent.

Sits between signal generation and execution. Responsibilities:
    - Position sizing (capped Kelly fraction)
    - Per-trade risk cap (max % of equity at risk)
    - Portfolio drawdown circuit breaker (flatten everything at -8%)
    - Asset-class exposure caps

Stateful: tracks current equity, high-water mark, open positions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    max_position_pct: float = 0.02        # max 2% of equity at risk per trade
    max_kelly_fraction: float = 0.25      # cap Kelly at 25%
    drawdown_kill_pct: float = 0.08       # flatten everything at 8% drawdown
    max_exposure_per_class_pct: float = 0.40
    min_signal_probability: float = 0.55


@dataclass
class Position:
    symbol: str
    asset_class: str            # "crypto" | "equity" | "forex"
    quantity: float
    entry_price: float
    direction: str              # "long" or "short"
    opened_at: datetime


@dataclass
class TradeDecision:
    """The output of the risk agent — either an order or a vetoed signal."""

    approve: bool
    symbol: str
    direction: str
    quantity: float
    reason: str = ""


@dataclass
class Portfolio:
    starting_equity: float = 10_000.0
    equity: float = 10_000.0
    high_water_mark: float = 10_000.0
    positions: dict[str, Position] = field(default_factory=dict)

    @property
    def drawdown(self) -> float:
        return (self.equity - self.high_water_mark) / self.high_water_mark

    def exposure_for_class(self, asset_class: str) -> float:
        total = sum(
            abs(p.quantity * p.entry_price)
            for p in self.positions.values()
            if p.asset_class == asset_class
        )
        return total / self.equity if self.equity else 0.0

    def mark_to_market(self, prices: dict[str, float]) -> None:
        """Refresh equity given current prices."""
        unrealized = 0.0
        for p in self.positions.values():
            if p.symbol in prices:
                sign = 1 if p.direction == "long" else -1
                unrealized += sign * p.quantity * (prices[p.symbol] - p.entry_price)
        self.equity = self.starting_equity + unrealized
        if self.equity > self.high_water_mark:
            self.high_water_mark = self.equity


class RiskAgent:
    """Stateful risk gateway between signals and execution."""

    def __init__(self, portfolio: Portfolio, config: RiskConfig | None = None) -> None:
        self.portfolio = portfolio
        self.config = config or RiskConfig()
        self._kill_switch_active = False

    # ----------------------------------------------------------- circuit breaker

    def check_circuit_breaker(self) -> bool:
        """If drawdown breaches the kill threshold, latch the switch on."""
        if self.portfolio.drawdown <= -self.config.drawdown_kill_pct:
            if not self._kill_switch_active:
                logger.warning(
                    "risk.kill_switch_engaged",
                    extra={"drawdown": self.portfolio.drawdown},
                )
            self._kill_switch_active = True
        return self._kill_switch_active

    # --------------------------------------------------------------- sizing

    def _kelly_size(self, probability: float, win_loss_ratio: float = 1.0) -> float:
        """Kelly fraction = p - (1-p)/win_loss_ratio. Capped, clipped at 0."""
        f = probability - (1 - probability) / win_loss_ratio
        f = max(0.0, min(f, self.config.max_kelly_fraction))
        return f

    def _max_quantity(self, price: float, probability: float) -> float:
        """How many units we can buy without breaching the trade cap."""
        kelly = self._kelly_size(probability)
        dollars_at_risk = min(
            kelly * self.portfolio.equity,
            self.config.max_position_pct * self.portfolio.equity,
        )
        return dollars_at_risk / price if price > 0 else 0.0

    # --------------------------------------------------------------- evaluate

    def evaluate(
        self,
        symbol: str,
        asset_class: str,
        direction: str,
        probability: float,
        price: float,
    ) -> TradeDecision:
        """Decide whether to act on a signal and at what size."""
        if self.check_circuit_breaker():
            return TradeDecision(False, symbol, direction, 0.0, "kill switch active")

        if direction == "flat":
            return TradeDecision(False, symbol, direction, 0.0, "flat signal")

        if probability < self.config.min_signal_probability:
            return TradeDecision(
                False, symbol, direction, 0.0,
                f"probability {probability:.2f} below min {self.config.min_signal_probability}",
            )

        if self.portfolio.exposure_for_class(asset_class) >= self.config.max_exposure_per_class_pct:
            return TradeDecision(
                False, symbol, direction, 0.0,
                f"exposure cap reached for {asset_class}",
            )

        qty = self._max_quantity(price, probability)
        if qty <= 0:
            return TradeDecision(False, symbol, direction, 0.0, "sized to zero")

        return TradeDecision(True, symbol, direction, qty, "ok")

    # --------------------------------------------------------------- bookkeeping

    def record_fill(
        self,
        symbol: str,
        asset_class: str,
        quantity: float,
        price: float,
        direction: str,
    ) -> None:
        self.portfolio.positions[symbol] = Position(
            symbol=symbol,
            asset_class=asset_class,
            quantity=quantity,
            entry_price=price,
            direction=direction,
            opened_at=datetime.now(timezone.utc),
        )

    def close_position(self, symbol: str, exit_price: float) -> float:
        """Close a position and return the realized P&L."""
        pos = self.portfolio.positions.pop(symbol, None)
        if not pos:
            return 0.0
        sign = 1 if pos.direction == "long" else -1
        pnl = sign * pos.quantity * (exit_price - pos.entry_price)
        self.portfolio.starting_equity += pnl  # realized
        return pnl
