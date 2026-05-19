"""Execution Agent.

Broker-agnostic order router. Implements the adapter pattern: one
`BrokerInterface` Protocol, three concrete adapters (Alpaca / OANDA /
Binance). The rest of the system never knows which venue an order
landed at.

Note: this file shows the structure and the contracts. Real broker
SDKs are imported lazily so the test suite can run without them.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ types


@dataclass
class Order:
    symbol: str
    quantity: float
    side: str             # "buy" | "sell"
    order_type: str = "market"
    limit_price: float | None = None
    client_order_id: str | None = None


@dataclass
class Fill:
    symbol: str
    quantity: float
    price: float
    side: str
    filled_at: datetime
    broker_order_id: str
    venue: str


# --------------------------------------------------------------- broker iface


class BrokerInterface(Protocol):
    """Every broker adapter must implement this contract."""

    venue: str

    def submit_order(self, order: Order) -> Fill: ...
    def cancel_order(self, broker_order_id: str) -> bool: ...
    def get_positions(self) -> dict[str, float]: ...
    def get_account_equity(self) -> float: ...


# --------------------------------------------------------------- adapters


class AlpacaBroker:
    """US equities via Alpaca.

    Real implementation calls `alpaca-py`. The stub below preserves the
    interface contract for tests without requiring the SDK.
    """

    venue = "alpaca"

    def __init__(self, api_key: str | None = None, secret: str | None = None, paper: bool = True):
        self.api_key = api_key or os.getenv("ALPACA_KEY", "")
        self.secret = secret or os.getenv("ALPACA_SECRET", "")
        self.paper = paper
        self._client = None  # lazy

    def _ensure_client(self):  # pragma: no cover - SDK shim
        if self._client is None:
            try:
                from alpaca.trading.client import TradingClient

                self._client = TradingClient(self.api_key, self.secret, paper=self.paper)
            except ImportError:
                raise RuntimeError("alpaca-py not installed")
        return self._client

    def submit_order(self, order: Order) -> Fill:  # pragma: no cover - external API
        client = self._ensure_client()
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        req = MarketOrderRequest(
            symbol=order.symbol,
            qty=order.quantity,
            side=OrderSide.BUY if order.side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        resp = client.submit_order(req)
        return Fill(
            symbol=order.symbol,
            quantity=float(resp.filled_qty or order.quantity),
            price=float(resp.filled_avg_price or 0.0),
            side=order.side,
            filled_at=datetime.now(timezone.utc),
            broker_order_id=str(resp.id),
            venue=self.venue,
        )

    def cancel_order(self, broker_order_id: str) -> bool:  # pragma: no cover
        self._ensure_client().cancel_order_by_id(broker_order_id)
        return True

    def get_positions(self) -> dict[str, float]:  # pragma: no cover
        return {p.symbol: float(p.qty) for p in self._ensure_client().get_all_positions()}

    def get_account_equity(self) -> float:  # pragma: no cover
        return float(self._ensure_client().get_account().equity)


class OandaBroker:
    """Forex via OANDA v20 REST."""

    venue = "oanda"

    def __init__(self, token: str | None = None, account_id: str | None = None):
        self.token = token or os.getenv("OANDA_TOKEN", "")
        self.account_id = account_id or os.getenv("OANDA_ACCOUNT_ID", "")

    def submit_order(self, order: Order) -> Fill:  # pragma: no cover
        # Real impl: POST /v3/accounts/{accountID}/orders
        raise NotImplementedError("Wire up oandapyV20 client here.")

    def cancel_order(self, broker_order_id: str) -> bool:  # pragma: no cover
        raise NotImplementedError

    def get_positions(self) -> dict[str, float]:  # pragma: no cover
        raise NotImplementedError

    def get_account_equity(self) -> float:  # pragma: no cover
        raise NotImplementedError


class BinanceBroker:
    """Crypto spot via Binance.

    Use spot-testnet for paper trading.
    """

    venue = "binance"

    def __init__(self, api_key: str | None = None, secret: str | None = None, testnet: bool = True):
        self.api_key = api_key or os.getenv("BINANCE_KEY", "")
        self.secret = secret or os.getenv("BINANCE_SECRET", "")
        self.testnet = testnet

    def submit_order(self, order: Order) -> Fill:  # pragma: no cover
        # Real impl uses `python-binance` client
        raise NotImplementedError("Wire up python-binance client here.")

    def cancel_order(self, broker_order_id: str) -> bool:  # pragma: no cover
        raise NotImplementedError

    def get_positions(self) -> dict[str, float]:  # pragma: no cover
        raise NotImplementedError

    def get_account_equity(self) -> float:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------- router


class ExecutionRouter:
    """Selects the right broker per symbol and routes the order."""

    def __init__(self, brokers: dict[str, BrokerInterface]) -> None:
        # brokers keyed by asset class: {"equity": AlpacaBroker(), ...}
        self.brokers = brokers
        self._fills: list[Fill] = []

    def submit(self, asset_class: str, order: Order) -> Fill:
        broker = self.brokers.get(asset_class)
        if broker is None:
            raise ValueError(f"no broker configured for {asset_class}")
        logger.info(
            "order.submit",
            extra={
                "venue": broker.venue,
                "symbol": order.symbol,
                "side": order.side,
                "qty": order.quantity,
            },
        )
        fill = broker.submit_order(order)
        self._fills.append(fill)
        logger.info(
            "order.filled",
            extra={
                "venue": fill.venue,
                "symbol": fill.symbol,
                "price": fill.price,
            },
        )
        return fill

    @property
    def fill_history(self) -> list[Fill]:
        return list(self._fills)


class PaperBroker:
    """Simple in-memory paper broker for local development.

    Fills at the requested limit_price, or a fixed mock price of 100.0
    if no limit is given. Useful for end-to-end smoke tests.
    """

    venue = "paper"

    def __init__(self) -> None:
        self._positions: dict[str, float] = {}
        self._equity = 100_000.0

    def submit_order(self, order: Order) -> Fill:
        price = order.limit_price if order.limit_price else 100.0
        qty = order.quantity if order.side == "buy" else -order.quantity
        self._positions[order.symbol] = self._positions.get(order.symbol, 0) + qty
        return Fill(
            symbol=order.symbol,
            quantity=order.quantity,
            price=price,
            side=order.side,
            filled_at=datetime.now(timezone.utc),
            broker_order_id=f"paper-{datetime.now(timezone.utc).timestamp()}",
            venue=self.venue,
        )

    def cancel_order(self, broker_order_id: str) -> bool:
        return True

    def get_positions(self) -> dict[str, float]:
        return dict(self._positions)

    def get_account_equity(self) -> float:
        return self._equity
