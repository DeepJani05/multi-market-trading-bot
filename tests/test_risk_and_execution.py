"""Tests for the risk and execution layers.

We avoid live brokers and the LLM stack — the focus is on the
deterministic logic that protects capital.
"""
from __future__ import annotations

import pytest

from agents.execution_agent import ExecutionRouter, Order, PaperBroker
from agents.risk_agent import Portfolio, RiskAgent, RiskConfig


@pytest.fixture
def portfolio() -> Portfolio:
    return Portfolio(starting_equity=10_000.0)


@pytest.fixture
def risk(portfolio: Portfolio) -> RiskAgent:
    return RiskAgent(portfolio, RiskConfig(max_position_pct=0.02, min_signal_probability=0.55))


# ---------------------------------------------------------------- risk


def test_flat_signal_rejected(risk: RiskAgent):
    decision = risk.evaluate("BTC/USDT", "crypto", "flat", 0.9, 50_000)
    assert decision.approve is False
    assert "flat" in decision.reason


def test_low_probability_signal_rejected(risk: RiskAgent):
    decision = risk.evaluate("BTC/USDT", "crypto", "long", 0.50, 50_000)
    assert decision.approve is False
    assert "probability" in decision.reason


def test_kill_switch_engages_at_drawdown(risk: RiskAgent, portfolio: Portfolio):
    portfolio.high_water_mark = 10_000
    portfolio.equity = 9_100  # 9% drawdown
    decision = risk.evaluate("BTC/USDT", "crypto", "long", 0.80, 50_000)
    assert decision.approve is False
    assert "kill switch" in decision.reason


def test_position_size_respects_max_pct(risk: RiskAgent, portfolio: Portfolio):
    # at 2% cap on $10k equity, max dollars at risk = $200
    decision = risk.evaluate("AAPL", "equity", "long", 0.90, 100.0)
    assert decision.approve is True
    assert decision.quantity * 100.0 <= 0.02 * portfolio.equity + 1e-6


def test_exposure_cap_per_asset_class(risk: RiskAgent, portfolio: Portfolio):
    # Pre-load 40%+ exposure in crypto
    risk.record_fill("ETH/USDT", "crypto", 1.5, 3_000, "long")  # $4,500 / $10k = 45%
    decision = risk.evaluate("BTC/USDT", "crypto", "long", 0.80, 50_000)
    assert decision.approve is False
    assert "exposure cap" in decision.reason


# -------------------------------------------------------------- execution


def test_paper_broker_round_trip():
    broker = PaperBroker()
    fill = broker.submit_order(Order(symbol="BTC/USDT", quantity=0.1, side="buy"))
    assert fill.symbol == "BTC/USDT"
    assert fill.quantity == 0.1
    assert broker.get_positions()["BTC/USDT"] == 0.1


def test_router_dispatches_to_correct_broker():
    crypto_broker = PaperBroker()
    equity_broker = PaperBroker()
    router = ExecutionRouter({"crypto": crypto_broker, "equity": equity_broker})

    router.submit("crypto", Order(symbol="BTC/USDT", quantity=0.05, side="buy"))
    router.submit("equity", Order(symbol="AAPL", quantity=10, side="buy"))

    assert "BTC/USDT" in crypto_broker.get_positions()
    assert "AAPL" in equity_broker.get_positions()
    assert "AAPL" not in crypto_broker.get_positions()


def test_router_raises_on_unknown_asset_class():
    router = ExecutionRouter({"crypto": PaperBroker()})
    with pytest.raises(ValueError, match="no broker"):
        router.submit("commodities", Order(symbol="GC", quantity=1, side="buy"))
