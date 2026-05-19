# Multi-Market Algorithmic Trading Bot

> An event-driven, multi-agent trading system that runs the same signal-generation pipeline across **crypto, US equities, and forex** — XGBoost + LSTM ensemble, live execution via Alpaca / OANDA / Binance, Streamlit P&L dashboard, Telegram alerts.

[![CI](https://github.com/<your-handle>/multi-market-trading-bot/actions/workflows/ci.yml/badge.svg)](./.github/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)

> ⚠️ **Educational use only.** Live trading carries the risk of substantial financial loss. This codebase is designed for paper-trading and research. Run with real money at your own risk and only after you understand every component.

---

## 1. The Business Problem

Most retail algo traders run a single bot, on a single asset class, with a single broker. That setup hits three walls fast:

- **Asset-class concentration risk.** Crypto can rip while equities chop. A bot that only knows one market sits idle exactly when the other is paying.
- **Code duplication.** Three brokers, three notebooks, three bespoke risk modules. Bugs hide in the gaps.
- **No observability.** You find out a fill went bad when you check Discord at 11 PM. By then it's a $400 mistake.

The opportunity: **one signal engine, three execution venues, one set of eyes**. Trade where the edge is biggest right now, not just where you happened to wire up a broker first.

## 2. Why I Built This

I wanted a single project that forced me to be honest about three things I care about as an engineer:

1. **Real systems, not notebooks.** Event-driven architecture, message-passing between agents, graceful degradation when an API rate-limits you mid-trade.
2. **ML that's accountable.** Walk-forward back-testing, feature store, deterministic seeds. Not "I tuned until the equity curve looked great."
3. **Ops as a first-class concern.** If the bot can't tell me what it did and why, it doesn't matter how good the model is.

This bot is the product of treating personal trading as a software engineering problem — not the other way around.

## 3. What It Does

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Market Data   │ →  │  Feature Agent  │ →  │  Signal Agent   │
│  Agent (WS×3)   │    │  (40+ features) │    │ (XGBoost+LSTM)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                              ┌────────────────────────┘
                              ▼
                       ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
                       │ Risk Agent   │ →  │   Execution  │ →  │  Telegram /  │
                       │ (sizing,     │    │   Agent      │    │  Streamlit   │
                       │  drawdown)   │    │ (3 brokers)  │    │  Dashboard   │
                       └──────────────┘    └──────────────┘    └──────────────┘
```

**Signal engine.** XGBoost trained on 40+ engineered features (rolling stats, momentum, lags, volatility regimes) for direction-class probability. LSTM on the same features for short-horizon return forecasts. The two are ensembled with learned weights per asset class.

**Risk module.** Kelly-fraction position sizing capped at 2% per trade, portfolio-level drawdown circuit breaker at 8%, asset-class exposure caps.

**Execution.** Adapter pattern: one `BrokerInterface`, three concrete clients (Alpaca for US equities, OANDA for forex, Binance for crypto). Same order types, same fill events, same audit log.

**Observability.** Streamlit dashboard shows live P&L, open positions, trade history, and feature contribution per signal. Telegram alerts on every fill, error, or risk-breach.

## 4. Repo Layout

```
multi-market-trading-bot/
├── agents/
│   ├── __init__.py
│   ├── market_data_agent.py    # WS streams from 3 venues
│   ├── feature_agent.py        # 40+ rolling features
│   ├── signal_agent.py         # XGBoost + LSTM ensemble
│   ├── risk_agent.py           # sizing + circuit breakers
│   └── execution_agent.py      # broker-agnostic order router
├── strategies/
│   ├── base.py
│   ├── momentum_breakout.py
│   └── mean_reversion.py
├── data/
│   ├── feature_engineering.py
│   └── backtester.py           # walk-forward back-test engine
├── dashboard/
│   └── app.py                  # Streamlit dashboard
├── alerts/
│   └── telegram_notifier.py
├── tests/
├── .github/workflows/ci.yml
├── config.yaml
├── requirements.txt
├── Dockerfile
└── README.md
```

## 5. Quickstart

```bash
git clone https://github.com/<your-handle>/multi-market-trading-bot.git
cd multi-market-trading-bot

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Fill in: ALPACA_KEY, OANDA_TOKEN, BINANCE_KEY, TELEGRAM_BOT_TOKEN

# 1) Run a walk-forward back-test to validate the strategy
python -m data.backtester --asset BTC/USDT --start 2022-01-01 --end 2024-06-30

# 2) Launch the live bot (paper-trading by default)
python -m agents.execution_agent --config config.yaml --mode paper

# 3) Watch the dashboard
streamlit run dashboard/app.py
```

## 6. Architecture Notes

**Why an ensemble?** XGBoost captures non-linear feature interactions and runs fast; LSTM captures temporal patterns XGBoost can't see (e.g., shape of the last 20 candles). Neither alone consistently beat the ensemble in our walk-forward tests.

**Why walk-forward, not k-fold?** Time series leak. Walk-forward — train on `[t-N, t]`, test on `[t, t+1]`, roll — is the only honest cross-validation for trading.

**Why the adapter pattern for brokers?** So I can swap Alpaca for Tradier without touching the signal or risk agents. Every broker speaks one internal language: `Order`, `Fill`, `Position`.

**Why circuit breakers?** Models break. The market regimes you trained on go away. A drawdown circuit breaker that flattens every position at -8% is the difference between a bad month and a blown account.

## 7. What's Honest About This Codebase

- Back-test results are not future returns. Slippage and fees in the back-tester are conservative estimates, not guarantees.
- The LSTM is small (2 layers, 64 units). On purpose — bigger models overfit on financial data.
- I do not claim Sharpe > 2. The point is the *system*, not a strategy you can lift off this repo and print money with.

## 8. Roadmap
- [ ] Add options Greeks support (currently spot only)
- [ ] Reinforcement-learning agent for sizing
- [ ] Replace Streamlit with a proper React dashboard
- [ ] Multi-account portfolio aggregation

## 9. License
MIT
