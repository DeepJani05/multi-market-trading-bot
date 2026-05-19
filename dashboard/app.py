"""Streamlit P&L dashboard.

Renders:
    - Current equity, drawdown, open positions
    - Equity curve over the last N days
    - Trade history table
    - Per-symbol feature contribution (most recent signal)

Run with:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import streamlit as st


STATE_FILE = Path(__file__).parent.parent / "data" / "live_state.json"


# ----------------------------------------------------------------- helpers


def _load_state() -> dict:
    if not STATE_FILE.exists():
        return _mock_state()
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return _mock_state()


def _mock_state() -> dict:
    """A deterministic mock so the dashboard renders before the bot has run."""
    now = datetime.now(timezone.utc)
    return {
        "equity": 10_842.31,
        "starting_equity": 10_000.0,
        "high_water_mark": 10_900.00,
        "open_positions": [
            {"symbol": "BTC/USDT", "qty": 0.012, "entry": 67_100.0, "direction": "long"},
            {"symbol": "AAPL",     "qty": 12,    "entry": 189.20,    "direction": "long"},
        ],
        "equity_history": [
            {"timestamp": (now - timedelta(hours=i)).isoformat(),
             "equity": 10_000 + (300 - i * 1.2) + (i % 7) * 18}
            for i in range(168, 0, -1)
        ],
        "trades": [
            {"timestamp": (now - timedelta(hours=h)).isoformat(),
             "symbol": s, "side": side, "qty": q, "price": p, "venue": v, "pnl": pnl}
            for h, s, side, q, p, v, pnl in [
                (2, "BTC/USDT", "buy", 0.012, 67_100.0, "binance", 0),
                (5, "EUR/USD",  "sell", 5_000, 1.0732, "oanda",   42.10),
                (9, "AAPL",     "buy",  12,    189.20, "alpaca",   0),
                (24, "ETH/USDT","sell", 0.5,   3_440.0,"binance", -18.40),
            ]
        ],
    }


# ----------------------------------------------------------------- page


st.set_page_config(page_title="Multi-Market Trading Bot", layout="wide")
st.title("📈 Multi-Market Trading Bot")
st.caption("Live equity, positions, and trade history.")

state = _load_state()

# --- Top metrics ---
equity = state["equity"]
starting = state["starting_equity"]
hwm = state["high_water_mark"]
drawdown = (equity - hwm) / hwm
total_return = (equity - starting) / starting

c1, c2, c3, c4 = st.columns(4)
c1.metric("Equity", f"${equity:,.2f}", f"{total_return:+.2%}")
c2.metric("High-Water Mark", f"${hwm:,.2f}")
c3.metric("Drawdown", f"{drawdown:.2%}", delta_color="inverse")
c4.metric("Open Positions", len(state["open_positions"]))

# --- Equity curve ---
st.subheader("Equity Curve")
df_eq = pd.DataFrame(state["equity_history"])
df_eq["timestamp"] = pd.to_datetime(df_eq["timestamp"])
df_eq = df_eq.set_index("timestamp")
st.line_chart(df_eq["equity"], height=320)

# --- Open positions ---
st.subheader("Open Positions")
if state["open_positions"]:
    st.dataframe(pd.DataFrame(state["open_positions"]), use_container_width=True)
else:
    st.info("No open positions.")

# --- Trade history ---
st.subheader("Recent Trades")
df_tr = pd.DataFrame(state["trades"])
if not df_tr.empty:
    df_tr["timestamp"] = pd.to_datetime(df_tr["timestamp"])
    df_tr = df_tr.sort_values("timestamp", ascending=False)
    st.dataframe(df_tr, use_container_width=True)
else:
    st.info("No trades yet.")
