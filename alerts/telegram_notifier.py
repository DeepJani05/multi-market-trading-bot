"""Telegram notifier — fire alerts on fills, errors, risk breaches.

Tiny on purpose. We don't pull in `python-telegram-bot` for this — the
HTTPS sendMessage endpoint is two lines.
"""
from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Sends one-way notifications to a Telegram chat."""

    def __init__(self, bot_token: str | None = None, chat_id: str | None = None) -> None:
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self._enabled = bool(self.bot_token and self.chat_id)
        if not self._enabled:
            logger.warning("telegram notifier disabled (missing token/chat_id)")

    def send(self, message: str, parse_mode: str = "Markdown") -> bool:
        """Returns True on a successful 200 from Telegram."""
        if not self._enabled:
            return False
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": message, "parse_mode": parse_mode}
        try:
            r = httpx.post(url, json=payload, timeout=10.0)
            r.raise_for_status()
            return True
        except Exception as e:
            logger.exception("telegram.send_failed: %s", e)
            return False

    # ---------- convenience formatters ----------

    def fill(self, symbol: str, side: str, qty: float, price: float, venue: str) -> bool:
        msg = (
            f"✅ *Fill* `{venue}`\n"
            f"`{symbol}` {side.upper()} {qty:g} @ {price:,.4f}"
        )
        return self.send(msg)

    def risk_breach(self, reason: str, drawdown: float) -> bool:
        msg = f"⚠️ *Risk breach*\n{reason}\nDrawdown: {drawdown:.2%}"
        return self.send(msg)

    def error(self, where: str, detail: str) -> bool:
        msg = f"❌ *Error* `{where}`\n```\n{detail[:500]}\n```"
        return self.send(msg)
