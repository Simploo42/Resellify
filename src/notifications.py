"""
Deal notifications via Discord and/or Telegram webhooks.

Configured under the `notifications` block in config/settings.yaml:

    notifications:
      discord_webhook_url: "https://discord.com/api/webhooks/..."
      telegram_bot_token: "123456:ABC..."
      telegram_chat_id: "987654321"
      notify_min_score: 75

A channel is active only when its credentials are present, so the notifier
degrades to a no-op when nothing is configured. Each deal is sent at most once
(tracked via DealScore.is_notified) so re-scans don't spam.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


class Notifier:
    """Fans a qualifying deal out to every configured channel."""

    def __init__(self, config: dict):
        notif = (config or {}).get("notifications", {}) or {}
        self.discord_url: str = (notif.get("discord_webhook_url") or "").strip()
        self.telegram_token: str = (notif.get("telegram_bot_token") or "").strip()
        self.telegram_chat_id: str = str(notif.get("telegram_chat_id") or "").strip()
        self.min_score: float = float(notif.get("notify_min_score", 75))
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def enabled(self) -> bool:
        """True when at least one channel is fully configured."""
        return bool(self.discord_url or (self.telegram_token and self.telegram_chat_id))

    def qualifies(self, total_score: float) -> bool:
        return self.enabled and total_score >= self.min_score

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ── Message formatting ──────────────────────────────────────────────────

    @staticmethod
    def _summary(deal: dict[str, Any]) -> str:
        title = deal.get("title", "Unknown item")
        score = round(float(deal.get("total_score", 0)))
        asking = round(float(deal.get("asking_price_ron", 0)))
        value = round(float(deal.get("estimated_value_ron", 0)))
        profit = round(float(deal.get("estimated_profit_ron", 0)))
        pct = round(float(deal.get("profit_percent", 0)))
        platform = (deal.get("platform") or "").upper()
        location = deal.get("location") or ""
        loc = f" · {location}" if location else ""
        return (
            f"{title}\n"
            f"Score {score} · {platform}{loc}\n"
            f"Asking {asking} RON → est. {value} RON "
            f"(+{profit} RON / {pct}%)"
        )

    # ── Channels ────────────────────────────────────────────────────────────

    async def _send_discord(self, deal: dict[str, Any]) -> bool:
        body = self._summary(deal)
        url = deal.get("url")
        content = f"**New deal**\n{body}"
        if url:
            content += f"\n{url}"
        try:
            client = await self._get_client()
            resp = await client.post(self.discord_url, json={"content": content})
            resp.raise_for_status()
            return True
        except Exception as e:
            logger.warning(f"[Notify] Discord send failed: {e}")
            return False

    async def _send_telegram(self, deal: dict[str, Any]) -> bool:
        body = self._summary(deal)
        url = deal.get("url")
        text = f"\U0001f3f7 New deal\n{body}"
        if url:
            text += f"\n{url}"
        api = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        try:
            client = await self._get_client()
            resp = await client.post(api, json={
                "chat_id": self.telegram_chat_id,
                "text": text,
                "disable_web_page_preview": False,
            })
            resp.raise_for_status()
            return True
        except Exception as e:
            logger.warning(f"[Notify] Telegram send failed: {e}")
            return False

    async def notify(self, deal: dict[str, Any]) -> bool:
        """Send `deal` to every configured channel. Returns True if any
        channel accepted it. Caller is responsible for the score gate via
        `qualifies()` and for dedup persistence."""
        if not self.enabled:
            return False
        sent = False
        if self.discord_url:
            sent = await self._send_discord(deal) or sent
        if self.telegram_token and self.telegram_chat_id:
            sent = await self._send_telegram(deal) or sent
        return sent
