"""
kalshi_client.py — Thin wrapper around Kalshi's public REST API.

Uses the unauthenticated external endpoint for market data reads.
No API key needed for any method in this file.

Key insight: category lives on the EVENT object, not the market object.
We use GET /events?with_nested_markets=true to get both in one call.
"""

import requests
import time
from typing import Optional

import config


class KalshiClient:
    """
    REST client for Kalshi market data (public endpoints, no auth required).

    Kalshi prices are in cents (0–99). We convert to floats (0.00–0.99)
    so they read as probabilities throughout the codebase.
    """

    def __init__(self):
        self.base_url = config.KALSHI_BASE_URL
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def _get(self, endpoint: str, params: dict = None) -> dict:
        """Make a GET request; handle rate limits with simple backoff."""
        url = f"{self.base_url}{endpoint}"
        for attempt in range(3):
            try:
                resp = self.session.get(url, params=params, timeout=10)
                if resp.status_code == 429:
                    wait = 2 ** attempt
                    print(f"  Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.RequestException as e:
                if attempt == 2:
                    raise RuntimeError(f"Kalshi API error on {endpoint}: {e}")
                time.sleep(1)
        return {}

    # ─── Events (with nested markets — this is how we get category) ──────────

    def get_events_with_markets(
        self,
        status: str = "open",
        limit: int = 50,
        cursor: Optional[str] = None,
    ) -> dict:
        """
        Fetch events with their nested markets in a single call.

        Category is on the event object; volume/price data is on each market.
        """
        params = {
            "status": status,
            "limit": limit,
            "with_nested_markets": "true",
        }
        if cursor:
            params["cursor"] = cursor
        return self._get("/events", params=params)

    def get_all_markets_with_category(
        self, status: str = "open", max_pages: int = 5
    ) -> list[dict]:
        """
        Paginate through events and return a flat list of markets,
        each enriched with its category from the parent event.

        Also inspects the first response to detect the actual volume
        field name used by the API (volume_24h vs volume_24h_fp etc.)
        """
        markets = []
        cursor = None
        vol_field = None   # detected on first response

        for _ in range(max_pages):
            data = self.get_events_with_markets(
                status=status, limit=config.MARKETS_LIMIT, cursor=cursor
            )
            events = data.get("events", [])
            if not events:
                break

            for event in events:
                category = (event.get("category") or "").strip()

                for raw_market in event.get("markets", []):
                    # Detect volume field name from first market we see
                    if vol_field is None:
                        for candidate in ("volume_24h", "volume_24h_fp", "volume24h", "daily_volume"):
                            if candidate in raw_market:
                                vol_field = candidate
                                break
                        if vol_field is None:
                            vol_field = "volume_24h"  # fallback

                    raw_market["_category"] = category
                    raw_market["_vol_field"] = vol_field
                    markets.append(raw_market)

            cursor = data.get("cursor")
            if not cursor:
                break
            time.sleep(0.3)

        return markets

    # ─── Single market lookup (for Bayesian pricer) ───────────────────────────

    def get_market(self, ticker: str) -> dict:
        """Fetch a single market by ticker (used by the Bayesian pricer)."""
        data = self._get(f"/markets/{ticker}")
        return data.get("market", {})

    def get_orderbook(self, ticker: str, depth: int = 5) -> dict:
        """Fetch order book for a market."""
        data = self._get(f"/markets/{ticker}/orderbook", params={"depth": depth})
        return data.get("orderbook", {})

    # ─── Price Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def cents_to_prob(cents) -> Optional[float]:
        """
        Convert Kalshi price to probability.
        Handles both integer cents (55) and dollar strings ("0.5500").
        """
        if cents is None:
            return None
        try:
            val = float(cents)
            # If value looks like dollars (0.0–1.0), use directly
            if val <= 1.0:
                return round(val, 4)
            # Otherwise treat as cents (0–99)
            return round(val / 100.0, 4)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def prob_to_cents(prob: float) -> int:
        """Convert probability (0.0–1.0) to Kalshi cents."""
        return max(1, min(99, round(prob * 100)))

    def parse_market(self, raw: dict) -> dict:
        """
        Extract and clean fields from a raw market dict.
        Handles both cents-based and dollar-based price formats.
        """
        # Price fields — API may use yes_bid (cents int) or yes_bid_dollars (string)
        yes_bid = self.cents_to_prob(
            raw.get("yes_bid") if raw.get("yes_bid") is not None
            else raw.get("yes_bid_dollars")
        )
        yes_ask = self.cents_to_prob(
            raw.get("yes_ask") if raw.get("yes_ask") is not None
            else raw.get("yes_ask_dollars")
        )

        if yes_bid is not None and yes_ask is not None:
            mid = round((yes_bid + yes_ask) / 2, 4)
        elif yes_bid is not None:
            mid = yes_bid
        elif yes_ask is not None:
            mid = yes_ask
        else:
            mid = None

        # Volume — detect whichever field is present
        vol_field = raw.get("_vol_field", "volume_24h")
        vol_24h_raw = raw.get(vol_field) or raw.get("volume_24h") or raw.get("volume_24h_fp") or 0
        try:
            volume_24h = int(float(vol_24h_raw))
        except (TypeError, ValueError):
            volume_24h = 0

        vol_raw = raw.get("volume") or raw.get("volume_fp") or 0
        try:
            volume = int(float(vol_raw))
        except (TypeError, ValueError):
            volume = 0

        oi_raw = raw.get("open_interest") or raw.get("open_interest_fp") or 0
        try:
            open_interest = int(float(oi_raw))
        except (TypeError, ValueError):
            open_interest = 0

        return {
            "ticker":        raw.get("ticker", ""),
            "title":         raw.get("title", ""),
            "status":        raw.get("status", ""),
            "category":      raw.get("_category", raw.get("category", "")).strip(),
            "event_ticker":  raw.get("event_ticker", ""),
            "yes_bid":       yes_bid,
            "yes_ask":       yes_ask,
            "mid":           mid,
            "volume":        volume,
            "volume_24h":    volume_24h,
            "open_interest": open_interest,
            "close_time":    raw.get("close_time", ""),
        }
