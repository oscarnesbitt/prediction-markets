"""
config.py — Loads settings from environment variables (via .env file).

Copy .env.example to .env and fill in your values.
Never commit .env to git.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── Kalshi API ──────────────────────────────────────────────────────────────

KALSHI_BASE_URL = "https://external-api.kalshi.com/trade-api/v2"

KALSHI_API_KEY_ID = os.getenv("KALSHI_API_KEY_ID", "")
KALSHI_PRIVATE_KEY_PATH = os.getenv("KALSHI_PRIVATE_KEY_PATH", "kalshi_private.pem")

# ─── Google Sheets ───────────────────────────────────────────────────────────

GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_CREDENTIALS_PATH", "google_credentials.json")
SPREADSHEET_ID = os.getenv("SPREADSHEET_ID", "")
SHEET_TAB_NAME = os.getenv("SHEET_TAB_NAME", "Kalshi Markets")

# ─── Scanner Settings ────────────────────────────────────────────────────────

MARKETS_LIMIT = int(os.getenv("MARKETS_LIMIT", "50"))
MARKET_STATUS = os.getenv("MARKET_STATUS", "open")
REFRESH_INTERVAL_SECONDS = int(os.getenv("REFRESH_INTERVAL_SECONDS", "60"))

# Minimum 24h volume to include a market in the display.
# Set to 0 in .env or via --min-volume 0 flag to show everything.
MIN_VOLUME_24H = int(os.getenv("MIN_VOLUME_24H", "1"))

# Categories to include in the default filtered view.
# These are NORMALIZED values (after CATEGORY_NORMALIZE is applied in market_scanner.py).
# "climate and weather" → "climate", "science and technology" → "technology"
FOCUS_CATEGORIES = {
    "economics",
    "financials",
    "politics",
    "elections",
    "crypto",
    "climate",
    "technology",
    "companies",
    "world",
}

# ─── Bayesian Pricer ─────────────────────────────────────────────────────────

EDGE_THRESHOLD = float(os.getenv("EDGE_THRESHOLD", "0.05"))
