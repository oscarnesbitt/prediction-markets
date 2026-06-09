"""
market_scanner.py — Pull live Kalshi markets and display a focused summary table.

Run directly:
    python market_scanner.py                        # macro/finance/crypto (no sports, no dead markets)
    python market_scanner.py --all                  # all categories (still filters zero-volume)
    python market_scanner.py --category elections   # single category (use exact name from breakdown)
    python market_scanner.py --min-volume 0         # show everything including zero-volume markets
"""

import sys
import pandas as pd
from tabulate import tabulate
from colorama import init, Fore, Style
from datetime import datetime, timezone

from kalshi_client import KalshiClient
import config

init(autoreset=True)

# Maps raw API category strings → display labels
CATEGORY_LABELS = {
    "economics":              "Economics",
    "financials":             "Financials",
    "politics":               "Politics",
    "crypto":                 "Crypto",
    "climate":                "Climate",
    "climate and weather":    "Climate",
    "technology":             "Technology",
    "science and technology": "Technology",
    "companies":              "Companies",
    "sports":                 "Sports",
    "entertainment":          "Entertainment",
    "elections":              "Elections",
    "social":                 "Social",
    "world":                  "World",
    "health":                 "Health",
    "transportation":         "Transportation",
    "":                       "Other",
}

# Normalize raw category strings to canonical keys for filtering.
# This handles the mismatch between what we assumed and what the API returns.
CATEGORY_NORMALIZE = {
    "climate and weather":    "climate",
    "science and technology": "technology",
}


def normalize_category(raw: str) -> str:
    """Normalize a raw API category string to a canonical lowercase key."""
    cleaned = raw.strip().lower()
    return CATEGORY_NORMALIZE.get(cleaned, cleaned)


def fetch_markets(status: str = None, max_pages: int = 5) -> list[dict]:
    """Pull and parse all markets from Kalshi, with category from parent event."""
    client = KalshiClient()
    status = status or config.MARKET_STATUS
    print(f"\n{Fore.CYAN}Fetching {status} markets from Kalshi...{Style.RESET_ALL}")
    raw_markets = client.get_all_markets_with_category(status=status, max_pages=max_pages)
    parsed = [client.parse_market(m) for m in raw_markets]
    # Normalize categories in-place
    for m in parsed:
        m["category_raw"] = m["category"]
        m["category"] = normalize_category(m["category"])
    return parsed


def filter_markets(
    markets: list[dict],
    categories: set[str] = None,
    min_volume_24h: int = None,
) -> list[dict]:
    """
    Filter markets by category and minimum 24h volume.

    - categories: normalized category strings to include. Pass empty set for all.
      Defaults to FOCUS_CATEGORIES from config.
    - min_volume_24h: drop markets below this threshold. Defaults to MIN_VOLUME_24H.
    """
    if categories is not None and len(categories) == 0:
        cat_filtered = markets
    else:
        focus = categories if categories is not None else config.FOCUS_CATEGORIES
        cat_filtered = [m for m in markets if m.get("category", "") in focus]

    min_vol = min_volume_24h if min_volume_24h is not None else config.MIN_VOLUME_24H
    return [m for m in cat_filtered if m.get("volume_24h", 0) >= min_vol]


def build_display_df(markets: list[dict]) -> pd.DataFrame:
    sorted_markets = sorted(markets, key=lambda m: m.get("volume_24h", 0), reverse=True)
    rows = []
    for m in sorted_markets:
        bid = m["yes_bid"]
        ask = m["yes_ask"]
        mid = m["mid"]
        spread = round((ask - bid) * 100, 1) if bid is not None and ask is not None else None
        cat_raw = m.get("category_raw", m.get("category", ""))
        category = CATEGORY_LABELS.get(cat_raw.strip().lower(), CATEGORY_LABELS.get(m.get("category",""), cat_raw))

        rows.append({
            "Category":  category,
            "Ticker":    m["ticker"],
            "Title":     m["title"][:58] + ("…" if len(m["title"]) > 58 else ""),
            "Bid":       f"{bid:.2f}" if bid is not None else "—",
            "Ask":       f"{ask:.2f}" if ask is not None else "—",
            "Mid":       f"{mid:.2f}" if mid is not None else "—",
            "Sprd¢":     f"{spread:.0f}" if spread is not None else "—",
            "Vol 24h":   f"{m['volume_24h']:,}",
            "OI":        f"{m['open_interest']:,}",
        })
    return pd.DataFrame(rows)


def print_market_table(markets: list[dict], label: str = "") -> None:
    if not markets:
        print(f"{Fore.RED}No markets found matching the current filters.{Style.RESET_ALL}")
        return

    df = build_display_df(markets)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    header = f"KALSHI MARKETS{' — ' + label if label else ''}  |  {len(markets)} markets  |  {timestamp}"

    print(f"\n{Fore.GREEN}{'─' * min(len(header) + 4, 110)}")
    print(f"  {header}")
    print(f"{'─' * min(len(header) + 4, 110)}{Style.RESET_ALL}\n")

    print(tabulate(df, headers="keys", tablefmt="rounded_outline", showindex=False, numalign="right"))

    cat_counts = {}
    for m in markets:
        cat_raw = m.get("category_raw", m.get("category", ""))
        label_str = CATEGORY_LABELS.get(cat_raw.strip().lower(), cat_raw)
        cat_counts[label_str] = cat_counts.get(label_str, 0) + 1
    cat_summary = "  ".join(f"{cat}: {n}" for cat, n in sorted(cat_counts.items()))

    print(f"\n{Fore.YELLOW}Prices as probabilities (0.00–1.00).  Mid = (Bid+Ask)/2.")
    print(f"Sorted by 24h volume descending.  Min 24h volume: {config.MIN_VOLUME_24H:,}")
    print(f"Categories shown: {cat_summary}{Style.RESET_ALL}\n")


def print_category_breakdown(markets: list[dict]) -> None:
    # Count by raw category so user sees exactly what the API returns
    counts = {}
    active_counts = {}
    for m in markets:
        cat = m.get("category_raw", m.get("category", "")) or "unknown"
        counts[cat] = counts.get(cat, 0) + 1
        if m.get("volume_24h", 0) > 0:
            active_counts[cat] = active_counts.get(cat, 0) + 1

    total = len(markets)
    active = sum(1 for m in markets if m.get("volume_24h", 0) > 0)

    print(f"\n{Fore.CYAN}Market breakdown ({total} fetched  |  {active} with 24h volume  |  {total - active} zero-volume):{Style.RESET_ALL}")
    for cat, n in sorted(counts.items(), key=lambda x: -x[1]):
        label = CATEGORY_LABELS.get(cat.strip().lower(), cat)
        normalized = normalize_category(cat)
        active_n = active_counts.get(cat, 0)
        bar = "█" * min(active_n, 40) + "░" * min(n - active_n, 10)
        in_focus = normalized in (config.FOCUS_CATEGORIES or set())
        marker = "" if in_focus else f"  {Fore.YELLOW}(excluded from default view){Style.RESET_ALL}"
        print(f"  {label:<24} {n:>4} total  {active_n:>4} active  {bar}{marker}")
    print()


def get_top_markets_by_volume(n: int = 10, filtered: bool = True) -> list[dict]:
    markets = fetch_markets()
    if filtered:
        markets = filter_markets(markets)
    return sorted(markets, key=lambda m: m.get("volume_24h", 0), reverse=True)[:n]


if __name__ == "__main__":
    args = sys.argv[1:]

    min_vol_override = None
    if "--min-volume" in args:
        idx = args.index("--min-volume")
        if idx + 1 < len(args):
            try:
                min_vol_override = int(args[idx + 1])
            except ValueError:
                print(f"{Fore.RED}--min-volume requires an integer, e.g. --min-volume 100{Style.RESET_ALL}")
                sys.exit(1)

    all_markets = fetch_markets()
    print_category_breakdown(all_markets)

    if "--all" in args:
        filtered = filter_markets(all_markets, categories=set(), min_volume_24h=min_vol_override)
        print_market_table(filtered, label="All Categories")

    elif "--category" in args:
        idx = args.index("--category")
        if idx + 1 < len(args):
            cat = normalize_category(args[idx + 1])
            filtered = filter_markets(all_markets, categories={cat}, min_volume_24h=min_vol_override)
            label = CATEGORY_LABELS.get(cat, cat)
            print_market_table(filtered, label=label)
        else:
            print(f"{Fore.RED}Usage: python market_scanner.py --category economics{Style.RESET_ALL}")

    else:
        filtered = filter_markets(all_markets, min_volume_24h=min_vol_override)
        print_market_table(filtered, label="Macro / Finance / Crypto / Politics / Climate / Tech")
