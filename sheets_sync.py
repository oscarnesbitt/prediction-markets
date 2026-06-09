"""
sheets_sync.py — Sync live Kalshi market data to a Google Sheet.

The sheet auto-updates on a configurable interval (default: 60s).

Setup required:
  1. Enable Google Sheets + Drive APIs in Google Cloud Console
  2. Create a Service Account → download JSON credentials
  3. Share your target sheet with the service account email
  4. Fill in config.py with GOOGLE_CREDENTIALS_PATH and SPREADSHEET_ID

Run directly:
    python sheets_sync.py          # runs indefinitely, updates every 60s
    python sheets_sync.py --once   # single update then exit
"""

import sys
import time
from datetime import datetime, timezone

try:
    import gspread
    from google.oauth2.service_account import Credentials
except ImportError:
    print("Missing Google auth libraries. Run: pip install gspread google-auth")
    sys.exit(1)

from market_scanner import fetch_markets, build_display_df
import config


SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
]


def get_sheet_client():
    """Authenticate and return a gspread worksheet handle."""
    try:
        creds = Credentials.from_service_account_file(
            config.GOOGLE_CREDENTIALS_PATH, scopes=SCOPES
        )
        gc = gspread.authorize(creds)
        spreadsheet = gc.open_by_key(config.SPREADSHEET_ID)

        # Get or create the target tab
        try:
            worksheet = spreadsheet.worksheet(config.SHEET_TAB_NAME)
        except gspread.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(
                title=config.SHEET_TAB_NAME, rows=200, cols=15
            )
            print(f"Created new sheet tab: '{config.SHEET_TAB_NAME}'")

        return worksheet

    except FileNotFoundError:
        raise RuntimeError(
            f"Google credentials file not found: {config.GOOGLE_CREDENTIALS_PATH}\n"
            "See README.md for setup instructions."
        )
    except Exception as e:
        raise RuntimeError(f"Google Sheets auth failed: {e}")


def write_markets_to_sheet(worksheet, markets: list[dict]) -> None:
    """Write market data to the Google Sheet, overwriting previous data."""
    if not markets:
        print("  No markets to write.")
        return

    # Sort by 24h volume descending
    sorted_markets = sorted(markets, key=lambda m: m.get("volume_24h", 0), reverse=True)
    df = build_display_df(sorted_markets)

    # Build rows: header + data + metadata footer
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    header_row = [f"Kalshi Live Markets — Last updated: {timestamp}"]
    blank_row = [""]
    col_headers = list(df.columns)
    data_rows = df.values.tolist()

    all_rows = [header_row, blank_row, col_headers] + data_rows

    # Clear and rewrite
    worksheet.clear()
    worksheet.update(all_rows, value_input_option="USER_ENTERED")

    # Bold the column header row (row 3)
    try:
        worksheet.format("A3:I3", {"textFormat": {"bold": True}})
    except Exception:
        pass  # formatting is optional; don't crash if it fails

    print(f"  ✓ Wrote {len(markets)} markets to Google Sheet at {timestamp}")


def run_once() -> None:
    """Single fetch + write cycle."""
    print("\nConnecting to Google Sheets...")
    worksheet = get_sheet_client()
    print("Fetching Kalshi markets...")
    markets = fetch_markets()
    write_markets_to_sheet(worksheet, markets)


def run_loop(interval: int = None) -> None:
    """Run fetch + write on a loop until interrupted."""
    interval = interval or config.REFRESH_INTERVAL_SECONDS
    print(f"\nStarting Kalshi → Google Sheets sync (every {interval}s). Ctrl+C to stop.\n")

    worksheet = get_sheet_client()

    while True:
        try:
            markets = fetch_markets()
            write_markets_to_sheet(worksheet, markets)
            print(f"  Next update in {interval}s...")
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\nSync stopped.")
            break
        except Exception as e:
            print(f"  Error during sync: {e}. Retrying in 30s...")
            time.sleep(30)


if __name__ == "__main__":
    if "--once" in sys.argv:
        run_once()
    else:
        run_loop()
