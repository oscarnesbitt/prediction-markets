"""
run_all.py — Main entry point for the Kalshi Prediction Markets toolkit.

Runs:
  1. Market scanner — fetch and display live markets
  2. Bayesian pricer demo — show the fair-value engine in action
  3. (Optional) Google Sheets sync

Usage:
    python run_all.py              # scanner + Bayes demo
    python run_all.py --sheets     # also sync to Google Sheets
    python run_all.py --demo-only  # just the Bayes demo (no API calls)
    python run_all.py --scan-only  # just the market scanner
"""

import sys
from colorama import init, Fore, Style

init(autoreset=True)


def print_banner():
    print(f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════╗
║       KALSHI PREDICTION MARKETS TRADER — PORTFOLIO PROJECT   ║
║       Built for DRW Prediction Markets Trader Application    ║
╚══════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
""")


def run_scanner():
    print(f"\n{Fore.GREEN}[1/2] MARKET SCANNER{Style.RESET_ALL}")
    print("─" * 50)
    from market_scanner import fetch_markets, print_market_table
    markets = fetch_markets()
    print_market_table(markets)
    return markets


def run_bayes_demo():
    print(f"\n{Fore.GREEN}[2/2] BAYESIAN FAIR-VALUE ESTIMATOR — DEMO{Style.RESET_ALL}")
    print("─" * 50)
    from bayesian_pricer import run_demo
    run_demo()


def run_sheets_sync():
    print(f"\n{Fore.GREEN}[+] GOOGLE SHEETS SYNC{Style.RESET_ALL}")
    print("─" * 50)
    try:
        from sheets_sync import run_once
        run_once()
    except RuntimeError as e:
        print(f"{Fore.YELLOW}Sheets sync skipped: {e}")
        print("See README.md for Google Sheets setup instructions.{Style.RESET_ALL}")


if __name__ == "__main__":
    args = sys.argv[1:]
    print_banner()

    if "--demo-only" in args:
        run_bayes_demo()
    elif "--scan-only" in args:
        run_scanner()
    else:
        run_scanner()
        run_bayes_demo()

        if "--sheets" in args:
            run_sheets_sync()

    print(f"\n{Fore.CYAN}Done. See README.md for next steps and extension ideas.{Style.RESET_ALL}\n")
