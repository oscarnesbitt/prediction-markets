"""
bayesian_pricer.py — Bayesian fair-value estimation for Kalshi prediction markets.

Core idea:
    Use Bayes' Rule to update a prior probability (the market price) given
    a new piece of evidence. Compare the posterior to the current market
    price to detect potential mispricings.

    P(H|E) = P(E|H) * P(H) / [P(E|H)*P(H) + P(E|~H)*P(~H)]

    Where:
        H       = event resolves YES
        E       = new evidence (e.g. a news event, data release)
        P(H)    = prior = current Kalshi market price
        P(E|H)  = likelihood: prob of seeing this evidence IF event resolves YES
        P(E|~H) = likelihood: prob of seeing this evidence IF event resolves NO
        P(H|E)  = posterior = your updated fair-value estimate

Run interactively:
    python bayesian_pricer.py

Or import and use programmatically:
    from bayesian_pricer import BayesianPricer
    pricer = BayesianPricer()
    result = pricer.update(prior=0.22, p_e_given_h=0.75, p_e_given_not_h=0.30)
"""

from dataclasses import dataclass
from typing import Optional
from colorama import init, Fore, Style

from kalshi_client import KalshiClient
import config

init(autoreset=True)


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class Evidence:
    """A single piece of evidence to feed into the Bayesian update."""
    description: str          # Human-readable description of the evidence
    p_e_given_h: float        # P(E|H): likelihood if YES resolves
    p_e_given_not_h: float    # P(E|~H): likelihood if NO resolves

    def __post_init__(self):
        assert 0 < self.p_e_given_h <= 1, "p_e_given_h must be in (0, 1]"
        assert 0 < self.p_e_given_not_h <= 1, "p_e_given_not_h must be in (0, 1]"


@dataclass
class BayesResult:
    """Output of a single Bayesian update."""
    ticker: str
    title: str
    prior: float              # Market price (starting probability)
    posterior: float          # Updated fair-value estimate
    market_mid: float         # Live market mid-price at time of analysis
    edge: float               # posterior - market_mid (+ means market underpriced)
    evidence: list[Evidence]
    flagged: bool             # True if |edge| > EDGE_THRESHOLD
    signal: str               # "LONG", "SHORT", or "NEUTRAL"


# ─── Core Engine ──────────────────────────────────────────────────────────────

class BayesianPricer:
    """
    Bayesian probability updater for binary event markets.

    Supports sequential updates (each piece of evidence applied in turn,
    with the posterior from each step becoming the prior for the next).
    """

    def __init__(self, threshold: float = None):
        self.threshold = threshold or config.EDGE_THRESHOLD
        self.client = KalshiClient()

    def bayes_update(
        self,
        prior: float,
        p_e_given_h: float,
        p_e_given_not_h: float,
    ) -> float:
        """
        Apply a single Bayesian update.

        Args:
            prior: P(H) — current probability estimate (0.0 to 1.0)
            p_e_given_h: P(E|H) — likelihood of evidence if event is YES
            p_e_given_not_h: P(E|~H) — likelihood of evidence if event is NO

        Returns:
            Posterior probability P(H|E)
        """
        not_h = 1.0 - prior
        numerator = p_e_given_h * prior
        denominator = numerator + (p_e_given_not_h * not_h)

        if denominator == 0:
            raise ValueError("Denominator is zero — check your likelihood values.")

        return numerator / denominator

    def update_sequential(self, prior: float, evidence_list: list[Evidence]) -> float:
        """
        Apply multiple evidence updates sequentially.

        The posterior of each step becomes the prior for the next.
        This is valid when evidence pieces are conditionally independent.
        """
        prob = prior
        for ev in evidence_list:
            prob = self.bayes_update(
                prior=prob,
                p_e_given_h=ev.p_e_given_h,
                p_e_given_not_h=ev.p_e_given_not_h,
            )
        return round(prob, 4)

    def analyze_market(
        self,
        ticker: str,
        evidence_list: list[Evidence],
        manual_prior: Optional[float] = None,
    ) -> BayesResult:
        """
        Full analysis pipeline for a single market.

        1. Fetch live market data from Kalshi
        2. Use mid-price as prior (or override with manual_prior)
        3. Apply Bayesian updates for each piece of evidence
        4. Compare posterior to live market mid
        5. Flag if edge exceeds threshold
        """
        market_data = self.client.get_market(ticker)
        if not market_data:
            raise ValueError(f"Market '{ticker}' not found on Kalshi.")

        parsed = self.client.parse_market(market_data)
        market_mid = parsed["mid"]

        if market_mid is None:
            raise ValueError(f"No mid-price available for '{ticker}'.")

        prior = manual_prior if manual_prior is not None else market_mid
        posterior = self.update_sequential(prior, evidence_list)
        edge = round(posterior - market_mid, 4)

        flagged = abs(edge) >= self.threshold
        if edge > 0:
            signal = "LONG (market underpriced)"
        elif edge < 0:
            signal = "SHORT (market overpriced)"
        else:
            signal = "NEUTRAL"

        return BayesResult(
            ticker=ticker,
            title=parsed["title"],
            prior=prior,
            posterior=posterior,
            market_mid=market_mid,
            edge=edge,
            evidence=evidence_list,
            flagged=flagged,
            signal=signal,
        )

    def print_result(self, result: BayesResult) -> None:
        """Pretty-print a BayesResult to the terminal."""
        flag_str = (
            f"{Fore.RED}⚑ FLAGGED{Style.RESET_ALL}" if result.flagged
            else f"{Fore.WHITE}○ Within threshold{Style.RESET_ALL}"
        )

        edge_color = Fore.GREEN if result.edge > 0 else (Fore.RED if result.edge < 0 else Fore.WHITE)

        print(f"\n{'─'*70}")
        print(f"{Fore.CYAN}Market:    {result.ticker}{Style.RESET_ALL}")
        print(f"Title:     {result.title}")
        print(f"{'─'*70}")
        print(f"Prior (market mid):  {result.prior:.4f}  ({result.prior*100:.1f}¢)")
        for i, ev in enumerate(result.evidence, 1):
            print(f"  Evidence {i}: {ev.description}")
            print(f"    P(E|YES)={ev.p_e_given_h:.2f}  P(E|NO)={ev.p_e_given_not_h:.2f}")
        print(f"Posterior:           {result.posterior:.4f}  ({result.posterior*100:.1f}¢)")
        print(f"Live market mid:     {result.market_mid:.4f}  ({result.market_mid*100:.1f}¢)")
        print(f"Edge:                {edge_color}{result.edge:+.4f}  ({result.edge*100:+.1f}¢){Style.RESET_ALL}")
        print(f"Signal:              {result.signal}")
        print(f"Status:              {flag_str}")
        print(f"{'─'*70}")


# ─── Batch Scanner ────────────────────────────────────────────────────────────

def scan_markets_with_evidence(
    evidence_list: list[Evidence],
    n_markets: int = 20,
    status: str = "open",
) -> list[BayesResult]:
    """
    Apply the same set of evidence to the top N markets by volume.

    Useful for screening: "given this macro news, which markets move the most?"

    Returns a list of BayesResult sorted by |edge| descending.
    """
    client = KalshiClient()
    pricer = BayesianPricer()

    print(f"\nFetching top {n_markets} markets...")
    raw_markets = client.get_all_markets(status=status)
    top_markets = sorted(raw_markets, key=lambda m: m.get("volume_24h", 0), reverse=True)[:n_markets]

    results = []
    for raw in top_markets:
        parsed = client.parse_market(raw)
        mid = parsed["mid"]
        if mid is None:
            continue
        posterior = pricer.update_sequential(mid, evidence_list)
        edge = round(posterior - mid, 4)
        flagged = abs(edge) >= pricer.threshold
        signal = "LONG" if edge > 0 else ("SHORT" if edge < 0 else "NEUTRAL")

        results.append(BayesResult(
            ticker=parsed["ticker"],
            title=parsed["title"],
            prior=mid,
            posterior=posterior,
            market_mid=mid,
            edge=edge,
            evidence=evidence_list,
            flagged=flagged,
            signal=signal,
        ))

    return sorted(results, key=lambda r: abs(r.edge), reverse=True)


# ─── Interactive CLI ──────────────────────────────────────────────────────────

def interactive_session():
    """Walk the user through a Bayesian update interactively."""
    pricer = BayesianPricer()

    print(f"\n{Fore.CYAN}{'='*70}")
    print("  KALSHI BAYESIAN FAIR-VALUE ESTIMATOR")
    print(f"{'='*70}{Style.RESET_ALL}")
    print("Enter a market ticker and evidence to update your probability estimate.")
    print("Type 'exit' at any prompt to quit.\n")

    while True:
        ticker = input("Market ticker (e.g. KXHIGHNY-25JUN10): ").strip()
        if ticker.lower() == "exit":
            break

        evidence_list = []
        print("\nAdd evidence pieces (press Enter with no description when done):")
        while True:
            desc = input("  Evidence description (or Enter to finish): ").strip()
            if not desc:
                break
            try:
                p_yes = float(input(f"  P(E | event resolves YES) [0–1]: "))
                p_no = float(input(f"  P(E | event resolves NO)  [0–1]: "))
                evidence_list.append(Evidence(
                    description=desc,
                    p_e_given_h=p_yes,
                    p_e_given_not_h=p_no,
                ))
            except ValueError:
                print("  Invalid input. Skipping this evidence piece.")

        if not evidence_list:
            print("No evidence added. Skipping.\n")
            continue

        try:
            result = pricer.analyze_market(ticker, evidence_list)
            pricer.print_result(result)
        except Exception as e:
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")

        again = input("\nAnalyze another market? (y/n): ").strip().lower()
        if again != "y":
            break

    print("\nGoodbye.")


# ─── Demo / Smoke Test ────────────────────────────────────────────────────────

def run_demo():
    """
    Demo: runs the Bayesian engine with a manually specified prior
    (no live API call needed to test the math).
    """
    pricer = BayesianPricer(threshold=0.05)

    print(f"\n{Fore.CYAN}BAYESIAN PRICER DEMO (manual prior, no API call){Style.RESET_ALL}")
    print("Scenario: 'Will Fed raise rates in July?' — market at 22¢")
    print("Evidence: Strong CPI print (inflation higher than expected)\n")

    prior = 0.22
    evidence = [
        Evidence(
            description="Strong CPI print (above consensus)",
            p_e_given_h=0.75,    # Hotter inflation strongly supports a hike
            p_e_given_not_h=0.30, # But could happen even without hike
        )
    ]

    posterior = pricer.update_sequential(prior, evidence)
    edge = round(posterior - prior, 4)
    flagged = abs(edge) >= pricer.threshold
    signal = "LONG (market underpriced)" if edge > 0 else "SHORT (market overpriced)"

    print(f"Prior (market):   {prior:.4f}  ({prior*100:.1f}¢)")
    print(f"Evidence:         {evidence[0].description}")
    print(f"  P(E|YES)={evidence[0].p_e_given_h}  P(E|NO)={evidence[0].p_e_given_not_h}")
    print(f"Posterior:        {posterior:.4f}  ({posterior*100:.1f}¢)")
    print(f"Edge:             {edge:+.4f}  ({edge*100:+.1f}¢)")
    print(f"Signal:           {signal}")
    print(f"Flagged:          {'YES ⚑' if flagged else 'No'}")
    print(f"\nInterpretation: The new CPI data pushed fair value from 22¢ to {posterior*100:.0f}¢.")
    print(f"At {edge*100:+.0f}¢ of edge, this would be flagged as a potential LONG.")


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        run_demo()
    else:
        interactive_session()
