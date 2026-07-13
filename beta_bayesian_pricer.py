"""
beta_bayesian_pricer.py — Distributional Bayesian fair-value estimation.

This is a distributional generalization of bayesian_pricer.py. Instead of
carrying a single point estimate of the fair value, it represents belief about
the true YES-probability theta as a Beta distribution. That buys three things
the scalar engine can't give you:

    1. A 90% CREDIBLE INTERVAL around fair value (how *sure* are we, not just
       where the point estimate sits).
    2. A probabilistic edge signal — P(market is underpriced) — estimated by
       MONTE CARLO integration over the posterior.
    3. A confidence flag: an edge only counts as high-conviction when the
       credible interval of the edge excludes zero.

─── Model ────────────────────────────────────────────────────────────────────

    theta = P(event resolves YES), treated as unknown.

    Prior:      theta ~ Beta(a0, b0),  seeded from the market mid so that
                    mean = a0/(a0+b0) = market_mid
                    a0 + b0 = kappa  (prior strength / pseudo-sample-size:
                                      how many "observations" of trust we give
                                      the market price).
                => a0 = mid * kappa,   b0 = (1 - mid) * kappa

    Evidence:   each piece is expressed in the SAME intuitive terms as the
                scalar engine — P(E|YES) and P(E|NO) — plus a weight w
                (pseudo-sample-size for that evidence). We convert it to a
                direction and pseudo-counts:
                    q      = P(E|YES) / (P(E|YES) + P(E|NO))   # support for YES
                    da     = w * q
                    db     = w * (1 - q)
                This is a conjugate Beta-Bernoulli update.

    Posterior:  theta ~ Beta(a0 + sum(da), b0 + sum(db))

    Fair value = posterior mean = a/(a+b)
    90% CI     = [Beta.ppf(0.05, a, b), Beta.ppf(0.95, a, b)]   (exact)
    P(underpriced) = P(theta > mid) = (1/N) * sum_i 1[theta_i > mid],
                     theta_i ~ Beta(a, b)                        (Monte Carlo)

Relationship to the scalar engine:
    The scalar bayesian_pricer applies Bayes' Rule to a single point and returns
    a point posterior. This module returns a full distribution. The posterior
    MEANS will not be numerically identical — they come from different (though
    related) models — which is expected: the value here is the uncertainty
    quantification, not reproducing the point estimate.

Run interactively:
    python beta_bayesian_pricer.py

Demo (no API / no credentials needed):
    python beta_bayesian_pricer.py --demo
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.stats import beta as beta_dist
from colorama import init, Fore, Style

import config

init(autoreset=True)


# ─── Config Defaults ──────────────────────────────────────────────────────────

# Prior strength: how many pseudo-observations of "trust" we grant the market
# mid when seeding the prior. Higher kappa = tighter prior = harder to move.
PRIOR_STRENGTH = float(getattr(config, "PRIOR_STRENGTH", 20.0))

# Default weight (pseudo-sample-size) per evidence piece if not specified.
DEFAULT_EVIDENCE_WEIGHT = float(getattr(config, "DEFAULT_EVIDENCE_WEIGHT", 5.0))

# Monte Carlo draws for probability estimates.
MC_DRAWS = int(getattr(config, "MC_DRAWS", 50_000))

# Credible-interval mass (0.90 => 5th/95th percentiles).
CI_LEVEL = float(getattr(config, "CI_LEVEL", 0.90))


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class Evidence:
    """
    A single piece of evidence.

    Expressed in the same terms as the scalar engine (P(E|YES), P(E|NO)) so
    the two pricers share an interface, plus a `weight` giving how many
    pseudo-observations this evidence is worth.
    """
    description: str
    p_e_given_h: float          # P(E|YES)
    p_e_given_not_h: float      # P(E|NO)
    weight: float = DEFAULT_EVIDENCE_WEIGHT

    def __post_init__(self):
        assert 0 < self.p_e_given_h <= 1, "p_e_given_h must be in (0, 1]"
        assert 0 < self.p_e_given_not_h <= 1, "p_e_given_not_h must be in (0, 1]"
        assert self.weight > 0, "weight must be positive"

    @property
    def direction(self) -> float:
        """q = normalized support for YES, in (0, 1)."""
        return self.p_e_given_h / (self.p_e_given_h + self.p_e_given_not_h)

    @property
    def pseudo_counts(self) -> tuple[float, float]:
        """(delta_alpha, delta_beta) contributed by this evidence."""
        q = self.direction
        return self.weight * q, self.weight * (1.0 - q)


@dataclass
class BetaBelief:
    """A Beta(alpha, beta) belief over the true YES-probability."""
    alpha: float
    beta: float

    @classmethod
    def from_mid(cls, mid: float, kappa: float = PRIOR_STRENGTH) -> "BetaBelief":
        """Seed a prior from a market mid with the given prior strength."""
        mid = min(max(mid, 1e-4), 1 - 1e-4)   # keep off the 0/1 boundary
        return cls(alpha=mid * kappa, beta=(1.0 - mid) * kappa)

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def concentration(self) -> float:
        return self.alpha + self.beta

    def credible_interval(self, level: float = CI_LEVEL) -> tuple[float, float]:
        """Exact central credible interval from the Beta quantile function."""
        tail = (1.0 - level) / 2.0
        lo = float(beta_dist.ppf(tail, self.alpha, self.beta))
        hi = float(beta_dist.ppf(1.0 - tail, self.alpha, self.beta))
        return lo, hi

    def update(self, evidence_list: list[Evidence]) -> "BetaBelief":
        """Conjugate update: add pseudo-counts from each evidence piece."""
        a, b = self.alpha, self.beta
        for ev in evidence_list:
            da, db = ev.pseudo_counts
            a += da
            b += db
        return BetaBelief(alpha=a, beta=b)


@dataclass
class BetaResult:
    """Output of a distributional analysis for one market."""
    ticker: str
    title: str
    market_mid: float
    prior: BetaBelief
    posterior: BetaBelief
    fair_value: float                 # posterior mean
    ci_low: float                     # fair-value credible interval
    ci_high: float
    edge: float                       # fair_value - market_mid
    edge_ci_low: float                # credible interval of the edge
    edge_ci_high: float
    prob_underpriced: float           # P(theta > mid), Monte Carlo
    evidence: list[Evidence]
    flagged: bool                     # |edge| >= threshold
    high_confidence: bool             # edge CI excludes zero
    signal: str


# ─── Core Engine ──────────────────────────────────────────────────────────────

class BetaBayesianPricer:
    """
    Distributional Bayesian pricer for binary event markets.

    Carries a Beta posterior, reports credible intervals, and estimates
    P(market underpriced) by Monte Carlo integration over the posterior.
    """

    def __init__(
        self,
        threshold: float = None,
        prior_strength: float = PRIOR_STRENGTH,
        mc_draws: int = MC_DRAWS,
        ci_level: float = CI_LEVEL,
        seed: Optional[int] = None,
    ):
        self.threshold = threshold if threshold is not None else config.EDGE_THRESHOLD
        self.prior_strength = prior_strength
        self.mc_draws = mc_draws
        self.ci_level = ci_level
        self._rng = np.random.default_rng(seed)
        self._client = None   # lazy: no credentials needed for demo / math

    @property
    def client(self):
        if self._client is None:
            from kalshi_client import KalshiClient
            self._client = KalshiClient()
        return self._client

    def price(
        self,
        market_mid: float,
        evidence_list: list[Evidence],
        ticker: str = "(manual)",
        title: str = "",
    ) -> BetaResult:
        """
        Core pricing routine — pure math, no network. Given a mid and evidence,
        produce the full distributional result.
        """
        prior = BetaBelief.from_mid(market_mid, self.prior_strength)
        posterior = prior.update(evidence_list)

        fair_value = posterior.mean
        ci_low, ci_high = posterior.credible_interval(self.ci_level)
        edge = fair_value - market_mid

        # Monte Carlo integration over the posterior for probability statements.
        draws = self._rng.beta(posterior.alpha, posterior.beta, size=self.mc_draws)
        prob_underpriced = float(np.mean(draws > market_mid))
        edge_draws = draws - market_mid
        tail = (1.0 - self.ci_level) / 2.0
        edge_ci_low = float(np.quantile(edge_draws, tail))
        edge_ci_high = float(np.quantile(edge_draws, 1.0 - tail))

        flagged = abs(edge) >= self.threshold
        high_confidence = (edge_ci_low > 0) or (edge_ci_high < 0)  # CI excludes 0

        if edge > 0:
            signal = "LONG (market underpriced)"
        elif edge < 0:
            signal = "SHORT (market overpriced)"
        else:
            signal = "NEUTRAL"

        return BetaResult(
            ticker=ticker,
            title=title,
            market_mid=market_mid,
            prior=prior,
            posterior=posterior,
            fair_value=round(fair_value, 4),
            ci_low=round(ci_low, 4),
            ci_high=round(ci_high, 4),
            edge=round(edge, 4),
            edge_ci_low=round(edge_ci_low, 4),
            edge_ci_high=round(edge_ci_high, 4),
            prob_underpriced=round(prob_underpriced, 4),
            evidence=evidence_list,
            flagged=flagged,
            high_confidence=high_confidence,
            signal=signal,
        )

    def analyze_market(
        self,
        ticker: str,
        evidence_list: list[Evidence],
        manual_prior: Optional[float] = None,
    ) -> BetaResult:
        """Fetch a live market, then price it distributionally."""
        market_data = self.client.get_market(ticker)
        if not market_data:
            raise ValueError(f"Market '{ticker}' not found on Kalshi.")

        parsed = self.client.parse_market(market_data)
        market_mid = parsed["mid"]
        if market_mid is None:
            raise ValueError(f"No mid-price available for '{ticker}'.")

        mid = manual_prior if manual_prior is not None else market_mid
        return self.price(mid, evidence_list, ticker=ticker, title=parsed["title"])

    def print_result(self, r: BetaResult) -> None:
        """Pretty-print a BetaResult."""
        flag_str = (
            f"{Fore.RED}\u2691 FLAGGED{Style.RESET_ALL}" if r.flagged
            else f"{Fore.WHITE}\u25cb Within threshold{Style.RESET_ALL}"
        )
        conf_str = (
            f"{Fore.GREEN}HIGH (90% CI excludes 0){Style.RESET_ALL}" if r.high_confidence
            else f"{Fore.YELLOW}LOW (90% CI spans 0){Style.RESET_ALL}"
        )
        edge_color = Fore.GREEN if r.edge > 0 else (Fore.RED if r.edge < 0 else Fore.WHITE)

        print(f"\n{'\u2500'*70}")
        print(f"{Fore.CYAN}Market:    {r.ticker}{Style.RESET_ALL}")
        if r.title:
            print(f"Title:     {r.title}")
        print(f"{'\u2500'*70}")
        print(f"Market mid:          {r.market_mid:.4f}  ({r.market_mid*100:.1f}\u00a2)")
        for i, ev in enumerate(r.evidence, 1):
            print(f"  Evidence {i}: {ev.description}")
            print(f"    P(E|YES)={ev.p_e_given_h:.2f}  P(E|NO)={ev.p_e_given_not_h:.2f}  weight={ev.weight:g}")
        print(f"Fair value (mean):   {r.fair_value:.4f}  ({r.fair_value*100:.1f}\u00a2)")
        print(f"90% credible int.:   [{r.ci_low:.4f}, {r.ci_high:.4f}]  "
              f"([{r.ci_low*100:.1f}\u00a2, {r.ci_high*100:.1f}\u00a2])")
        print(f"Edge:                {edge_color}{r.edge:+.4f}  ({r.edge*100:+.1f}\u00a2){Style.RESET_ALL}")
        print(f"Edge 90% CI:         [{r.edge_ci_low:+.4f}, {r.edge_ci_high:+.4f}]")
        print(f"P(underpriced):      {r.prob_underpriced*100:.1f}%   (Monte Carlo, {self.mc_draws:,} draws)")
        print(f"Signal:              {r.signal}")
        print(f"Conviction:          {conf_str}")
        print(f"Status:              {flag_str}")
        print(f"{'\u2500'*70}")


# ─── Batch Scanner ────────────────────────────────────────────────────────────

def scan_markets_with_evidence(
    evidence_list: list[Evidence],
    n_markets: int = 20,
    status: str = "open",
    pricer: Optional[BetaBayesianPricer] = None,
) -> list[BetaResult]:
    """
    Apply the same evidence to the top N markets by volume and rank by |edge|.
    Each result carries its own credible interval and P(underpriced).
    """
    from kalshi_client import KalshiClient
    client = KalshiClient()
    pricer = pricer or BetaBayesianPricer()

    print(f"\nFetching top {n_markets} markets...")
    raw_markets = client.get_all_markets(status=status)
    top = sorted(raw_markets, key=lambda m: m.get("volume_24h", 0), reverse=True)[:n_markets]

    results = []
    for raw in top:
        parsed = client.parse_market(raw)
        mid = parsed["mid"]
        if mid is None:
            continue
        results.append(
            pricer.price(mid, evidence_list, ticker=parsed["ticker"], title=parsed["title"])
        )
    return sorted(results, key=lambda r: abs(r.edge), reverse=True)


# ─── Interactive CLI ──────────────────────────────────────────────────────────

def interactive_session():
    pricer = BetaBayesianPricer()
    print(f"\n{Fore.CYAN}{'='*70}")
    print("  KALSHI BETA-BAYESIAN FAIR-VALUE ESTIMATOR (distributional)")
    print(f"{'='*70}{Style.RESET_ALL}")
    print("Enter a ticker and evidence to get a fair value WITH a credible interval.")
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
                p_yes = float(input("  P(E | YES) [0-1]: "))
                p_no = float(input("  P(E | NO)  [0-1]: "))
                w_raw = input("  Weight (pseudo-obs, Enter for default): ").strip()
                w = float(w_raw) if w_raw else DEFAULT_EVIDENCE_WEIGHT
                evidence_list.append(Evidence(desc, p_yes, p_no, w))
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

        if input("\nAnalyze another market? (y/n): ").strip().lower() != "y":
            break
    print("\nGoodbye.")


# ─── Demo / Smoke Test ────────────────────────────────────────────────────────

def run_demo():
    """Runs the distributional engine with a manual prior — no API call."""
    pricer = BetaBayesianPricer(threshold=0.05, seed=42)

    print(f"\n{Fore.CYAN}BETA-BAYESIAN PRICER DEMO (manual prior, no API call){Style.RESET_ALL}")
    print("Scenario: 'Will the Fed raise rates in July?' — market at 22\u00a2")
    print("Evidence: Strong CPI print (inflation above consensus)\n")

    evidence = [
        Evidence(
            description="Strong CPI print (above consensus)",
            p_e_given_h=0.75,
            p_e_given_not_h=0.30,
            weight=6.0,
        )
    ]
    result = pricer.price(0.22, evidence, ticker="DEMO-FED-JUL", title="Fed hike in July?")
    pricer.print_result(result)

    print("\nInterpretation:")
    print(f"  Fair value moved to {result.fair_value*100:.0f}\u00a2, but the 90% credible interval")
    print(f"  [{result.ci_low*100:.0f}\u00a2, {result.ci_high*100:.0f}\u00a2] tells you how firm that estimate is.")
    print(f"  Monte Carlo puts P(market underpriced) at {result.prob_underpriced*100:.0f}%.")
    verdict = "high-conviction" if result.high_confidence else "directional but not yet high-conviction"
    print(f"  Because the edge's 90% CI {'excludes' if result.high_confidence else 'still spans'} zero, "
          f"this is {verdict}.")


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        run_demo()
    else:
        interactive_session()
