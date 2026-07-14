# Methodology

## Overview

This project has two components: a **live market data pipeline** and a **Bayesian fair-value estimation engine**. Together they form a lightweight framework for identifying potentially mispriced binary event contracts on Kalshi.

---

## Market Data Pipeline

Kalshi exposes a public REST API at `https://api.elections.kalshi.com/trade-api/v2`. No authentication is required for read-only market data, which covers all open markets across categories including economics, politics, climate, and sports.

The pipeline fetches markets, paginates through results, and normalizes prices from Kalshi's native **cent format** (integers 0–99) into **decimal probabilities** (0.00–0.99). This makes prices directly interpretable as implied probabilities — a market at 34¢ implies a 34% chance of YES resolution.

The mid-price is computed as:

```
mid = (yes_bid + yes_ask) / 2
```

This is used as the market's consensus probability estimate throughout the Bayesian engine.

---

## Bayesian Fair-Value Estimation

### The Core Problem

Prediction market prices reflect the crowd's aggregate probability estimate. When new information enters the market, such a a data release or news event, prices should update to reflect the new evidence. In practice, markets often lag, especially in thinner markets or immediately after a surprise.

The goal of the Bayesian engine is to compute a **fair-value posterior** given new evidence and compare it to the live market price. A meaningful gap between the two is a potential trading signal.

### The Math

The engine applies **Bayes' Rule** for binary outcomes:

```
P(H | E) = P(E | H) · P(H)  /  [P(E | H) · P(H)  +  P(E | ¬H) · P(¬H)]
```

Where:

| Symbol | Meaning |
|--------|---------|
| `H` | The event resolves YES |
| `E` | A new piece of observed evidence |
| `P(H)` | **Prior** — the current Kalshi market mid-price |
| `P(E \| H)` | **Likelihood** — how probable is this evidence if YES resolves? |
| `P(E \| ¬H)` | **Likelihood** — how probable is this evidence if NO resolves? |
| `P(H \| E)` | **Posterior** — the updated fair-value estimate |

### Sequential Updates

When multiple pieces of evidence are available, the engine applies updates sequentially — the posterior from each step becomes the prior for the next. This is valid under the assumption of **conditional independence** between evidence pieces (i.e., knowing one piece of evidence doesn't change how informative the other pieces are, given the outcome).

```python
for evidence in evidence_list:
    prior = bayes_update(prior, evidence.p_e_given_h, evidence.p_e_given_not_h)
```

### Edge Detection

After computing the posterior, the engine calculates the **edge**:

```
edge = posterior - market_mid
```

If `|edge| ≥ threshold` (default: 5¢), the market is flagged:
- **Positive edge** → market is underpriced relative to fair value → potential LONG
- **Negative edge** → market is overpriced → potential SHORT

### Worked Example

**Market:** "Will the Fed raise rates in July?" — trading at **22¢**

**Evidence:** A CPI print comes in significantly above consensus (hotter than expected inflation).

**Likelihood assessment:**
- `P(hot CPI | Fed hikes)` = 0.75 — a Fed hike would be more consistent with high inflation
- `P(hot CPI | Fed doesn't hike)` = 0.30 — high inflation can occur without a hike, but less consistent

**Applying Bayes' Rule:**

```
P(hike | hot CPI) = (0.75 × 0.22) / [(0.75 × 0.22) + (0.30 × 0.78)]
                  = 0.165 / (0.165 + 0.234)
                  = 0.165 / 0.399
                  ≈ 0.41
```

**Edge = 0.41 − 0.22 = +0.19** → flagged as a potential LONG

### Key Design Choices

**The likelihoods are subjective.** The engine doesn't auto-generate `P(E|H)` — that judgment comes from the analyst. This is intentional: the value of the tool is in forcing explicit, quantified reasoning about evidence strength, not in replacing that reasoning.

**The market price is the prior.** Using the live mid as the starting probability means the engine takes the crowd's estimate seriously as a baseline. Overriding this with a manual prior is supported (e.g., if you believe the market is already significantly off before any new evidence).

**No position sizing here.** The engine flags potential mispricings but doesn't compute optimal bet sizes. A natural extension would be Kelly Criterion sizing: `f* = (bp − q) / b`, where `b` is the net odds, `p` is your posterior, and `q = 1 − p`.

---

## Extending the Project

- **WebSocket streaming** — replace REST polling with Kalshi's WebSocket feed for real-time price updates
- **Automated evidence ingestion** — pipe in BLS CPI releases, Fed speech NLP sentiment, or Polymarket prices as structured evidence inputs
- **Cross-platform arbitrage** — compare Kalshi and Polymarket prices on equivalent markets to surface arb opportunities
- **Backtesting** — store historical snapshots and evaluate whether flagged edges were predictive of subsequent price moves

---

## Distributional Extension: Beta Model with Credible Intervals & Monte Carlo

`beta_bayesian_pricer.py` generalizes the point-estimate engine above by
representing belief about the true YES-probability `θ` as a **Beta
distribution** rather than a single number. This adds three things the scalar
engine cannot express: a credible interval around fair value, a Monte-Carlo
estimate of `P(market underpriced)`, and a conviction flag.

### The model

**Prior.** Treat `θ = P(event resolves YES)` as unknown and seed a Beta prior
from the market mid, so the prior *mean* equals the market price and the prior
*strength* `κ` (a pseudo-sample-size expressing how much we trust the crowd)
controls how hard it is to move:

```
θ ~ Beta(a₀, b₀),   a₀ = mid · κ,   b₀ = (1 − mid) · κ
```

**Evidence.** Each piece is expressed in the same `P(E|YES)`, `P(E|NO)` terms as
the scalar engine, plus a weight `w` (pseudo-observations). It is converted to a
direction and conjugate pseudo-counts:

```
q  = P(E|YES) / (P(E|YES) + P(E|NO))       # normalized support for YES
Δa = w · q,   Δb = w · (1 − q)
```

**Posterior** (conjugate Beta-Bernoulli update):

```
θ ~ Beta(a₀ + Σ Δa,  b₀ + Σ Δb)
```

### What it reports

- **Fair value** = posterior mean `a / (a + b)`
- **90% credible interval** = `[Beta.ppf(0.05, a, b), Beta.ppf(0.95, a, b)]` — computed exactly from the Beta quantile function
- **P(market underpriced)** = `P(θ > mid)`, estimated by **Monte Carlo integration**: draw `θ₁…θ_N ~ Beta(a, b)` and take the fraction exceeding the mid
- **Conviction** — an edge is only marked *high-confidence* when the 90% credible interval of the edge (`θ − mid`) excludes zero

### Relationship to the scalar engine

The scalar `bayesian_pricer` applies Bayes' Rule to a single point and returns a
point posterior. This module returns a full distribution, so the posterior
*means* will not be numerically identical — they come from related but distinct
models. That is by design: the point of the Beta version is **uncertainty
quantification**, not reproducing the point estimate. A wide credible interval
that straddles the market price is itself the signal — it says "the point edge
is positive, but we don't yet have enough evidence to bet on it."
