# Polymarket Quant Playbook

6 formulas hedge funds use to extract systematic edges from prediction markets, replicated for retail traders.

## The 6 Formulas

### Formula 1 - LMSR Pricing Model
The AMM powering Polymarket. Models trade impact and spots mispricings in low-liquidity pools.

```
Price_i = exp(q_i / b) / sum(exp(q_j / b))
```

- b = liquidity depth (small b = bigger edges)
- Edge: ~$500/day on impact arb in volatile markets

---

### Formula 2 - Kelly Criterion
Maximizes geometric growth. Used by Renaissance, Two Sigma, and every serious quant fund.

```
f* = (p * odds - (1 - p)) / odds
```

- Use fractional Kelly (0.25-0.5x) in practice
- Edge: compounds small consistent edges into large returns

---

### Formula 3 - Expected Value Gap
Bet only when your model probability beats the market implied probability.

```
EV = (p_true - price) * payout
```

- Entry threshold: EV > 0.05 after fees
- Edge: $300+/day scanning geo/politics markets on $2K bankroll

---

### Formula 4 - KL Divergence
Measures distance between probability distributions of correlated markets.

```
D_KL(P || Q) = sum(P_i * log(P_i / Q_i))
```

- Arb threshold: KL > 0.2
- Edge: 15% portfolio uplift on diversified bets

---

### Formula 5 - Bregman Projection
Scans multi-outcome markets for risk-free arb by projecting onto the probability simplex.

```
min D_phi(mu || theta)   subject to: sum(mu_i) = 1, mu_i >= 0
```

- Spots price inconsistencies across elections, awards, sports
- Edge: ~$496 average per trade, near-zero downside

---

### Formula 6 - Bayesian Update
Updates beliefs dynamically as new evidence arrives.

```
P(H|E) = P(E|H) * P(H) / P(E)
```

- Feed in tweets, polls, news as evidence
- Edge: +12% accuracy in volatile geo/news markets

---

## Quick Start

```bash
git clone https://github.com/varshavenkataraman/polymarket-quant-playbook.git
cd polymarket-quant-playbook

pip install -r requirements.txt

cp .env.example .env

python formulas/01_lmsr_pricing.py

python bot/main.py

jupyter notebook notebooks/backtest.ipynb
```

---

## Bot Usage

```bash
python bot/main.py
python bot/main.py --loop 60
```

---

## API Keys

| Key | Where to get it | Requirement |
|-----|----------------|----------|
| POLYGON_API_KEY | polygon.io, free signup | Yes, for live data |
| TELEGRAM_BOT_TOKEN | Telegram, @BotFather, /newbot | No, only for alerts |
| TELEGRAM_CHAT_ID | Telegram, @userinfobot | No, only for alerts |

Keys go in .env — never commit this file (already in .gitignore).

---

## Risk Management

| Rule | Detail |
|------|--------|
| Fractional Kelly | Never bet full Kelly, use 25-50% |
| Drawdown stop | Hard stop at 20% portfolio drawdown |
| Walk-forward only | Never test in-sample |
| Fee awareness | 1-2% fees erode small edges fast |
| Target Sharpe | Aim for Sharpe > 1.5 |

---

## Homework Checklist

- Run 01_lmsr_pricing.py on a real Polymarket pool
- Backtest Kelly sizing on 50 historical resolutions
- Pull Polygon data for 10 markets and compute EV gaps
- Compute KL divergence for 5 correlated market pairs
- Extend Bregman projection to 3+ outcomes
- Build a Bayesian updater fed by real X/news data

---

Based on the Quant Playbook for Polymarket thread by @0xricker on X.