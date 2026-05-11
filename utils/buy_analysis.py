import pandas as pd
import numpy as np

# ---------------------------------------------------------
# Safe numeric conversion
# ---------------------------------------------------------
def safe_val(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return float(x)
    except:
        return None

# ---------------------------------------------------------
# Momentum calculation
# ---------------------------------------------------------
def compute_momentum(prices, window=60):
    """
    Computes simple momentum: (Price_today / Price_60_days_ago) - 1
    """
    try:
        return prices.pct_change(window).iloc[-1]
    except Exception:
        return pd.Series({c: None for c in prices.columns})

# ---------------------------------------------------------
# Risk calculation (volatility)
# ---------------------------------------------------------
def compute_risk(prices, window=60):
    """
    Computes rolling volatility as risk measure.
    """
    try:
        returns = prices.pct_change()
        return returns.rolling(window).std().iloc[-1]
    except Exception:
        return pd.Series({c: None for c in prices.columns})

# ---------------------------------------------------------
# Composite scoring model
# ---------------------------------------------------------
def compute_score(momentum, risk, fundamentals):
    """
    Combines momentum, risk, and fundamentals into a single score.
    Higher is better.
    """
    scores = {}

    for t in fundamentals.keys():
        m = safe_val(momentum.get(t))
        r = safe_val(risk.get(t))

        pe = safe_val(fundamentals[t].get("PE"))
        pb = safe_val(fundamentals[t].get("PB"))
        dy = safe_val(fundamentals[t].get("DividendYield"))

        score = 0

        # Momentum
        if m is not None:
            score += m * 50

        # Risk (lower is better)
        if r is not None and r > 0:
            score += (0.30 - r) * 100

        # Valuation
        if pe is not None and pe > 0:
            score += max(0, 50 - pe)

        if pb is not None and pb > 0:
            score += max(0, 20 - pb)

        # Dividend
        if dy is not None and dy > 0:
            score += dy * 100

        scores[t] = score

    return scores

# ---------------------------------------------------------
# Rating model
# ---------------------------------------------------------
def compute_rating(score):
    if score >= 60:
        return "Buy"
    elif score >= 25:
        return "Hold"
    else:
        return "Sell"

# ---------------------------------------------------------
# Main Buy Analysis Function
# ---------------------------------------------------------
def run_buy_analysis(tickers, fundamentals, prices):
    """
    Main entry point for Buy Analysis.
    Uses cleaned price data from app.py.
    """

    # Momentum & Risk
    momentum = compute_momentum(prices)
    risk = compute_risk(prices)

    # Composite score
    scores = compute_score(momentum, risk, fundamentals)

    # Build output table
    rows = []
    for t in tickers:
        f = fundamentals.get(t, {})

        row = {
            "Ticker": t,
            "Momentum": safe_val(momentum.get(t)),
            "Risk": safe_val(risk.get(t)),
            "PE": safe_val(f.get("PE")),
            "PB": safe_val(f.get("PB")),
            "DividendYield": safe_val(f.get("DividendYield")),
            "Score": safe_val(scores.get(t)),
        }

        row["Rating"] = compute_rating(row["Score"] if row["Score"] is not None else 0)

        rows.append(row)

    df = pd.DataFrame(rows)
    return df
