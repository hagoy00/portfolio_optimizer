import pandas as pd
import numpy as np


# =========================================================
# SAFE VALUE CONVERSION
# =========================================================
def safe_val(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None


# =========================================================
# MOMENTUM (ANNUALIZED)
# =========================================================
def compute_momentum(prices, window=60):
    """
    Computes annualized momentum using last N days of returns.
    More stable than simple pct_change(window).
    """

    if prices is None or prices.empty:
        return pd.Series(dtype=float)

    try:
        returns = prices.pct_change().dropna()
        if len(returns) < window:
            return returns.mean() * 252  # fallback: mean return
        return returns.tail(window).mean() * 252
    except Exception:
        return pd.Series({c: None for c in prices.columns})


# =========================================================
# RISK (ANNUALIZED VOLATILITY)
# =========================================================
def compute_risk(prices, window=60):
    """
    Computes annualized volatility.
    """

    if prices is None or prices.empty:
        return pd.Series(dtype=float)

    try:
        returns = prices.pct_change().dropna()
        if len(returns) < window:
            vol = returns.std() * np.sqrt(252)
        else:
            vol = returns.rolling(window).std().iloc[-1] * np.sqrt(252)
        return vol
    except Exception:
        return pd.Series({c: None for c in prices.columns})


# =========================================================
# COMPOSITE SCORING MODEL (INSTITUTIONAL-GRADE)
# =========================================================
def compute_score(momentum, risk, fundamentals):
    """
    Multi-factor scoring model:
    - Momentum (positive is good)
    - Risk (lower is better)
    - Valuation (PE, PB)
    - Dividend Yield
    """

    scores = {}

    for t in fundamentals.keys():
        m = safe_val(momentum.get(t))
        r = safe_val(risk.get(t))

        pe = safe_val(fundamentals[t].get("PE"))
        pb = safe_val(fundamentals[t].get("PB"))
        dy = safe_val(fundamentals[t].get("DividendYield"))

        score = 0

        # -------------------------
        # MOMENTUM (weight: 30%)
        # -------------------------
        if m is not None:
            score += np.tanh(m) * 30  # bounded, stable

        # -------------------------
        # RISK (weight: 25%)
        # -------------------------
        if r is not None and r > 0:
            score += (0.30 - r) * 80  # lower vol → higher score

        # -------------------------
        # VALUATION (weight: 30%)
        # -------------------------
        if pe is not None and pe > 0:
            score += max(0, 40 - pe)

        if pb is not None and pb > 0:
            score += max(0, 15 - pb)

        # -------------------------
        # DIVIDEND (weight: 15%)
        # -------------------------
        if dy is not None and dy > 0:
            score += dy * 100

        scores[t] = score

    return scores


# =========================================================
# RATING MODEL (CLEAN)
# =========================================================
def compute_rating(score):
    if score is None:
        return "Sell"
    if score >= 60:
        return "Buy"
    elif score >= 25:
        return "Hold"
    return "Sell"


# =========================================================
# MAIN BUY ANALYSIS FUNCTION
# =========================================================
def run_buy_analysis(tickers, fundamentals, prices):
    """
    Main entry point for Buy Analysis.
    Fully crash-proof.
    """

    if prices is None or prices.empty:
        return pd.DataFrame(columns=[
            "Ticker", "Momentum", "Risk", "PE", "PB", "DividendYield", "Score", "Rating"
        ])

    # Compute factors
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

        row["Rating"] = compute_rating(row["Score"])

        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by score descending
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)

    return df
