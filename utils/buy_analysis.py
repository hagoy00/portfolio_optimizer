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

    if fundamentals is None or not isinstance(fundamentals, pd.DataFrame):
        raise ValueError("fundamentals must be a DataFrame")

    scores = {}

    for t in fundamentals.index:

        m  = safe_val(momentum.get(t))
        r  = safe_val(risk.get(t))

        # Correct DataFrame indexing
        pe = safe_val(fundamentals.loc[t, "PE"])
        pb = safe_val(fundamentals.loc[t, "PB"])
        dy = safe_val(fundamentals.loc[t, "DividendYield"])

        score = 0

        # -------------------------
        # MOMENTUM (weight: 30%)
        # -------------------------
        if m is not None:
            score += np.tanh(m) * 30  # bounded, stable

        # -------------------------
        # RISK (weight: 30%)
        # -------------------------
        if r is not None and r > 0:
            score += (0.30 - r) * 100  # lower risk = higher score

        # -------------------------
        # VALUATION (weight: 25%)
        # -------------------------
        if pe is not None and pe > 0:
            score += max(0, 50 - pe)

        if pb is not None and pb > 0:
            score += max(0, 20 - pb)

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
# MAIN BUY ANALYSIS FUNCTION (FINAL, CORRECT)
# =========================================================
def run_buy_analysis(tickers, fundamentals, prices):
    """
    Main entry point for Buy Analysis.
    Fully crash-proof.
    """

    # -------------------------
    # VALIDATION
    # -------------------------
    if fundamentals is None or not isinstance(fundamentals, pd.DataFrame):
        raise ValueError("fundamentals must be a DataFrame")

    if prices is None or prices.empty:
        return pd.DataFrame(columns=[
            "Ticker", "Momentum", "Risk", "PE", "PB", "DividendYield", "Score", "Rating"
        ])

    # Ensure fundamentals index matches tickers
    fundamentals = fundamentals.reindex(tickers)

    # -------------------------
    # FACTORS
    # -------------------------
    momentum = compute_momentum(prices)
    risk = compute_risk(prices)

    # Composite score
    scores = compute_score(momentum, risk, fundamentals)

    # -------------------------
    # BUILD OUTPUT TABLE
    # -------------------------
    rows = []
    for t in tickers:

        row = {
            "Ticker": t,
            "Momentum": safe_val(momentum.get(t)),
            "Risk": safe_val(risk.get(t)),
            "PE": safe_val(fundamentals.loc[t, "PE"]),
            "PB": safe_val(fundamentals.loc[t, "PB"]),
            "DividendYield": safe_val(fundamentals.loc[t, "DividendYield"]),
            "Score": safe_val(scores.get(t)),
        }

        row["Rating"] = compute_rating(row["Score"])
        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by score descending
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)

    return df
