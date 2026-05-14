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

def compute_risk(prices, window=60):
    """
    Computes annualized volatility.
    Ensures output index matches ticker symbols exactly.
    """

    if prices is None or prices.empty:
        return pd.Series(dtype=float)

    # Ensure columns are clean ticker symbols
    prices = prices.copy()
    prices.columns = [str(c).upper() for c in prices.columns]

    try:
        # Daily returns
        returns = prices.pct_change().dropna()

        # If insufficient data, fallback to simple std
        if len(returns) < window:
            vol = returns.std() * np.sqrt(252)
        else:
            vol = returns.rolling(window).std().iloc[-1] * np.sqrt(252)

        # Ensure index matches tickers
        vol.index = prices.columns

        return vol

    except Exception:
        # Return None for each ticker if something breaks
        return pd.Series({c: None for c in prices.columns})

# =========================================================
# MOMENTUM (ANNUALIZED)
# =========================================================

def compute_momentum(prices, window=60):
    """
    Computes annualized momentum using last N days of returns.
    Ensures output index matches ticker symbols exactly.
    """

    if prices is None or prices.empty:
        return pd.Series(dtype=float)

    # Ensure columns are clean ticker symbols
    prices = prices.copy()
    prices.columns = [str(c).upper() for c in prices.columns]

    try:
        # Daily returns
        returns = prices.pct_change().dropna()

        # If insufficient data, fallback to mean return
        if len(returns) < window:
            return returns.mean() * 252

        # Annualized momentum over last N days
        momentum = returns.tail(window).mean() * 252

        # Ensure index matches tickers
        momentum.index = prices.columns

        return momentum

    except Exception:
        # Return None for each ticker if something breaks
        return pd.Series({c: None for c in prices.columns})

# =========================================================
# RISK (ANNUALIZED VOLATILITY)
# =========================================================

def compute_momentum(prices, window=60):
    """
    Computes annualized momentum using last N days of returns.
    Ensures output index matches ticker symbols exactly.
    """

    if prices is None or prices.empty:
        return pd.Series(dtype=float)

    # Ensure columns are clean ticker symbols
    prices = prices.copy()
    prices.columns = [str(c).upper() for c in prices.columns]

    try:
        # Daily returns
        returns = prices.pct_change().dropna()

        # If insufficient data, fallback to mean return
        if len(returns) < window:
            return returns.mean() * 252

        # Annualized momentum over last N days
        momentum = returns.tail(window).mean() * 252

        # Ensure index matches tickers
        momentum.index = prices.columns

        return momentum

    except Exception:
        # Return None for each ticker if something breaks
        return pd.Series({c: None for c in prices.columns})

# =========================================================
# COMPOSITE SCORING MODEL
# =========================================================
def compute_score(momentum, risk, fundamentals_df):
    if fundamentals_df is None or not isinstance(fundamentals_df, pd.DataFrame):
        raise ValueError("fundamentals_df must be a DataFrame")

    scores = {}

    for t in fundamentals_df.index:
        m  = safe_val(momentum.get(t))
        r  = safe_val(risk.get(t))

        pe = safe_val(fundamentals_df.loc[t, "PE"])
        pb = safe_val(fundamentals_df.loc[t, "PB"])
        dy = safe_val(fundamentals_df.loc[t, "DividendYield"])

        score = 0

        # Momentum (30%)
        if m is not None:
            score += np.tanh(m) * 30

        # Risk (30%)
        if r is not None and r > 0:
            score += (0.30 - r) * 100

        # Valuation (25%)
        if pe is not None and pe > 0:
            score += max(0, 50 - pe)

        if pb is not None and pb > 0:
            score += max(0, 20 - pb)

        # Dividend (15%)
        if dy is not None and dy > 0:
            score += dy * 100

        scores[t] = score

    return scores


# =========================================================
# RATING MODEL
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
def run_buy_analysis(tickers, fundamentals_df, prices):
    if fundamentals_df is None or not isinstance(fundamentals_df, pd.DataFrame):
        raise ValueError("fundamentals_df must be a DataFrame")

    if prices is None or prices.empty:
        return pd.DataFrame(columns=[
            "Ticker", "Momentum", "Risk", "PE", "PB", "DividendYield", "Score", "Rating"
        ])

    fundamentals_df = fundamentals_df.reindex(tickers)

    momentum = compute_momentum(prices)
    risk = compute_risk(prices)

    scores = compute_score(momentum, risk, fundamentals_df)

    rows = []
    for t in tickers:
        row = {
            "Ticker": t,
            "Momentum": safe_val(momentum.get(t)),
            "Risk": safe_val(risk.get(t)),
            "PE": safe_val(fundamentals_df.loc[t, "PE"]),
            "PB": safe_val(fundamentals_df.loc[t, "PB"]),
            "DividendYield": safe_val(fundamentals_df.loc[t, "DividendYield"]),
            "Score": safe_val(scores.get(t)),
        }

        row["Rating"] = compute_rating(row["Score"])
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)

    return df
