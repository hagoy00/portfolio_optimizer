import pandas as pd
import numpy as np

def get_adj_close(full_prices):
    out = {}
    for t in full_prices.columns.levels[0]:
        if "Adj Close" in full_prices[t].columns:
            out[t] = full_prices[t]["Adj Close"].dropna()
        else:
            out[t] = full_prices[t]["Close"].dropna()
    return out


def momentum_and_risk(ticker, full_prices):
    adj = get_adj_close(full_prices)[ticker]

    if len(adj) < 60:
        return "Insufficient Data", "Insufficient Data", adj

    # Momentum: 30‑day return
    mom = adj.pct_change(30).iloc[-1]

    # Risk: 30‑day volatility
    vol = adj.pct_change().rolling(30).std().iloc[-1]

    # Labels
    if mom > 0.10:
        momentum_label = "Strong Momentum"
    elif mom > 0:
        momentum_label = "Mild Momentum"
    else:
        momentum_label = "Weak / Negative Momentum"

    if vol < 0.02:
        risk_label = "Low Risk"
    elif vol < 0.04:
        risk_label = "Moderate Risk"
    else:
        risk_label = "High Risk"

    return momentum_label, risk_label, adj
# ---------------------------------------------------------
# BUY / HOLD / SELL ANALYSIS ENGINE (REQUIRED BY app.py)
# ---------------------------------------------------------

def run_buy_analysis(tickers, fundamentals, performance):
    """
    Produces a Buy / Hold / Sell score for each ticker using:
    - Momentum
    - Risk
    - Fundamentals
    - Portfolio performance
    """

    results = []

    for t in tickers:

        # ---------- Momentum & Risk ----------
        try:
            mom_label, risk_label, _ = momentum_and_risk(t, fundamentals["full_prices"])
        except Exception:
            mom_label, risk_label = "N/A", "N/A"

        # ---------- Fundamentals ----------
        pe = fundamentals.get(t, {}).get("PE", np.nan)
        pb = fundamentals.get(t, {}).get("PB", np.nan)
        div = fundamentals.get(t, {}).get("DividendYield", np.nan)

        # ---------- Scoring ----------
        score = 0

        # Momentum scoring
        if mom_label == "Strong Momentum":
            score += 2
        elif mom_label == "Mild Momentum":
            score += 1
        else:
            score -= 1

        # Risk scoring
        if risk_label == "Low Risk":
            score += 2
        elif risk_label == "Moderate Risk":
            score += 1
        else:
            score -= 1

        # Fundamentals scoring
        if isinstance(pe, (int, float)) and pe < 20:
            score += 1
        if isinstance(pb, (int, float)) and pb < 3:
            score += 1
        if isinstance(div, (int, float)) and div > 0.02:
            score += 1

        # ---------- Final Rating ----------
        if score >= 4:
            rating = "BUY"
        elif score >= 1:
            rating = "HOLD"
        else:
            rating = "SELL"

        results.append({
            "Ticker": t,
            "Momentum": mom_label,
            "Risk": risk_label,
            "PE": pe,
            "PB": pb,
            "DividendYield": div,
            "Score": score,
            "Rating": rating
        })

    return pd.DataFrame(results)
