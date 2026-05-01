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
