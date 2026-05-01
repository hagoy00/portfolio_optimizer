import streamlit as st
import pandas as pd
import numpy as np

def render_tab1(tab, model):
    tab.subheader("Portfolio Manager Dashboard")

    if model is None:
        tab.info("Run optimization to generate portfolio metrics.")
        return

    perf = model.get("performance", {})
    dd = model.get("drawdown", None)
    weights = model.get("weights", {})
    sector_weights = model.get("sector_weights", {})

    # --- Core Metrics ---
    exp_ret = perf.get("return", 0)
    vol = perf.get("volatility", 0)
    sharpe = perf.get("sharpe", 0)
    max_dd = dd.min().min() if dd is not None else None

    # --- Concentration ---
    top_weight = max(weights.values())
    diversification = 1 - top_weight

    # --- Summary Table ---
    summary = pd.DataFrame({
        "Metric": [
            "Expected Annual Return",
            "Annualized Volatility",
            "Sharpe Ratio",
            "Max Drawdown",
            "Largest Position Weight",
            "Diversification Score"
        ],
        "Value": [
            f"{exp_ret:.2%}",
            f"{vol:.2%}",
            f"{sharpe:.2f}",
            f"{max_dd:.2%}" if max_dd else "N/A",
            f"{top_weight:.2%}",
            f"{diversification:.2%}"
        ]
    })

    tab.markdown("### 📊 Core Portfolio Metrics")
    tab.dataframe(summary, hide_index=True)

    # --- Allocation Snapshot ---
    tab.markdown("### 🧩 Allocation Snapshot")

    w_series = pd.Series(weights).sort_values(ascending=False)
    tab.bar_chart(w_series)

    # --- Sector Exposure ---
    if sector_weights:
        tab.markdown("### 🏢 Sector Exposure")
        tab.bar_chart(pd.Series(sector_weights))

    # --- PM Commentary ---
    tab.markdown("### 🧠 PM Commentary")

    commentary = f"""
The portfolio exhibits a **Sharpe ratio of {sharpe:.2f}**, supported by an expected return of 
**{exp_ret:.2%}** and volatility of **{vol:.2%}**. Drawdown behavior remains controlled with a 
maximum drawdown of **{max_dd:.2%}**, indicating stable downside risk.

Positioning is moderately diversified with the largest weight at **{top_weight:.2%}**, producing 
a diversification score of **{diversification:.2%}**. Sector exposure is balanced, with allocations 
tilted toward areas showing favorable risk‑adjusted characteristics.

Overall, the portfolio is positioned for efficient growth with controlled risk.
"""
    tab.markdown(commentary)
