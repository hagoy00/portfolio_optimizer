import streamlit as st
import pandas as pd
import numpy as np

print(">>> USING LATEST tab1_summary.py")

def render_tab1(tab, prices, model):
    tab.markdown("## Portfolio Manager Dashboard")

    if model is None:
        tab.info("Run optimization to generate portfolio metrics.")
        return

    # Extract model components
    perf = model.get("performance", {})
    dd = model.get("drawdown", None)
    weights = model.get("weights", {})
    sector_weights = model.get("sector_weights", {})

    # --- Core Metrics ---
    exp_ret = perf.get("return", 0)
    vol = perf.get("volatility", 0)
    sharpe = perf.get("sharpe", 0)
    max_dd = dd.min().min() if dd is not None else None

    # --- Safe weights normalization ---
    print(">>> DEBUG TAB1 weights:", type(weights), weights)

    if weights is None:
        weights = pd.Series(dtype=float)
    elif isinstance(weights, (float, int, np.floating)):
        weights = pd.Series(dtype=float)
    elif isinstance(weights, (list, tuple, np.ndarray)):
        weights = pd.Series(dtype=float)
    else:
        try:
            weights = pd.Series(weights)
        except Exception:
            print(">>> ERROR: Invalid weights:", type(weights), weights)
            weights = pd.Series(dtype=float)

    top_weight = weights.max() if len(weights) > 0 else 0
    diversification = 1 - top_weight

    # ---------------------------------------------------
    # METRICS PANEL
    # ---------------------------------------------------
    tab.markdown("### Core Portfolio Metrics")

    col1, col2, col3 = tab.columns(3)
    col1.metric("Expected Return", f"{exp_ret:.2%}")
    col2.metric("Volatility", f"{vol:.2%}")
    col3.metric("Sharpe Ratio", f"{sharpe:.2f}")

    col4, col5, col6 = tab.columns(3)
    col4.metric("Max Drawdown", f"{max_dd:.2%}" if max_dd is not None else "N/A")
    col5.metric("Largest Position", f"{top_weight:.2%}")
    col6.metric("Diversification Score", f"{diversification:.2%}")

    tab.markdown("---")

    # ---------------------------------------------------
    # ALLOCATION SNAPSHOT (TABLE ONLY – NO CHART)
    # ---------------------------------------------------
    tab.markdown("### Allocation Snapshot")

    if len(weights) == 0:
        tab.info("No weights available to display.")
    else:
        w_series = weights.sort_values(ascending=False)
        alloc_df = w_series.to_frame(name="Weight")
        alloc_df.index.name = "Ticker"
        tab.dataframe(alloc_df.style.format({"Weight": "{:.2%}"}))

    tab.markdown("---")

    # ---------------------------------------------------
    # SECTOR EXPOSURE (TABLE ONLY – NO CHART)
    # ---------------------------------------------------
    if sector_weights:
        tab.markdown("### Sector Exposure")

        sector_series = pd.Series(sector_weights, name="Weight")
        sector_series.index.name = "Sector"
        sector_df = sector_series.to_frame()
        tab.dataframe(sector_df.style.format({"Weight": "{:.2%}"}))

        tab.markdown("---")

    # ---------------------------------------------------
    # PM COMMENTARY
    # ---------------------------------------------------
    tab.markdown("### Portfolio Manager Commentary")

    commentary = f"""
The portfolio demonstrates a **Sharpe ratio of {sharpe:.2f}**, supported by an expected return of 
**{exp_ret:.2%}** and volatility of **{vol:.2%}**. Drawdown behavior remains controlled with a 
maximum drawdown of **{max_dd:.2%}**, indicating stable downside risk.

Positioning shows moderate concentration with the largest weight at **{top_weight:.2%}**, 
resulting in a diversification score of **{diversification:.2%}**. Sector exposure is balanced 
and aligned with favorable risk‑adjusted characteristics.

Overall, the portfolio is positioned for efficient growth with controlled risk.
"""

    with tab.expander("Read Commentary", expanded=True):
        tab.markdown(commentary)
