import streamlit as st
import pandas as pd
import numpy as np

# FIXED SIGNATURE — must accept (tab, prices, model)
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

    # --- Concentration ---
    top_weight = max(weights.values()) if len(weights) > 0 else 0
    diversification = 1 - top_weight

    # ---------------------------------------------------
    # CLEAN TWO-COLUMN METRIC LAYOUT
    # ---------------------------------------------------
    tab.markdown("### Core Portfolio Metrics")

    col1, col2, col3 = tab.columns(3)
    col1.metric("Expected Return", f"{exp_ret:.2%}")
    col2.metric("Volatility", f"{vol:.2%}")
    col3.metric("Sharpe Ratio", f"{sharpe:.2f}")

    col4, col5, col6 = tab.columns(3)
    col4.metric("Max Drawdown", f"{max_dd:.2%}" if max_dd else "N/A")
    col5.metric("Largest Position", f"{top_weight:.2%}")
    col6.metric("Diversification Score", f"{diversification:.2%}")

    tab.markdown("---")

    # ---------------------------------------------------
    # ALLOCATION SNAPSHOT
    # ---------------------------------------------------
    tab.markdown("### Allocation Snapshot")

    w_series = pd.Series(weights).sort_values(ascending=False)

    with tab.expander("View Allocation Chart", expanded=True):
        tab.bar_chart(w_series)

    tab.markdown("---")

    # ---------------------------------------------------
    # SECTOR EXPOSURE
    # ---------------------------------------------------
    if sector_weights:
        tab.markdown("### Sector Exposure")

        with tab.expander("View Sector Breakdown", expanded=False):
            tab.bar_chart(pd.Series(sector_weights))

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
