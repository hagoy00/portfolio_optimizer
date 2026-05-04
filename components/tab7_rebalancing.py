import streamlit as st
import pandas as pd
from utils.optimizer_core import rebalancing_backtest


def render_rebalancing_tab(tab, prices, model):
    tab.markdown("## Rebalancing Backtest")

    if model is None:
        tab.error("Model is missing.")
        return

    # ---------------------------------------------------------
    # Frequency selector (correct pandas codes)
    # ---------------------------------------------------------
    freq_label = tab.selectbox(
        "Rebalancing Frequency",
        options=["Monthly", "Quarterly", "Annual", "Weekly"],
        index=0
    )

    freq_map = {
        "Monthly": "ME",     # Month-End
        "Quarterly": "QE",   # Quarter-End
        "Annual": "YE",      # Year-End
        "Weekly": "W",       # Weekly
    }
    freq = freq_map[freq_label]

    # ---------------------------------------------------------
    # Convert weights to a clean Series
    # ---------------------------------------------------------
    target_weights = model.get("weights")
    if target_weights is None:
        tab.error("No weights found in model.")
        return

    # Convert dict → Series
    w = pd.Series(target_weights)

    # Align weights to tickers in prices
    if isinstance(prices.columns, pd.MultiIndex):
        tickers = prices.columns.get_level_values("Ticker").unique()
    else:
        tickers = prices.columns

    w = w.reindex(tickers).fillna(0)

    # ---------------------------------------------------------
    # Run backtest
    # ---------------------------------------------------------
    try:
        result = rebalancing_backtest(prices, w, freq=freq)
    except Exception as e:
        tab.error(f"Rebalancing failed: {e}")
        return

    if result is None or result.empty:
        tab.warning("Rebalancing backtest returned no data.")
        return

    # ---------------------------------------------------------
    # Display results
    # ---------------------------------------------------------
    tab.markdown("### Portfolio Value Over Time")
    tab.line_chart(result["Portfolio Value"])

    with tab.expander("Show Recent Data"):
        tab.dataframe(result.tail(), use_container_width=True)
