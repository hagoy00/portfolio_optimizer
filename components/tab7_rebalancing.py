import streamlit as st
import pandas as pd

from utils.optimizer_core import rebalancing_backtest

st.write("DEBUG — entering Tab 7")

def render_rebalancing_tab(tab, prices, model):
    tab.markdown("## Rebalancing Backtest")

    if prices is None or model is None:
        tab.info("Run optimization first.")
        return

    weights = model.get("weights")
    if weights is None:
        tab.error("Weights missing from model.")
        return

    # User-friendly frequency selector
    freq_label = tab.selectbox(
        "Rebalancing Frequency",
        ["Monthly", "Quarterly", "Annual"],
        index=0
    )

    # Convert UI → Pandas-valid frequency codes
    freq_map = {
        "Monthly": "ME",
        "Quarterly": "QE",
        "Annual": "YE"
    }

    freq = freq_map[freq_label]

    # Run backtest
    try:
        result = rebalancing_backtest(
            prices,
            weights,
            freq=freq
        )
    except Exception as e:
        tab.error(f"Rebalancing failed: {e}")
        return

    if result is None or result.empty:
        tab.warning("Rebalancing returned no data.")
        return

    tab.markdown("### Portfolio Value Over Time")
    tab.line_chart(result["Portfolio Value"])
