import streamlit as st
import pandas as pd
from utils.optimizer_core import rebalancing_backtest

def render_tab7(tab, prices, model):
    tab.markdown("## Rebalancing Backtest")

    if prices is None or model is None:
        tab.info("Run optimization first.")
        return

    weights = model.get("weights")
    if weights is None:
        tab.error("Weights missing from model.")
        return

    # Frequency selector
    freq = tab.selectbox(
        "Rebalancing Frequency",
        ["M", "Q", "A"],
        index=0,
        help="M = Monthly, Q = Quarterly, A = Annual"
    )

    # Run backtest
    try:
        result = rebalancing_backtest(
            prices,
            weights,
            freq=freq  # works with our tolerant function
        )
    except Exception as e:
        tab.error(f"Rebalancing failed: {e}")
        return

    if result is None or result.empty:
        tab.warning("Rebalancing returned no data.")
        return

    tab.markdown("### Portfolio Value Over Time")
    tab.line_chart(result["Portfolio Value"])
