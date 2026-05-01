import streamlit as st
import pandas as pd
from utils.optimizer import rebalancing_backtest

def render_tab7(tab, prices, model):
    tab.subheader("Rebalancing Backtest")

    if model is None:
        tab.info("Run optimization to see rebalancing results.")
        return

    # User selects frequency
    freq = tab.selectbox(
        "Rebalancing Frequency",
        ["ME", "W", "Q"],
        index=0,
        help="ME = Month-End, W = Weekly, Q = Quarterly"
    )

    # Convert Q → QE (Pandas 2.2+ requirement)
    freq_map = {"Q": "QE", "ME": "ME", "W": "W"}
    freq = freq_map[freq]

    # Run backtest
    rb = rebalancing_backtest(prices, model["weights"], freq=freq)

    if rb is None or rb.empty:
        tab.warning("Rebalancing results unavailable.")
        return

    tab.markdown("### Portfolio Value Over Time")
    tab.line_chart(rb)

    tab.markdown(f"Final Value: **{rb.iloc[-1]:.3f}**")
