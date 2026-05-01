import streamlit as st
import pandas as pd
from utils.optimizer import rebalancing_backtest

def render_tab7(tab, prices, model):
    tab.markdown("## Rebalancing Backtest")

    if model is None:
        tab.info("Run optimization to see rebalancing results.")
        return

    # ---------------------------------------------------
    # FREQUENCY SELECTION
    # ---------------------------------------------------
    tab.markdown("### Rebalancing Settings")

    freq = tab.selectbox(
        "Rebalancing Frequency",
        ["ME", "W", "Q"],
        index=0,
        help="ME = Month-End, W = Weekly, Q = Quarterly"
    )

    # Convert Q → QE (Pandas requirement)
    freq_map = {"Q": "QE", "ME": "ME", "W": "W"}
    freq = freq_map[freq]

    # ---------------------------------------------------
    # RUN BACKTEST
    # ---------------------------------------------------
    try:
        rb = rebalancing_backtest(prices, model["weights"], freq=freq)
    except Exception as e:
        tab.error(f"Rebalancing failed: {e}")
        return

    if rb is None or rb.empty:
        tab.warning("Rebalancing results unavailable.")
        return

    # ---------------------------------------------------
    # CHART
    # ---------------------------------------------------
    tab.markdown("### Portfolio Value Over Time")

    with tab.expander("View Rebalancing Chart", expanded=True):
        tab.line_chart(rb)

    tab.markdown("---")

    # ---------------------------------------------------
    # METRICS PANEL
    # ---------------------------------------------------
    tab.markdown("### Backtest Summary")

    final_value = rb.iloc[-1]

    col1, col2 = tab.columns(2)
    col1.metric("Final Portfolio Value", f"{final_value:.3f}")
    col2.metric("Rebalancing Frequency", freq)

    tab.markdown("---")

    # ---------------------------------------------------
    # RAW DATA
    # ---------------------------------------------------
    with tab.expander("View Raw
