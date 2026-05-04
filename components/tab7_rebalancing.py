import streamlit as st
import pandas as pd

from utils.optimizer_core import rebalancing_backtest


def render_rebalancing_tab(tab, prices, model):
    tab.markdown("## Rebalancing Backtest")

    if model is None:
        tab.error("Model is missing.")
        return

    # Frequency selector
    freq_label = tab.selectbox(
        "Rebalancing Frequency",
        options=["Monthly", "Quarterly", "Yearly", "Weekly"],
        index=0
    )

    freq_map = {
        "Monthly": "M",
        "Quarterly": "Q",
        "Yearly": "A",
        "Weekly": "W",
    }
    freq = freq_map[freq_label]

    # Get weights from model (dict or Series)
    target_weights = model.get("weights")
    if target_weights is None:
        tab.error("No weights found in model.")
        return

    # Run backtest
    try:
        result = rebalancing_backtest(prices, target_weights, freq=freq)
    except Exception as e:
        tab.error(f"Rebalancing failed: {e}")
        return

    if result is None or result.empty:
        tab.warning("Rebalancing backtest returned no data.")
        return

    # Plot + table
    tab.line_chart(result["Portfolio Value"])
    tab.dataframe(result.tail(), use_container_width=True)
