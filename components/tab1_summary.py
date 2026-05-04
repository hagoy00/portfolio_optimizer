import streamlit as st
import pandas as pd

def render_summary_tab(tab, prices, model):
    tab.markdown("## Portfolio Summary")

    if model is None:
        tab.error("Model is missing.")
        return

    # ---------------------------------------------------
    # Extract metrics from model["performance"]
    # ---------------------------------------------------
    perf = model.get("performance", {})

    exp_return = perf.get("expected_return")
    volatility = perf.get("volatility")
    sharpe = perf.get("sharpe")

    weights = model.get("weights")
    sector_weights = model.get("sector_weights")

    if isinstance(weights, dict):
        weights = pd.Series(weights)

    # ---------------------------------------------------
    # Metrics Display
    # ---------------------------------------------------
    col1, col2, col3 = tab.columns(3)

    col1.metric("Expected Return (Annualized)", f"{exp_return:.2%}" if exp_return is not None else "N/A")
    col2.metric("Volatility (Annualized)", f"{volatility:.2%}" if volatility is not None else "N/A")
    col3.metric("Sharpe Ratio", f"{sharpe:.2f}" if sharpe is not None else "N/A")

    tab.markdown("---")

    # ---------------------------------------------------
    # Weights Table
    # ---------------------------------------------------
    tab.markdown("### Portfolio Weights")
    tab.dataframe(weights.to_frame("Weight"))

    tab.markdown("---")

    # ---------------------------------------------------
    # Sector Exposure
    # ---------------------------------------------------
    if sector_weights is not None:
        tab.markdown("### Sector Exposure")
        tab.dataframe(sector_weights.to_frame("Weight"))
    else:
        tab.info("Sector weights not available.")
