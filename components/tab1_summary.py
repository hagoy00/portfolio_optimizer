import streamlit as st
import pandas as pd

def render_summary_tab(tab, prices, model):
    tab.markdown("## Portfolio Summary")

    if model is None:
        tab.error("Model is missing.")
        return

    weights = model.get("weights")
    exp_return = model.get("expected_return")
    volatility = model.get("volatility")
    sharpe = model.get("sharpe")
    sector_weights = model.get("sector_weights")

    if isinstance(weights, dict):
        weights = pd.Series(weights)

    col1, col2, col3 = tab.columns(3)
    col1.metric("Expected Return (Annualized)", f"{exp_return:.2%}")
    col2.metric("Volatility (Annualized)", f"{volatility:.2%}")
    col3.metric("Sharpe Ratio", f"{sharpe:.2f}")

    tab.markdown("---")
    tab.markdown("### Portfolio Weights")
    tab.dataframe(weights.to_frame("Weight"))

    tab.markdown("---")
    if sector_weights is not None:
        tab.markdown("### Sector Exposure")
        tab.dataframe(sector_weights.to_frame("Weight"))
    else:
        tab.info("Sector weights not available.")
