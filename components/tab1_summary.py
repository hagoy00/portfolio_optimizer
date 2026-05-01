import streamlit as st
import pandas as pd

def render_tab1(tab, prices, model):
    tab.markdown("## Portfolio Summary")

    if model is None:
        tab.error("Model is missing.")
        return

    # -------------------------------
    # LOAD MODEL FIELDS SAFELY
    # -------------------------------
    weights = model.get("weights")
    exp_return = model.get("expected_return")
    volatility = model.get("volatility")
    sharpe = model.get("sharpe")
    sector_weights = model.get("sector_weights")

    # -------------------------------
    # FIX WEIGHTS TYPE (bulletproof)
    # -------------------------------
    if isinstance(weights, pd.Series):
        pass
    elif isinstance(weights, dict):
        weights = pd.Series(weights)
    elif hasattr(weights, "__len__") and len(weights) == len(model["tickers"]):
        weights = pd.Series(weights, index=model["tickers"])
    else:
        tab.error("Weights missing or invalid.")
        return

    # -------------------------------
    # SUMMARY METRICS
    # -------------------------------
    col1, col2, col3 = tab.columns(3)
    col1.metric("Expected Return (Annualized)", f"{exp_return:.2%}")
    col2.metric("Volatility (Annualized)", f"{volatility:.2%}")
    col3.metric("Sharpe Ratio", f"{sharpe:.2f}")

    tab.markdown("---")

    # -------------------------------
    # WEIGHTS TABLE
    # -------------------------------
    tab.markdown("### Portfolio Weights")
    tab.dataframe(weights.to_frame("Weight"))

    tab.markdown("---")

    # -------------------------------
    # SECTOR WEIGHTS
    # -------------------------------
    if sector_weights is not None:
        tab.markdown("### Sector Exposure")
        tab.dataframe(sector_weights.to_frame("Weight"))
    else:
        tab.info("Sector weights not available.")
