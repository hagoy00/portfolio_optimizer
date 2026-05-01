import streamlit as st
import pandas as pd

def render_tab3(tab, prices, model):
    """
    Weights & Shares tab.
    Shows portfolio weights, dollar allocation, and share counts.
    """

    tab.header("Portfolio Weights & Shares")

    # ---------------------------------------------------
    # GUARD CLAUSES
    # ---------------------------------------------------
    if model is None:
        tab.info("Run optimization to see portfolio weights.")
        return

    if prices is None or prices.empty:
        tab.error("Price data missing — cannot compute shares.")
        return

    weights = model.get("weights", None)
    investment_amount = model.get("investment_amount", None)

    if weights is None:
        tab.warning("Model exists but weight data is missing.")
        return

    if investment_amount is None:
        tab.warning("Investment amount missing from model.")
        return

    # Convert weights to Series
    w = pd.Series(weights, name="Weight").sort_values(ascending=False)

    # ---------------------------------------------------
    # LATEST PRICES
    # ---------------------------------------------------
    try:
        latest_prices = prices.xs("Adj Close", level=1, axis=1).iloc[-1]
        latest_prices = latest_prices.reindex(w.index)
    except Exception as e:
        tab.error(f"Error extracting latest prices: {e}")
        return

    # ---------------------------------------------------
    # DOLLAR ALLOCATION + SHARES
    # ---------------------------------------------------
    dollar_alloc = w * investment_amount
    shares = (dollar_alloc / latest_prices).fillna(0)

    df = pd.DataFrame({
        "Weight": w,
        "Latest Price": latest_prices,
        "Dollar Allocation": dollar_alloc,
        "Shares": shares
    })

    # ---------------------------------------------------
    # DISPLAY
    # ---------------------------------------------------
    tab.subheader("Allocation Overview")

    col1, col2 = tab.columns([1, 1])

    with col1:
        tab.metric("Total Investment", f"${investment_amount:,.0f}")

    with col2:
        tab.metric("Number of Holdings", len(w))

    tab.dataframe(
        df.style.format({
            "Weight": "{:.2%}",
            "Latest Price": "${:,.2f}",
            "Dollar Allocation": "${:,.0f}",
            "Shares": "{:,.2f}"
        })
    )
