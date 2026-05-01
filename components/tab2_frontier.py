import streamlit as st
import pandas as pd

def render_tab2(tab, prices, model):
    """
    Efficient Frontier tab.
    Uses model["frontier"] DataFrame with columns: return, volatility, sharpe.
    """

    tab.subheader("Efficient Frontier")

    if model is None:
        tab.info("Run optimization to see the efficient frontier.")
        return

    frontier = model.get("frontier", None)
    weights = model.get("weights", None)

    if frontier is None or isinstance(frontier, pd.DataFrame) and frontier.empty:
        tab.warning("Model exists but frontier data is missing or empty.")
        return

    tab.markdown("### Frontier Scatter")

    tab.scatter_chart(
        frontier,
        x="volatility",
        y="return"
    )

    if weights is not None:
        perf = model.get("performance", {})
        if perf:
            tab.markdown("### Selected Portfolio (Max Sharpe)")
            tab.write(
                f"Expected Return: **{perf['return']:.2%}**, "
                f"Volatility: **{perf['volatility']:.2%}**, "
                f"Sharpe: **{perf['sharpe']:.2f}**"
            )
