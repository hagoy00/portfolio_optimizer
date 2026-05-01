import streamlit as st
import pandas as pd

def render_tab2(tab, prices, model):
    """
    Efficient Frontier tab.
    Uses model["frontier"] DataFrame with columns: return, volatility, sharpe.
    """

    tab.markdown("## Efficient Frontier")

    if model is None:
        tab.info("Run optimization to see the efficient frontier.")
        return

    frontier = model.get("frontier", None)
    perf = model.get("performance", {})
    weights = model.get("weights", None)

    # ---------------------------------------------------
    # GUARD CLAUSE — missing frontier
    # ---------------------------------------------------
    if frontier is None or (isinstance(frontier, pd.DataFrame) and frontier.empty):
        tab.warning("Model exists but frontier data is missing or empty.")
        return

    # ---------------------------------------------------
    # FRONTIER CHART
    # ---------------------------------------------------
    tab.markdown("### Frontier Visualization")

    with tab.expander("View Efficient Frontier Chart", expanded=True):
        tab.scatter_chart(
            frontier,
            x="volatility",
            y="return"
        )

    tab.markdown("---")

    # ---------------------------------------------------
    # SELECTED PORTFOLIO METRICS
    # ---------------------------------------------------
    if perf:
        tab.markdown("### Selected Portfolio (Max Shar
