import streamlit as st
import pandas as pd

def render_tab5(tab, prices, model):
    """
    Drawdown Analysis tab.
    Safe even when model is None.
    """

    tab.subheader("Drawdown Analysis")

    if model is None:
        tab.info(
            "Drawdown analysis will appear here once the optimization model "
            "and portfolio returns are implemented."
        )
        return

    try:
        # Expected future structure:
        # model["drawdown"] = pd.Series(...)
        dd = model.get("drawdown", None)

        if dd is None:
            tab.warning("Model exists but drawdown data is missing.")
            return

        tab.line_chart(dd)

    except Exception as e:
        tab.error(f"Error rendering drawdown analysis: {e}")
