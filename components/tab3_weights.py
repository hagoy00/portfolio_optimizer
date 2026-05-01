import streamlit as st
import pandas as pd

def render_tab3(tab, prices, model):
    """
    Weights & Shares tab.
    Shows portfolio weights and share counts.
    """

    tab.subheader("Portfolio Weights & Shares")

    if model is None:
        tab.info("Run optimization to see portfolio weights.")
        return

    try:
        weights = model.get("weights", None)
        shares = model.get("shares", None)

        if weights is None:
            tab.warning("Model exists but weight data is missing.")
            return

        w = pd.Series(weights, name="Weight")

        if shares is not None:
            s = pd.Series(shares, name="Shares").reindex(w.index)
        else:
            s = pd.Series([None] * len(w), index=w.index, name="Shares")

        df = pd.concat([w, s], axis=1)

        tab.markdown("### Weights and Shares")
        tab.dataframe(
            df.style.format({
                "Weight": "{:.2%}"
            })
        )

    except Exception as e:
        tab.error(f"Error rendering weights: {e}")
