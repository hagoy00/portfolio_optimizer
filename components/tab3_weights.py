import streamlit as st
import pandas as pd

def render_tab3(tab, prices, model):
    """
    Weights & Shares tab.
    Shows portfolio weights and share counts.
    """

    tab.markdown("## Portfolio Weights & Shares")

    if model is None:
        tab.info("Run optimization to see portfolio weights.")
        return

    try:
        weights = model.get("weights", None)
        shares = model.get("shares", None)

        # ---------------------------------------------------
        # GUARD CLAUSE — missing weights
        # ---------------------------------------------------
        if weights is None:
            tab.warning("Model exists but weight data is missing.")
            return

        # Convert weights to Series
        w = pd.Series(weights, name="Weight").sort_values(ascending=False)

        # Convert shares to Series (aligned with weights)
        if shares is not None:
            s = pd.Series(shares, name="Shares").reindex(w.index)
        else:
            s = pd.Series([None] * len(w), index=w.index, name="Shares")

        df = pd.concat([w, s], axis=1)

        # ---------------------------------------------------
        # CLEAN LAYOUT
        # ---------------------------------------------------
        tab.markdown("### Allocation Overview")

        col1, col2 = tab.columns([2, 3])

        # --- Column 1: Weight Summary ---
        with col1:
           
