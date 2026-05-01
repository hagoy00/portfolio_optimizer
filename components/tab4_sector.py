import streamlit as st
import pandas as pd

def render_tab4(tab, sector_weights):
    """
    Sector Exposure tab.
    Safe even when sector_weights is None.
    """

    tab.subheader("Sector Exposure")

    if sector_weights is None:
        tab.info(
            "Sector exposure will appear here once the optimization model "
            "and sector mapping are implemented."
        )
        return

    try:
        df = pd.DataFrame.from_dict(sector_weights, orient="index", columns=["Weight"])
        tab.bar_chart(df)

    except Exception as e:
        tab.error(f"Error rendering sector exposure: {e}")
