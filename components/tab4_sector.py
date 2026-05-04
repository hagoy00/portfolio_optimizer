import streamlit as st
import pandas as pd

#st.write("DEBUG — entering Tab 4")

def render_sector_tab(tab, prices, model):
    tab.header("Sector Exposure")

    # ---------------------------------------------------
    # GUARD CLAUSES
    # ---------------------------------------------------
    if model is None:
        tab.info("Run optimization to see sector exposure.")
        return

    sector_weights = model.get("sector_weights", None)

    if sector_weights is None:
        tab.warning("Sector weights not available.")
        return

    if isinstance(sector_weights, pd.Series):
        sw = sector_weights.dropna()
    else:
        tab.error("Sector weights must be a pandas Series.")
        return

    if sw.empty:
        tab.warning("Sector weights are empty.")
        return

    # ---------------------------------------------------
    # CONVERT TO DATAFRAME FOR STREAMLIT
    # ---------------------------------------------------
    df_sw = (
        sw.reset_index()
          .rename(columns={"index": "Sector", 0: "Weight"})
    )

    # Ensure correct column names
    if "Sector" not in df_sw.columns or "Weight" not in df_sw.columns:
        tab.error("Sector weight data is malformed.")
        return

    # ---------------------------------------------------
    # DISPLAY
    # ---------------------------------------------------
    tab.subheader("Sector Allocation")

    tab.bar_chart(df_sw, x="Sector", y="Weight")

    tab.dataframe(
        df_sw.style.format({
            "Weight": "{:.2%}"
        })
    )
