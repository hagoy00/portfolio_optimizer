import streamlit as st
import pandas as pd

def render_tab4(tab, sector_weights):
    """
    Sector Exposure tab.
    Safe even when sector_weights is None.
    """

    tab.markdown("## Sector Exposure")

    # ---------------------------------------------------
    # GUARD CLAUSE — no sector data
    # ---------------------------------------------------
    if sector_weights is None or len(sector_weights) == 0:
        tab.info(
            "Sector exposure will appear here once the optimization model "
            "and sector mapping are available."
        )
        return

    try:
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(
            sector_weights, orient="index", columns=["Weight"]
        ).sort_values("Weight", ascending=False)

        # ---------------------------------------------------
        # CLEAN LAYOUT
        # ---------------------------------------------------
        tab.markdown("### Sector Allocation Breakdown")

        col1, col2 = tab.columns([2, 3])

        # --- Column 1: Sector Table ---
        with col1:
            tab.markdown("#### Sector Weights")
            tab.dataframe(
                df.style.format({"Weight": "{:.2%}"}),
                hide_index=False
            )

        # --- Column 2: Sector Chart ---
        with col2:
            tab.markdown("#### Sector Chart")
            tab.bar_chart(df)

        tab.markdown("---")

        # ---------------------------------------------------
        # EXPANDER: RAW DATA
        # ---------------------------------------------------
        with tab.expander("View Raw Sector Data", expanded=False):
            tab.dataframe(
                df.style.format({"Weight": "{:.2%}"}),
                hide_index=False
            )

    except Exception as e:
        tab.error(f"Error rendering sector exposure: {e}")
