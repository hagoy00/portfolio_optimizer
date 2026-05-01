import streamlit as st
import pandas as pd

def render_tab4(tab, prices, model):
    tab.markdown("## Sector Exposure")

    if model is None:
        tab.info("Run optimization to generate sector exposure.")
        return

    sector_weights = model.get("sector_weights", {})

    if not sector_weights:
        tab.warning("Sector weights unavailable.")
        return

    sw = pd.Series(sector_weights).sort_values(ascending=False)

    with tab.expander("View Sector Breakdown", expanded=True):
        tab.bar_chart(sw)

    tab.markdown("---")

    tab.markdown("### Commentary")

    commentary = f"""
The portfolio shows diversified exposure across sectors, with the largest allocation in 
**{sw.index[0]}** at **{sw.iloc[0]:.2%}**. This distribution supports balanced risk and 
reduces concentration in any single economic segment.
"""

    with tab.expander("Read Commentary", expanded=True):
        tab.markdown(commentary)
