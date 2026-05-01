import streamlit as st
import pandas as pd
import numpy as np

def render_tab6(tab, prices, model):
    """
    Monte Carlo Simulation tab.
    Safe even when model is None.
    """

    tab.markdown("## Monte Carlo Simulation")

    if model is None:
        tab.info(
            "Monte Carlo results will appear here once the optimization model "
            "and simulation engine are implemented."
        )
        return

    try:
        mc = model.get("montecarlo", None)

        # ---------------------------------------------------
        # GUARD CLAUSE — missing MC data
        # ---------------------------------------------------
        if mc is None or (isinstance(mc, pd.DataFrame) and mc.empty):
            tab.warning("Model exists but Monte Carlo data is missing.")
            return

        # ---------------------------------------------------
        # CLEAN LAYOUT
        # ---------------------------------------------------
        tab.markdown("### Simulation Paths")

        with tab.expander("View Monte Carlo Simulation Paths", expanded=True):
            tab.line_chart(mc)

        tab.markdown("---")

        # ---------------------------------------------------
        # STATISTICS PANEL
        # ---------------------------------------------------
        tab.markdown("### Simulation Statistics")

        # Final values from each simulation path
        final_values = mc.iloc[-1]

        mean_val = final_values.mean()
        median_val = final_values.median()
        min_val = final_values.min()
        max_val = final_values.max()
        pct_5 = final_values.quantile(0.05)
        pct_95 = final_values.quantile(0.95)

        col1, col2, col3 = tab.columns(3)

        col1.metric("Mean Final Value", f"${mean_val:,.0f}")
        col2.metric("Median Final Value", f"${median_val:,.0f}")
        col3.metric("5th Percentile", f"${pct_5:,.0f}")

        col4, col5, col6 = tab.columns(3)

        col4.metric("95th Percentile", f"${pct_95:,.0f}")
        col5.metric("Min Final Value", f"${min_val:,.0f}")
        col6.metric("Max Final Value", f"${max_val:,.0f}")

        tab.markdown("---")

        # ---------------------------------------------------
        # DISTRIBUTION HISTOGRAM
        # ---------------------------------------------------
        tab.markdown("### Distribution of Final Portfolio Values")

        with tab.expander("View Distribution Histogram", expanded=False):
            tab.bar_chart(final_values)

        tab.markdown("---")

        # ---------------------------------------------------
        # RAW DATA
        # ---------------------------------------------------
        with tab.expander("View Raw Monte Carlo Data", expanded=False):
            tab.dataframe(mc, hide_index=False)

    except Exception as e:
        tab.error(f"Error rendering Monte Carlo simulation: {e}")
