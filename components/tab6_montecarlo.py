import streamlit as st
import pandas as pd

def render_tab6(tab, prices, model):
    tab.markdown("## Monte Carlo Simulation")

    if model is None:
        tab.info("Run optimization first.")
        return

    # Get Monte Carlo data from model
    mc = model.get("monte_carlo", None)

    if mc is None or not isinstance(mc, pd.DataFrame) or mc.empty:
        tab.warning("Model exists but Monte Carlo data is missing.")
        return

    tab.markdown("### Simulation Paths")
    tab.line_chart(mc)

    # Summary statistics
    final_values = mc.iloc[-1]

    tab.markdown("### Summary Statistics")
    col1, col2, col3 = tab.columns(3)
    col1.metric("Mean Final Value", f"{final_values.mean():.2f}")
    col2.metric("Median Final Value", f"{final_values.median():.2f}")
    col3.metric("Worst Case", f"{final_values.min():.2f}")

    tab.markdown("### Distribution of Final Values")
    tab.bar_chart(final_values)
