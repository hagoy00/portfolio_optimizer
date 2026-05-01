import streamlit as st

def render_tab6(tab, prices, model):
    """
    Monte Carlo Simulation tab.
    Safe even when model is None.
    """

    tab.subheader("Monte Carlo Simulation")

    if model is None:
        tab.info(
            "Monte Carlo results will appear here once the optimization model "
            "and simulation engine are implemented."
        )
        return

    try:
        mc = model.get("montecarlo", None)

        if mc is None:
            tab.warning("Model exists but Monte Carlo data is missing.")
            return

        tab.line_chart(mc)

    except Exception as e:
        tab.error(f"Error rendering Monte Carlo simulation: {e}")
