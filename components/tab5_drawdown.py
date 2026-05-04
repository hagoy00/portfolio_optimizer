import streamlit as st
import pandas as pd

#st.write("DEBUG — entering Tab 5")

def render_drawdown_tab(tab, prices, model):
    """
    Drawdown Analysis tab.
    Safe even when model is None.
    """

    tab.markdown("## Drawdown Analysis")

    if model is None:
        tab.info(
            "Drawdown analysis will appear here once the optimization model "
            "and portfolio returns are implemented."
        )
        return

    try:
        dd = model.get("drawdown", None)

        # ---------------------------------------------------
        # GUARD CLAUSE — missing drawdown data
        # ---------------------------------------------------
        if dd is None or isinstance(dd, pd.DataFrame) and dd.empty:
            tab.warning("Model exists but drawdown data is missing.")
            return

        # ---------------------------------------------------
        # CLEAN LAYOUT
        # ---------------------------------------------------
        tab.markdown("### Portfolio Drawdown Curve")

        with tab.expander("View Drawdown Chart", expanded=True):
            tab.line_chart(dd)

        tab.markdown("---")

        # ---------------------------------------------------
        # OPTIONAL: DRAWdown STATISTICS
        # ---------------------------------------------------
        tab.markdown("### Drawdown Statistics")

        max_dd = dd.min().min() if isinstance(dd, pd.DataFrame) else dd.min()
        recovery_days = dd[dd == 0].shape[0]  # simplistic placeholder

        col1, col2 = tab.columns(2)

        col1.metric("Maximum Drawdown", f"{max_dd:.2%}")
        col2.metric("Recovery Days", f"{recovery_days}")

        tab.markdown("---")

        # ---------------------------------------------------
        # EXPANDER: RAW DATA
        # ---------------------------------------------------
        with tab.expander("View Raw Drawdown Data", expanded=False):
            tab.dataframe(dd, hide_index=False)

    except Exception as e:
        tab.error(f"Error rendering drawdown analysis: {e}")
