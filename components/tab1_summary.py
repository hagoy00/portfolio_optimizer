import streamlit as st
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# TAB 1 — PORTFOLIO SUMMARY
# ---------------------------------------------------------

def render_tab1(tab, prices, model):
    with tab:

        st.header("Portfolio Manager Dashboard")

        # -----------------------------
        # GUARD CLAUSES
        # -----------------------------
        if prices is None or prices.empty:
            st.error("Price data is missing. Cannot compute portfolio summary.")
            return

        if model is None or "weights" not in model:
            st.error("Model is missing weights. Cannot compute portfolio summary.")
            return

        weights = model.get("weights")

# Force-correct the type no matter what
if isinstance(weights, pd.Series):
    pass
elif isinstance(weights, dict):
    weights = pd.Series(weights)
elif hasattr(weights, "__len__") and len(weights) == len(model["tickers"]):
    weights = pd.Series(weights, index=model["tickers"])
else:
    st.error("Weights missing or invalid.")
    return


        if weights.empty:
            st.error("Weights are empty. Cannot compute portfolio summary.")
            return

        # -----------------------------
        # CORE METRICS
        # -----------------------------
        returns = prices.pct_change().dropna()

        # Expected return
        exp_return = np.sum(weights * returns.mean() * 252)

        # Volatility
        vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))

        # Sharpe ratio
        sharpe = exp_return / vol if vol > 0 else 0

        # Max drawdown
        cumulative = (1 + returns.dot(weights)).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()

        # Largest position
        largest_pos = weights.max()

        # Diversification score (simple inverse concentration)
        diversification = 1 - largest_pos

        # -----------------------------
        # METRICS DISPLAY
        # -----------------------------
        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)

        col1.metric("Expected Return", f"{exp_return:.2%}")
        col2.metric("Volatility", f"{vol:.2%}")
        col3.metric("Sharpe Ratio", f"{sharpe:.2f}")

        col4.metric("Max Drawdown", f"{max_dd:.2%}")
        col5.metric("Largest Position", f"{largest_pos:.2%}")
        col6.metric("Diversification Score", f"{diversification:.2%}")

        st.divider()

        # -----------------------------
        # ALLOCATION SNAPSHOT
        # -----------------------------
        st.subheader("Allocation Snapshot")

        if st.button("View Allocation Chart"):

            if weights is None or len(weights) == 0:
                st.warning("No weights available to plot.")
            else:
                df_alloc = (
                    weights.reset_index()
                           .rename(columns={"index": "Ticker", 0: "Weight"})
                )

                # Ensure correct column names
                if "Ticker" not in df_alloc.columns or "Weight" not in df_alloc.columns:
                    st.error("Allocation data is malformed.")
                else:
                    st.bar_chart(df_alloc, x="Ticker", y="Weight")

        st.divider()

        st.caption("Tab 1 — Summary Overview")
