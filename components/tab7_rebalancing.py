import streamlit as st
import pandas as pd
import numpy as np

def render_rebalancing_tab(tab, prices, model):

    with tab:

        st.subheader("Portfolio Rebalancing Backtest")

        # ---- Basic checks ----
        if "returns" not in model or model["returns"] is None:
            st.warning("No returns data available for rebalancing.")
            return

        returns = model["returns"]
        tickers = model.get("tickers", [])

        if not isinstance(returns, pd.DataFrame) or returns.empty:
            st.warning("Rebalancing backtest returned no data (empty returns).")
            return

        if not tickers:
            st.warning("No tickers available for rebalancing.")
            return

        # Ensure index is datetime
        ret_df = returns[tickers].copy()
        if not isinstance(ret_df.index, pd.DatetimeIndex):
            try:
                ret_df.index = pd.to_datetime(ret_df.index)
            except Exception:
                st.warning("Rebalancing requires a DatetimeIndex. Could not convert index to datetime.")
                return

        # ---- User Inputs ----
        st.markdown("### Rebalancing Settings")

        freq = st.selectbox(
            "Rebalancing Frequency",
            ["Monthly", "Quarterly", "Annual", "Weekly"],
            index=0
        )

        freq_map = {
            "Weekly": "W",
            "Monthly": "M",
            "Quarterly": "Q",
            "Annual": "A"
        }

        rebalance_offset = freq_map.get(freq)

        if rebalance_offset is None:
            st.warning("Invalid rebalancing frequency selected.")
            return

        # ---- Prepare Data ----
        w0 = np.array([1 / len(tickers)] * len(tickers))
        portfolio_value = [1.0]
        current_weights = w0.copy()

        try:
            rebalance_dates = ret_df.resample(rebalance_offset).first().index
        except Exception as e:
            st.warning(f"Could not compute rebalance dates: {e}")
            return

        # ---- Rebalancing Loop ----
        for i in range(1, len(ret_df)):

            daily_ret = np.dot(current_weights, ret_df.iloc[i].values)
            new_value = portfolio_value[-1] * (1 + daily_ret)
            portfolio_value.append(new_value)

            if ret_df.index[i] in rebalance_dates:
                current_weights = w0.copy()

        if len(portfolio_value) != len(ret_df):
            st.warning("Rebalancing backtest returned no data.")
            return

        portfolio_series = pd.Series(
            portfolio_value,
            index=ret_df.index,
            name="Rebalanced Portfolio"
        )

        st.line_chart(portfolio_series)

        # ---- Metrics ----
        daily_pct = portfolio_series.pct_change().dropna()
        if daily_pct.empty:
            st.warning("Not enough data to compute performance metrics.")
            return

        total_return = portfolio_series.iloc[-1] - 1
        annualized_return = daily_pct.mean() * 252
        annualized_vol = daily_pct.std() * np.sqrt(252)
        sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0

        st.markdown("### Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Return", f"{total_return:.2%}")
        col2.metric("Annualized Return", f"{annualized_return:.2%}")
        col3.metric("Annualized Volatility", f"{annualized_vol:.2%}")
        col4.metric("Sharpe Ratio", f"{sharpe:.2f}")
