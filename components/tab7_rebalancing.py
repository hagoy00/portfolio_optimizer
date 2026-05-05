import streamlit as st
import pandas as pd
import numpy as np

def render_rebalancing_tab(tab, prices, model):

    with tab:

        st.subheader("Portfolio Rebalancing Backtest")

        tickers = model["tickers"]
        weights = model["weights"]
        returns = model["returns"]

        if len(tickers) == 0:
            st.warning("No tickers available for rebalancing.")
            return

        # ---------------------------------------------------------
        # User Inputs
        # ---------------------------------------------------------
        st.markdown("### Rebalancing Settings")

        freq = st.selectbox(
            "Rebalancing Frequency",
            ["Monthly", "Quarterly", "Annual", "Weekly"],
            index=0
        )

        # Convert frequency to pandas offset
        freq_map = {
            "Weekly": "W",
            "Monthly": "M",
            "Quarterly": "Q",
            "Annual": "A"
        }
        rebalance_offset = freq_map[freq]

        # ---------------------------------------------------------
        # Prepare Data
        # ---------------------------------------------------------
        price_df = prices.copy()
        ret_df = returns[tickers].copy()

        # Initial equal weights
        w0 = np.array([1 / len(tickers)] * len(tickers))

        # Portfolio value tracking
        portfolio_value = [1.0]
        current_weights = w0.copy()

        # ---------------------------------------------------------
        # Rebalancing Loop
        # ---------------------------------------------------------
        rebalance_dates = ret_df.resample(rebalance_offset).first().index

        for i in range(1, len(ret_df)):

            # Apply daily returns
            daily_ret = np.dot(current_weights, ret_df.iloc[i].values)
            new_value = portfolio_value[-1] * (1 + daily_ret)
            portfolio_value.append(new_value)

            # Rebalance on scheduled dates
            if ret_df.index[i] in rebalance_dates:
                current_weights = w0.copy()

        # ---------------------------------------------------------
        # Results
        # ---------------------------------------------------------
        portfolio_series = pd.Series(
            portfolio_value,
            index=ret_df.index,
            name="Rebalanced Portfolio"
        )

        st.line_chart(portfolio_series)

        # ---------------------------------------------------------
        # Metrics
        # ---------------------------------------------------------
        total_return = portfolio_series.iloc[-1] - 1
        annualized_return = portfolio_series.pct_change().mean() * 252
        annualized_vol = portfolio_series.pct_change().std() * np.sqrt(252)
        sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0

        st.markdown("### Performance Metrics")
        st.metric("Total Return", f"{total_return:.2%}")
        st.metric("Annualized Return", f"{annualized_return:.2%}")
        st.metric("Annualized Volatility", f"{annualized_vol:.2%}")
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
