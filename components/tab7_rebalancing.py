import streamlit as st
import pandas as pd
from utils.optimizer_core import rebalancing_backtest


def render_rebalancing_tab(tab, prices, model):
    tab.markdown("## Rebalancing Backtest")

    if model is None:
        tab.error("Model is missing.")
        return

    # ---------- Frequency selector ----------
    freq_label = tab.selectbox(
        "Rebalancing Frequency",
        options=["Monthly", "Quarterly", "Annual", "Weekly"],
        index=0
    )

    freq_map = {
        "Monthly": "M",
        "Quarterly": "Q",
        "Annual": "Y",
        "Weekly": "W",
    }
    freq = freq_map[freq_label]

    # ---------- Weights ----------
    target_weights = model.get("weights")
    if target_weights is None:
        tab.error("No weights found in model.")
        return

    w = pd.Series(target_weights)

    # Align to tickers in prices
    if isinstance(prices.columns, pd.MultiIndex):
        tickers = prices.columns.get_level_values("Ticker").unique()
    else:
        tickers = prices.columns
    w = w.reindex(tickers).fillna(0.0)

    # ---------- Run backtest ----------
    try:
        result = rebalancing_backtest(prices, w, freq=freq)
    except Exception as e:
        tab.error(f"Rebalancing failed: {e}")
        return

    if result is None:
        tab.warning("Rebalancing backtest returned no data.")
        return

    equity = result["equity_curve"]
    metrics = result["metrics"]
    turnover_df = result["turnover"]

    if equity is None or equity.empty:
        tab.warning("No equity curve generated.")
        return

    # ---------- Chart ----------
    tab.markdown("### Portfolio Value Over Time")
    tab.line_chart(equity["Portfolio Value"])

    # ---------- KPIs ----------
    col1, col2, col3 = tab.columns(3)
    col4, col5, col6 = tab.columns(3)

    col1.metric("CAGR", f"{metrics['CAGR']*100:.2f}%" if pd.notna(metrics["CAGR"]) else "N/A")
    col2.metric("Volatility (Ann.)", f"{metrics['Volatility']*100:.2f}%" if pd.notna(metrics["Volatility"]) else "N/A")
    col3.metric("Sharpe", f"{metrics['Sharpe']:.2f}" if pd.notna(metrics["Sharpe"]) else "N/A")

    col4.metric("Max Drawdown", f"{metrics['Max Drawdown']*100:.2f}%" if pd.notna(metrics["Max Drawdown"]) else "N/A")
    col5.metric("Rebalances", f"{metrics['Rebalance Count']}")
    col6.metric("Avg Turnover", f"{metrics['Average Turnover']*100:.2f}%" if pd.notna(metrics["Average Turnover"]) else "N/A")

    # ---------- Tables ----------
    with tab.expander("Equity Curve (Recent)"):
        tab.dataframe(equity.tail(), use_container_width=True)

    with tab.expander("Turnover by Rebalance Date"):
        if turnover_df is not None and not turnover_df.empty:
            tab.dataframe(turnover_df, use_container_width=True)
        else:
            tab.write("No turnover data available.")
