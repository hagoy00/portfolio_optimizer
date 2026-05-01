import streamlit as st
import pandas as pd
from utils.data_loader import load_price_data
from utils.optimizer import run_optimizer

# Import tab components
from components.tab1_summary import render_tab1
from components.tab2_frontier import render_tab2
from components.tab3_weights import render_tab3
from components.tab4_sector import render_tab4
from components.tab5_drawdown import render_tab5
from components.tab6_montecarlo import render_tab6
from components.tab7_rebalancing import render_tab7
from components.tab8_ai_commentary import render_tab8
from components.tab9_buy_analysis import render_tab9

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Portfolio Optimizer Dashboard",
    layout="wide",
)

st.title("Portfolio Optimizer Dashboard")

# ---------------------------------------------------
# SIDEBAR — Ticker Input
# ---------------------------------------------------
st.sidebar.header("Portfolio Settings")

tickers_input = st.sidebar.text_input(
    "Tickers (comma-separated)",
    value="AAPL, MSFT, AMZN, GOOG, META"
)

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

run_button = st.sidebar.button("Run Optimization")

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
prices = None
model = None
sector_weights = None

if run_button:
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    prices = load_price_data(tickers, start_date, end_date)

    if prices is None or prices.empty:
        st.error("Failed to load price data.")
    else:
        model = run_optimizer(prices)
        if model is None:
            st.error("Optimization failed.")
        else:
            sector_weights =
