import streamlit as st
import pandas as pd
import yfinance as yf
import sys, os

# ---------------------------------------------------
# FIX PATH BEFORE IMPORTS
# ---------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------
# IMPORTS (NOW SAFE)
# ---------------------------------------------------
from utils.optimizer_core import run_optimizer
from utils.data_loader import load_price_data

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
# ---------------------------------------------------
# CLEAN HEADER
# ---------------------------------------------------
st.markdown(
    """
    <div style="padding: 20px 0 10px 0; border-bottom: 1px solid #444;">
        <h1 style="margin-bottom: 0; color: #2E86C1;">Portfolio Optimizer Dashboard</h1>
        <p style="color: #888; font-size: 16px;">
            A professional multi‑factor portfolio analysis suite with optimization, risk modeling, and AI insights.
        </p>
    </div>
    """,
    unsafe_allow_html=True
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

investment_amount = st.sidebar.slider(
    "How much would you like to invest ($)?",
    min_value=1000,
    max_value=1000000,
    value=50000,
    step=5000,
    format="%d"
)

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

    if prices is None:
        st.error("Failed to load price data.")
        st.stop()

    try:
        adj = prices.xs("Adj Close", level=1, axis=1)
        if adj.dropna(how="all").empty:
            st.error("No Adj Close values found.")
            st.stop()
    except Exception:
        st.error("Adj Close missing.")
        st.stop()

    # Run optimizer
    model = run_optimizer(prices, investment_amount=investment_amount)

    if model is None:
        st.error("Optimization failed.")
        st.stop()

    sector_weights = model.get("sector_weights", None)

# ---------------------------------------------------
# TABS
# ---------------------------------------------------
tabs = st.tabs([
    "Summary",
    "Efficient Frontier",
    "Weights & Shares",
    "Sector Exposure",
    "Drawdown Analysis",
    "Monte Carlo",
    "Rebalancing Backtest",
    "AI Commentary",
    "Buy Analysis"
])

if model is not None and prices is not None:
    render_tab1(tabs[0], prices, model)
    render_tab2(tabs[1], prices, model)
    render_tab3(tabs[2], prices, model)
    render_tab4(tabs[3], prices, model)
    render_tab5(tabs[4], prices, model)
    render_tab6(tabs[5], prices, model)
    render_tab7(tabs[6], prices, model)
    render_tab8(tabs[7], model, sector_weights)
    render_tab9(tabs[8], prices)
else:
    tabs[0].info("Set your settings in the sidebar and click **Run Optimization**.")
