import streamlit as st
import pandas as pd
import yfinance as yf

# -----------------------------
# IMPORT TAB RENDERERS
# -----------------------------
from components.tab1_summary import render_summary_tab
from components.tab2_frontier import render_frontier_tab
from components.tab3_weights import render_weights_tab
from components.tab4_sector import render_sector_tab
from components.tab5_drawdown import render_drawdown_tab
from components.tab6_montecarlo import render_montecarlo_tab
from components.tab7_rebalancing import render_rebalancing_tab
from components.tab8_ai_commentary import render_ai_commentary_tab
from components.tab9_buy_analysis import render_buy_analysis_tab

# -----------------------------
# IMPORT UTILS
# -----------------------------
from utils.data_loader import (
    clean_ticker_input,
    load_full_prices_from_raw,
    extract_adj_close,
)

from utils.optimizer_core import run_optimizer


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Portfolio Optimizer",
    layout="wide",
)

st.title("Portfolio Optimizer Dashboard")


# -----------------------------
# USER INPUTS
# -----------------------------
tickers_raw = st.text_input("Tickers (comma separated)")
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date")
with col2:
    end_date = st.date_input("End Date")

tickers = clean_ticker_input(tickers_raw)

if not tickers:
    st.info("Enter at least one ticker to begin.")
    st.stop()


# -----------------------------
# LOAD RAW PRICES
# -----------------------------
full_prices = load_full_prices_from_raw(tickers, start_date, end_date)

if full_prices is None or full_prices.empty:
    st.error("No price data returned. Check tickers and date range.")
    st.stop()


# -----------------------------
# EXTRACT ADJ CLOSE
# -----------------------------
prices = extract_adj_close(full_prices)

if prices is None or prices.empty:
    st.error("Adjusted close data missing or invalid.")
    st.stop()


# -----------------------------
# RUN OPTIMIZER
# -----------------------------
model = run_optimizer(prices)

if model is None:
    st.error("Optimizer failed — model returned None.")
    st.stop()


# -----------------------------
# TABS
# -----------------------------
tabs = st.tabs([
    "Summary",
    "Efficient Frontier",
    "Weights",
    "Sector Exposure",
    "Drawdown",
    "Monte Carlo",
    "Rebalancing",
    "AI Commentary",
    "Buy Analysis",
])

with tabs[0] as tab:
    render_summary_tab(tab, prices, model)

with tabs[1] as tab:
    render_frontier_tab(tab, prices, model)

with tabs[2] as tab:
    render_weights_tab(tab, prices, model)

with tabs[3] as tab:
    render_sector_tab(tab, prices, model)

with tabs[4] as tab:
    render_drawdown_tab(tab, prices, model)

with tabs[5] as tab:
    render_montecarlo_tab(tab, prices, model)

with tabs[6] as tab:
    render_rebalancing_tab(tab, prices, model)

with tabs[7] as tab:
    render_ai_commentary_tab(tab, prices, model)

with tabs[8] as tab:
    render_buy_analysis_tab(tab, full_prices, model)
