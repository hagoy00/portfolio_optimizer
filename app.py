import streamlit as st
import datetime
import sys
import socket
import ssl
import yfinance as yf

from utils.data_loader import (
    clean_ticker_input,
    load_full_prices_from_raw,
    extract_adj_close
)

from utils.optimizer import (
    compute_returns,
    max_sharpe_portfolio,
    min_vol_portfolio,
    risk_parity_weights,
    portfolio_performance,
    compute_drawdown,
    monte_carlo_simulation,
    rebalancing_backtest,
    compute_sector_exposure,
    buy_score,
    compute_frontier
)

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
# DEBUG INFO
# ---------------------------------------------------
st.write("DEBUG optimizer loaded:", sys.executable)

st.title("Portfolio Optimizer Dashboard")

# ---------------------------------------------------
# SIDEBAR INPUTS (clean UI)
# ---------------------------------------------------
with st.sidebar:
    st.header("Portfolio Controls")

    default_start = datetime.date(2020, 1, 1)
    default_end = datetime.date.today()

    tickers_raw = st.text_input(
        "Tickers (comma separated)",
        "AAPL, TSLA, NVDA, AMZN, GOOG, WFC, APP, SANA, PCG, PONY, BYDDF, GELHY"
    )

    start_date = st.date_input("Start Date", default_start)
    end_date = st.date_input("End Date", default_end)

    # GUARD CLAUSE — prevent future date crash
    today = datetime.date.today()
    if end_date > today:
        st.warning(f"End date {end_date} is in the future. Using {today} instead.")
        end_date = today

    # PURCHASE POWER
    purchase_power = st.selectbox(
        "Purchase Power",
        ["10,000", "25,000", "50,000", "100,000", "250,000", "500,000", "1,000,000"]
    )
    purchase_power = int(purchase_power.replace(",", ""))

    # GUARD CLAUSE — minimum capital
    if purchase_power < 1000:
        st.error("Purchase power must be at least $1,000.")
        st.stop()

    # Rebalancing frequency (used later)
    freq = st.selectbox("Rebalancing Frequency", ["ME", "W", "Q"])

    # Store frequency in session_state to prevent tab flipping
    st.session_state["rebalance_freq"] = freq

    run_opt = st.button("Run Optimization")


# ---------------------------------------------------
# RUN OPTIMIZATION
# ---------------------------------------------------
if run_opt:

    tickers = clean_ticker_input(tickers_raw)

    # Network test
    try:
        socket.gethostbyname("query1.finance.yahoo.com")
    except:
        st.error("DNS failure
