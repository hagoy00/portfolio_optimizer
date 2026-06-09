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
# INPUTS
# ---------------------------------------------------
default_start = datetime.date(2020, 1, 1)
default_end = datetime.date.today()

tickers_raw = st.text_input(
    "Tickers (comma separated)",
    "AAPL, TSLA, NVDA, AMZN, GOOG, WFC,APP,SANA,PCG,PONY,BYDDF,GELHY"
)
start_date = st.date_input("Start Date", default_start)
end_date = st.date_input("End Date", default_end)

# ---------------------------------------------------
# RUN OPTIMIZATION
# ---------------------------------------------------
if st.button("Run Optimization"):

    tickers = clean_ticker_input(tickers_raw)

    # Network test
    try:
        socket.gethostbyname("query1.finance.yahoo.com")
    except:
        st.error("DNS failure")
        st.stop()

    # Download data
    raw = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True
    )

    full_prices = load_full_prices_from_raw(raw, tickers)
    if full_prices is None or full_prices.empty:
        st.error("No valid price data found.")
        st.stop()

    prices = extract_adj_close(full_prices)
    returns = compute_returns(prices)

    # ---------------------------------------------------
    # OPTIMIZER ENGINE
    # ---------------------------------------------------
    max_sharpe, w_sharpe = max_sharpe_portfolio(returns)
    min_vol, w_minvol = min_vol_portfolio(returns)
    w_rp = risk_parity_weights(returns)

    perf = portfolio_performance(w_sharpe, returns)
    dd = compute_drawdown(prices)
    mc = monte_carlo_simulation(prices, weights=w_sharpe)
    rb = rebalancing_backtest(prices, w_sharpe, freq="ME")
    frontier = compute_frontier(returns, n=3000)

    # Sector map
    sector_map = {
        "AAPL": "Technology",
        "TSLA": "Consumer Discretionary",
        "NVDA": "Technology",
        "AMZN": "Consumer Discretionary",
        "GOOG": "Communication Services",
        "WFC": "Financials"
    }
    sector_weights = compute_sector_exposure(w_sharpe, sector_map)

    # Shares
    latest_prices = prices.iloc[-1]
    portfolio_value = 25000
    shares = {t: (portfolio_value * w_sharpe[t]) / latest_prices[t] for t in w_sharpe}

    # Buy score
    score = buy_score(prices)

    # Save model
    st.session_state["model"] = {
        "weights": w_sharpe,
        "minvol_weights": w_minvol,
        "risk_parity": w_rp,
        "performance": perf,
        "frontier": frontier,
        "drawdown": dd,
        "montecarlo": mc,
        "rebalancing": rb,
        "shares": shares,
        "sector_weights": sector_weights,
        "buy_score": score
    }

    st.session_state["prices"] = prices
    st.session_state["full_prices"] = full_prices

    st.success("Optimization complete.")

# ---------------------------------------------------
# RENDER TABS
# ---------------------------------------------------
if "model" in st.session_state:

    model = st.session_state["model"]
    prices = st.session_state["prices"]
    full_prices = st.session_state["full_prices"]

    tabs = st.tabs([
        "Summary",
        "Efficient Frontier",
        "Weights & Shares",
        "Sector Exposure",
        "Drawdown Analysis",
        "Monte Carlo Simulation",
        "Rebalancing Backtest",
        "AI Commentary",
        "Is This Stock a Good Buy?"
    ])

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = tabs

    render_tab1(tab1, model)
    render_tab2(tab2, prices, model)
    render_tab3(tab3, prices, model)
    render_tab4(tab4, model["sector_weights"])
    render_tab5(tab5, prices, model)
    render_tab6(tab6, prices, model)
    render_tab7(tab7, prices, model)
    render_tab8(tab8, model, model["sector_weights"])
    render_tab9(tab9, full_prices)

else:
    st.info("Enter tickers and click Run Optimization.")
