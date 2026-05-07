import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from utils.data_loader import load_price_data, load_returns_data
from utils.fundamentals_loader import load_fundamentals
from utils.optimizer_core import run_optimizer
from utils.buy_analysis import run_buy_analysis
from utils.analytics import run_monte_carlo_simulation

# ---------------------------------------------------------
# Page config + sticky header
# ---------------------------------------------------------
st.set_page_config(page_title="Portfolio Optimizer Dashboard", layout="wide")

st.markdown("""
<style>

.fixed-title {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    background-color: white;
    padding: 14px 24px;
    font-size: 30px;
    font-weight: 700;
    color: #007BFF;
    border-bottom: 1px solid #E5E5E5;
    z-index: 9999;
    line-height: 1.2;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
}

.main .block-container {
    padding-top: 80px !important;
}

div[data-testid="stAppViewContainer"] {
    overflow: visible !important;
}

</style>

<div class="fixed-title">Portfolio Optimizer Dashboard</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------
st.sidebar.header("Configuration")

tickers_input = st.sidebar.text_area(
    "Enter your stock tickers (one per line)",
    placeholder="AAPL\nMSFT\nNVDA"
)
tickers = [t.strip().upper() for t in tickers_input.split("\n") if t.strip()]

if not tickers:
    st.sidebar.info("Please enter at least one ticker.")
    st.stop()

end_date = st.sidebar.date_input("End Date", value=datetime.today())
start_date = st.sidebar.date_input("Start Date", value=end_date - timedelta(days=365))

if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

st.sidebar.subheader("Analysis Controls")
run_button = st.sidebar.button("Run Analysis")

mc_sims = st.sidebar.slider("Monte Carlo Simulations", 200, 3000, 500)
mc_horizon = st.sidebar.slider("Monte Carlo Horizon (days)", 50, 500, 252)

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
prices = load_price_data(tickers, start_date, end_date)
returns = load_returns_data(tickers, start_date, end_date)

if prices is None or prices.empty:
    st.error("Price data could not be loaded.")
    st.stop()

if returns is None or returns.empty:
    st.error("Could not compute returns.")
    st.stop()

cov = returns.cov()
if cov is None or cov.empty:
    st.error("Covariance matrix is empty.")
    st.stop()

weights = np.array([1 / len(tickers)] * len(tickers))
fundamentals = load_fundamentals(tickers)

# ---------------------------------------------------------
# Drawdown
# ---------------------------------------------------------
def compute_drawdown(prices):
    adj = prices.xs("Adj Close", level=1, axis=1, drop_level=False)
    adj_simple = adj.droplevel(1, axis=1)
    w = np.array([1 / adj_simple.shape[1]] * adj_simple.shape[1])
    portfolio = (adj_simple * w).sum(axis=1)
    cum = portfolio / portfolio.iloc[0]
    dd = (cum - cum.cummax()) / cum.cummax()
    return pd.DataFrame({"Drawdown": dd})

drawdown_df = compute_drawdown(prices)

# ---------------------------------------------------------
# Performance
# ---------------------------------------------------------
def compute_performance(returns, weights):
    port_ret = returns.dot(weights)
    mu = port_ret.mean() * 252
    vol = port_ret.std() * np.sqrt(252)
    sharpe = mu / vol if vol > 0 else 0
    return {"expected_return": mu, "volatility": vol, "sharpe": sharpe}

performance = compute_performance(returns, weights)

# ---------------------------------------------------------
# Sector weights
# ---------------------------------------------------------
def compute_sector_weights(weights, tickers):
    sector_map = {
        "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
        "AMZN": "Consumer Discretionary", "GOOG": "Communication Services",
        "META": "Communication Services", "TSLA": "Consumer Discretionary",
        "JPM": "Financials", "XOM": "Energy"
    }
    sectors = [sector_map.get(t, "Other") for t in tickers]
    df = pd.DataFrame({"Ticker": tickers, "Weight": weights
