import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Loaders
from utils.data_loader import load_price_data, load_returns_data
from utils.fundamentals_loader import load_fundamentals
from utils.optimizer_core import run_optimizer
from utils.buy_analysis import run_buy_analysis
from utils.analytics import run_monte_carlo_simulation

# ---------------------------------------------------------
# Sticky Title
# ---------------------------------------------------------
st.set_page_config(page_title="Portfolio Optimizer Dashboard", layout="wide")

st.markdown("""
    <style>
        .sticky-title {
            position: sticky;
            top: 0;
            background-color: white;
            padding: 14px 0 14px 0;
            margin: 0;
            font-size: 32px;
            font-weight: 700;
            z-index: 999999;
            border-bottom: 1px solid #e0e0e0;
        }
        div[data-testid="stAppViewBlockContainer"] {
            overflow: visible !important;
        }
    </style>

    <div class="sticky-title">Portfolio Optimizer Dashboard</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Inputs
# ---------------------------------------------------------
st.sidebar.header("Configuration")
# ---------------------------------------------------------
# Sidebar Inputs
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

tickers_input = st.sidebar.text_input(
    "Enter your stock tickers (comma-separated)",
    placeholder="AAPL, MSFT, NVDA"
)

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

if not tickers:
    st.sidebar.info("Please enter at least one ticker.")
    st.stop()


if not tickers:
    st.sidebar.info("Please type a ticker and press Enter.")
    st.stop()

# Date range
end_date = st.sidebar.date_input("End Date", value=datetime.today())
start_date = st.sidebar.date_input("Start Date", value=end_date - timedelta(days=365))

if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

# Heavy computation controls
st.sidebar.subheader("Analysis Controls")
run_button = st.sidebar.button("Run Analysis")

# Monte Carlo settings
mc_sims = st.sidebar.slider("Monte Carlo Simulations", 200, 3000, 500)
mc_horizon = st.sidebar.slider("Monte Carlo Horizon (days)", 50, 500, 252)

# ---------------------------------------------------------
# Auto-run Light Model Builder
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

# Equal weights for light tabs
weights = np.array([1 / len(tickers)] * len(tickers))

# Fundamentals
fundamentals = load_fundamentals(tickers)

# Drawdown
def compute_drawdown(prices):
    adj = prices.xs("Adj Close", level=1, axis=1, drop_level=False)
    adj_simple = adj.droplevel(1, axis=1)
    w = np.array([1 / adj_simple.shape[1]] * adj_simple.shape[1])
    portfolio = (adj_simple * w).sum(axis=1)
    cum = portfolio / portfolio.iloc[0]
    dd = (cum - cum.cummax()) / cum.cummax()
    return pd.DataFrame({"Drawdown": dd})

drawdown_df = compute_drawdown(prices)

# Performance
def compute_performance(returns, weights):
    port_ret = returns.dot(weights)
    mu = port_ret.mean() * 252
    vol = port_ret.std() * np.sqrt(252)
    sharpe = mu / vol if vol > 0 else 0
    return {"expected_return": mu, "volatility": vol, "sharpe": sharpe}

performance = compute_performance(returns, weights)

# Sector weights
def compute_sector_weights(weights, tickers):
    sector_map = {
        "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
        "AMZN": "Consumer Discretionary", "GOOG": "Communication Services",
        "META": "Communication Services", "TSLA": "Consumer Discretionary",
        "JPM": "Financials", "XOM": "Energy"
    }
    sectors = [sector_map.get(t, "Other") for t in tickers]
    df = pd.DataFrame({"Ticker": tickers, "Weight": weights, "Sector": sectors})
    return df.groupby("Sector")["Weight"].sum().to_dict()

sector_weights = compute_sector_weights(weights, tickers)

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "Overview", "Performance", "Risk", "Sectors",
    "Fundamentals", "Optimizer", "Weights",
    "AI Commentary", "Buy Analysis"
])

# ---------------------------------------------------------
# LIGHT TABS (Auto-run)
# ---------------------------------------------------------
with tab1:
    st.subheader("Overview")
    st.write(prices.tail())

with tab2:
    st.subheader("Performance")
    st.write(performance)

with tab3:
    st.subheader("Risk & Drawdown")
    st.line_chart(drawdown_df)

with tab4:
    st.subheader("Sector Exposure")
    st.write(sector_weights)

with tab5:
    st.subheader("Fundamentals")
    st.write(pd.DataFrame(fundamentals).T)

with tab7:
    st.subheader("Weights")
    st.write(pd.DataFrame({"Ticker": tickers, "Weight": weights}))

with tab8:
    st.subheader("AI Commentary")
    st.write("AI Commentary based on performance, risk, and fundamentals will go here.")

# ---------------------------------------------------------
# HEAVY TABS (Run Analysis button required)
# ---------------------------------------------------------
with tab6:
    st.subheader("Optimizer")
    if not run_button:
        st.info("Run Analysis to generate optimizer results.")
    else:
        opt_results = run_optimizer(returns, cov)
        st.write(opt_results)

with tab9:
    st.subheader("Buy / Hold / Sell Analysis")
    if not run_button:
        st.info("Run Analysis to generate buy analysis.")
    else:
        buy_results = run_buy_analysis(tickers, fundamentals, performance)
        st.write(buy_results)

# Monte Carlo (shown inside Optimizer or Buy tab depending on your design)
if run_button:
    mc_df = run_monte_carlo_simulation(returns, mc_sims, mc_horizon)
    st.subheader("Monte Carlo Simulation")
    st.line_chart(mc_df)
else:
    st.info("Run Analysis to generate Monte Carlo simulation.")
