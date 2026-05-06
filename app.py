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
# Sticky Title (blue + fixed)
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
            padding: 16px 0 16px 20px;
            font-size: 32px;
            font-weight: 700;
            color: #007BFF;
            border-bottom: 1px solid #e0e0e0;
            z-index: 99999;
        }
        .main .block-container {
            padding-top: 100px !important;
        }
    </style>

    <div class="fixed-title">Portfolio Optimizer Dashboard</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Inputs (ONE ticker input)
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
# Load Data
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

# Fundamentals (now includes full_prices)
#fundamentals = load_fundamentals(tickers, full_prices=prices)
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
# Sector Weights
# ---------------------------------------------------------
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
tab1, tab2, tab3, tab4, tab5, tab7, tab8, tab9, tab6 = st.tabs([
    "Overview", "Performance", "Risk", "Sectors",
    "Fundamentals", "Weights",
    "AI Commentary", "Buy Analysis",
    "Optimizer"
])

# ---------------------------------------------------------
# OVERVIEW
# ---------------------------------------------------------
with tab1:
    st.markdown("<h2 style='color:#1E90FF;'>Optimizer Dashboard Report</h2>", unsafe_allow_html=True)
    st.subheader("Overview")
    st.dataframe(prices.tail())

# ---------------------------------------------------------
# PERFORMANCE (formatted)
# ---------------------------------------------------------
with tab2:
    st.subheader("Performance Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Expected Return", f"{performance['expected_return']:.2%}")
    col2.metric("Volatility", f"{performance['volatility']:.2%}")
    col3.metric("Sharpe Ratio", f"{performance['sharpe']:.2f}")

# ---------------------------------------------------------
# RISK & DRAWDOWN
# ---------------------------------------------------------
with tab3:
    st.subheader("Risk & Drawdown")
    st.line_chart(drawdown_df)

# ---------------------------------------------------------
# SECTOR EXPOSURE (chart)
# ---------------------------------------------------------
with tab4:
    st.subheader("Sector Exposure")
    sector_df = pd.DataFrame.from_dict(sector_weights, orient="index", columns=["Weight"])
    st.bar_chart(sector_df)

# ---------------------------------------------------------
# FUNDAMENTALS
# ---------------------------------------------------------
with tab5:
    st.subheader("Fundamentals")
    fundamentals_df = pd.DataFrame(fundamentals).T.drop("full_prices", errors="ignore")
    st.dataframe(fundamentals_df)

# ---------------------------------------------------------
# OPTIMIZER (button + cached)
# ---------------------------------------------------------
@st.cache_data(show_spinner=True)
def run_optimizer_cached(returns, cov):
    return run_optimizer(returns, cov)

with tab6:
    st.subheader("Optimizer & Monte Carlo")

    if st.button("Run Optimization"):
        opt_results = run_optimizer_cached(returns, cov)
        st.success("Optimization complete!")
        st.write(opt_results)

    else:
        opt_results = run_optimizer_cached(returns, cov)
        st.success("Optimization complete!")
        st.write(opt_results)

        mc_df = run_monte_carlo_simulation(returns, mc_sims, mc_horizon)
        st.subheader("Monte Carlo Simulation")
        st.line_chart(mc_df)

# ---------------------------------------------------------
# WEIGHTS
# ---------------------------------------------------------
with tab7:
    st.subheader("Weights")
    weights_df = pd.DataFrame({"Ticker": tickers, "Weight": weights})
    st.dataframe(weights_df)

# ---------------------------------------------------------
# AI COMMENTARY
# ---------------------------------------------------------
with tab8:
    st.subheader("AI Commentary")
    st.write("AI Commentary based on performance, risk, and fundamentals will go here.")

# ---------------------------------------------------------
# BUY ANALYSIS
# ---------------------------------------------------------
with tab9:
    st.subheader("Buy / Hold / Sell Analysis")
    if not run_button:
        st.info("Run Analysis to generate buy analysis.")
    else:
        buy_results = run_buy_analysis(tickers, fundamentals, prices)
        st.dataframe(buy_results)
