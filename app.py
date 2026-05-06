import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from utils.data_loader import load_price_data, load_returns_data
# Components (import only those you actually have)
from components.tab8_ai_commentary import render_ai_commentary_tab
# from components.tab1_overview import render_overview_tab
# from components.tab2_performance import render_performance_tab
# from components.tab3_risk import render_risk_tab
# from components.tab4_sectors import render_sector_tab
# from components.tab5_fundamentals import render_fundamentals_tab
# from components.tab6_optimizer import render_optimizer_tab
# from components.tab7_weights import render_weights_tab
# from components.tab9_buy_analysis import render_buy_analysis_tab


# ---------------------------------------------------------
# Simple helpers (you can replace with your real ones)
# ---------------------------------------------------------
def compute_drawdown(prices: pd.DataFrame) -> pd.DataFrame:
    if prices is None or prices.empty:
        return pd.DataFrame()

    # Use equal-weighted portfolio for drawdown
    adj = prices.xs("Adj Close", level=1, axis=1, drop_level=False)
    adj_simple = adj.droplevel(1, axis=1)
    weights = np.array([1 / adj_simple.shape[1]] * adj_simple.shape[1])

    portfolio = (adj_simple * weights).sum(axis=1)
    cum = portfolio / portfolio.iloc[0]
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max

    return pd.DataFrame({"Drawdown": dd})


def compute_performance(returns: pd.DataFrame, weights: np.ndarray) -> dict:
    if returns is None or returns.empty:
        return {}

    port_ret = returns.dot(weights)
    mu = port_ret.mean() * 252
    vol = port_ret.std() * np.sqrt(252)
    sharpe = mu / vol if vol > 0 else 0.0

    return {
        "expected_return": float(mu),
        "volatility": float(vol),
        "sharpe": float(sharpe),
    }


def run_monte_carlo(prices: pd.DataFrame, weights: np.ndarray, n_sims: int = 500, horizon_days: int = 252) -> pd.DataFrame:
    if prices is None or prices.empty:
        return pd.DataFrame()

    adj = prices.xs("Adj Close", level=1, axis=1, drop_level=False)
    adj_simple = adj.droplevel(1, axis=1)
    returns = adj_simple.pct_change().dropna(how="all")

    if returns.empty:
        return pd.DataFrame()

    port_ret = returns.dot(weights)
    mu = port_ret.mean()
    sigma = port_ret.std()

    sims = []
    for _ in range(n_sims):
        shocks = np.random.normal(mu, sigma, horizon_days)
        path = (1 + shocks).cumprod()
        sims.append(path)

    mc_df = pd.DataFrame(sims).T
    return mc_df


def compute_sector_weights(weights: np.ndarray, tickers: list) -> dict:
    # Minimal static sector map; extend as needed
    sector_map = {
        "AAPL": "Technology",
        "MSFT": "Technology",
        "NVDA": "Technology",
        "AMZN": "Consumer Discretionary",
        "GOOG": "Communication Services",
        "META": "Communication Services",
        "TSLA": "Consumer Discretionary",
        "JPM": "Financials",
        "XOM": "Energy",
    }

    sectors = [sector_map.get(t, "Other") for t in tickers]
    df = pd.DataFrame({"Ticker": tickers, "Weight": weights, "Sector": sectors})
    return df.groupby("Sector")["Weight"].sum().to_dict()


def load_fundamentals(tickers: list) -> dict:
    # Placeholder: you can wire your real fundamentals loader here
    fundamentals = {}
    for t in tickers:
        fundamentals[t] = {
            "pe": None,
            "ps": None,
            "pb": None,
            "recommendation": None,
            "target_mean_price": None,
            "dividend_yield": None,
            "beta": None,
        }
    return fundamentals


# ---------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------
st.set_page_config(page_title="Portfolio Optimizer Dashboard", layout="wide")
st.title("Portfolio Optimizer Dashboard")

# Sidebar inputs
st.sidebar.header("Configuration")

tickers_input = st.sidebar.text_input(
    "Tickers (comma-separated)",
    value="AAPL, MSFT, NVDA"
)

end_date = st.sidebar.date_input("End Date", value=datetime.today())
start_date = st.sidebar.date_input("Start Date", value=end_date - timedelta(days=365))

if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

if not tickers:
    st.sidebar.error("Please enter at least one valid ticker.")
    st.stop()

# ---------------------------------------------------------
# Load prices and returns with guard clauses
# ---------------------------------------------------------
prices = load_price_data(tickers, start_date, end_date)
returns = load_returns_data(tickers, start_date, end_date)

if prices is None or prices.empty:
    st.error("Price data could not be loaded. Check tickers or date range.")
    st.stop()

if returns is None or returns.empty:
    st.error("Could not compute returns. Not enough price data.")
    st.stop()

cov = returns.cov()
if cov is None or cov.empty:
    st.error("Covariance matrix is empty. Cannot compute optimization.")
    st.stop()

# ---------------------------------------------------------
# Simple equal-weight portfolio (replace with optimizer later)
# ---------------------------------------------------------
weights = np.array([1 / len(tickers)] * len(tickers))

# ---------------------------------------------------------
# Compute sector weights, fundamentals, drawdown, performance, MC
# ---------------------------------------------------------
try:
    sector_weights = compute_sector_weights(weights, tickers)
except Exception:
    sector_weights = None

fundamentals = load_fundamentals(tickers)
drawdown_df = compute_drawdown(prices)
performance = compute_performance(returns, weights)
mc_df = run_monte_carlo(prices, weights)

# ---------------------------------------------------------
# Build model dictionary
# ---------------------------------------------------------
model = {
    "prices": prices,
    "returns": returns,
    "cov": cov,
    "weights": weights,
    "sector_weights": sector_weights,
    "fundamentals": fundamentals,
    "drawdown": drawdown_df,
    "performance": performance,
    "monte_carlo": mc_df,
    "tickers": tickers,
}

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
    [
        "Overview",
        "Performance",
        "Risk",
        "Sectors",
        "Fundamentals",
        "Optimizer",
        "Weights",
        "AI Commentary",
        "Buy Analysis",
    ]
)

# You can wire real renderers for tabs 1–7 and 9 later.
with tab1:
    st.write("Overview placeholder.")

with tab2:
    st.write("Performance placeholder.")

with tab3:
    st.write("Risk placeholder.")

with tab4:
    st.write("Sector Exposure")
    if sector_weights:
        st.write(sector_weights)
    else:
        st.write("Sector weights not available.")

with tab5:
    st.write("Fundamentals placeholder.")
    st.write(pd.DataFrame.from_dict(fundamentals, orient="index"))

with tab6:
    st.write("Optimizer placeholder.")

with tab7:
    st.write("Weights")
    st.write(pd.DataFrame({"Ticker": tickers, "Weight": weights}))

# Tab 8: AI Commentary (real implementation)
render_ai_commentary_tab(tab8, prices, model)

with tab9:
    st.write("Buy / Hold / Sell Analysis placeholder.")
