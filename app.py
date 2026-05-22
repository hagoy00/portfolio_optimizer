import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go

from utils.data_loader import load_price_data
from utils.fundamentals_loader import load_fundamentals
from utils.optimizer_core import run_optimizer
from utils.buy_analysis import run_buy_analysis
from utils.analytics import run_monte_carlo_simulation

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(page_title="Portfolio Optimizer Dashboard", layout="wide")

# ---------------------------------------------------------
# Helper: robust price & returns loader
# ---------------------------------------------------------
def load_prices_and_returns(tickers):
    import yfinance as yf
    import pandas as pd

    data = yf.download(tickers, period="1y", auto_adjust=False)

    if data.empty:
        raise ValueError("Yahoo Finance returned no data for the given tickers.")

    # MultiIndex (multiple tickers)
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(0):
            adj_close = data["Adj Close"]
        elif "Close" in data.columns.get_level_values(0):
            adj_close = data["Close"]
        else:
            raise KeyError("Neither 'Adj Close' nor 'Close' found in downloaded data.")
    else:
        # Single ticker
        if "Adj Close" in data.columns:
            adj_close = data["Adj Close"].to_frame()
        elif "Close" in data.columns:
            adj_close = data["Close"].to_frame()
        else:
            raise KeyError("Neither 'Adj Close' nor 'Close' found for single ticker.")

    # Remove empty columns (invalid tickers)
    adj_close = adj_close.dropna(axis=1, how="all")

    if adj_close.empty:
        raise ValueError("All tickers returned empty price data.")

    returns_df = adj_close.pct_change().dropna()
    latest_prices = adj_close.iloc[-1]

    return latest_prices, returns_df

# ---------------------------------------------------------
# Sidebar: inputs
# ---------------------------------------------------------
st.sidebar.header("Portfolio Settings")

# Normalize tickers from text input (spaces or commas)
tickers_raw = st.sidebar.text_input(
    "Tickers (space or comma separated)",
    value="AAPL MSFT GOOGL"
)

tickers = [t.strip().upper() for t in tickers_raw.replace(",", " ").split() if t.strip()]

run_button = st.sidebar.button("Run Analysis")

# ---------------------------------------------------------
# RUN ANALYSIS — MUST BE ABOVE ALL TABS
# ---------------------------------------------------------
if run_button and tickers:
    try:
        latest_prices, returns_df = load_prices_and_returns(tickers)
        fundamentals = load_fundamentals(tickers)

        st.session_state["latest_prices"] = latest_prices
        st.session_state["returns_df"] = returns_df
        st.session_state["fundamentals"] = fundamentals
        st.session_state["prices"] = latest_prices
        st.session_state["tickers"] = tickers
    except Exception as e:
        st.error(f"Data load failed: {e}")
        st.stop()

# ---------------------------------------------------------
# Precompute portfolio metrics (if data available)
# ---------------------------------------------------------
annual_return = np.nan
annual_volatility = np.nan
sharpe_ratio = np.nan
max_drawdown = np.nan
portfolio_beta = np.nan
diversification_score = np.nan

if "returns_df" in st.session_state:
    try:
        returns_df = st.session_state["returns_df"]

        # Use equal weight if optimizer not yet run
        if "portfolio_weights" in st.session_state:
            w = np.array(st.session_state["portfolio_weights"])
        else:
            w = np.repeat(1 / returns_df.shape[1], returns_df.shape[1])

        w = w / w.sum()
        port_ret = (returns_df @ w)

        # Annualized metrics (252 trading days)
        annual_return = (1 + port_ret.mean()) ** 252 - 1
        annual_volatility = port_ret.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else np.nan

        # Max drawdown
        cum = (1 + port_ret).cumprod()
        peak = cum.cummax()
        drawdown = (cum - peak) / peak
        max_drawdown = drawdown.min()

        # Simple diversification proxy: 10 - average pairwise correlation * 10
        corr = returns_df.corr()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        avg_corr = upper.stack().mean() if not upper.stack().empty else 0
        diversification_score = max(0, 10 - avg_corr * 10)

        # Beta vs SPY
        try:
            spy = yf.download("SPY", period="1y")["Adj Close"].pct_change().dropna()
            aligned = pd.concat([port_ret, spy], axis=1).dropna()
            cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1]
            var_mkt = np.var(aligned.iloc[:, 1])
            portfolio_beta = cov / var_mkt if var_mkt != 0 else np.nan
        except Exception:
            portfolio_beta = np.nan

    except Exception as e:
        st.error(f"Portfolio metrics failed: {e}")
        st.stop()

# ---------------------------------------------------------
# TABS START HERE
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "Overview",
    "Performance",
    "Risk",
    "Sectors",
    "Fundamentals",
    "Weights",
    "AI Commentary",
    "Buy Analysis",
    "Optimizer"
])

# ---------------------------------------------------------
# Tab 1 — Overview
# ---------------------------------------------------------
with tab1:
    st.subheader("Portfolio Overview")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Annual Return", f"{annual_return:.2%}" if not np.isnan(annual_return) else "N/A")
    with col2:
        st.metric("Volatility", f"{annual_volatility:.2%}" if not np.isnan(annual_volatility) else "N/A")
    with col3:
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}" if not np.isnan(sharpe_ratio) else "N/A")

    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("Max Drawdown", f"{max_drawdown:.2%}" if not np.isnan(max_drawdown) else "N/A")
    with col5:
        st.metric("Beta vs SPY", f"{portfolio_beta:.2f}" if not np.isnan(portfolio_beta) else "N/A")
    with col6:
        st.metric("Diversification Score", f"{diversification_score:.1f}/10" if not np.isnan(diversification_score) else "N/A")

# ---------------------------------------------------------
# Tab 2 — Performance
# ---------------------------------------------------------
with tab2:
    st.subheader("Performance")

    if "returns_df" not in st.session_state:
        st.info("Run Analysis first to load data.")
        st.stop()

    returns_df = st.session_state["returns_df"]

    if "portfolio_weights" in st.session_state:
        w = np.array(st.session_state["portfolio_weights"])
    else:
        w = np.repeat(1 / returns_df.shape[1], returns_df.shape[1])

    w = w / w.sum()
    port_ret = (returns_df @ w)
    cum_port = (1 + port_ret).cumprod()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum_port.index, y=cum_port.values, mode="lines", name="Portfolio"))
    fig.update_layout(title="Cumulative Portfolio Return", xaxis_title="Date", yaxis_title="Cumulative Return")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# Tab 3 — Risk
# ---------------------------------------------------------
with tab3:
    st.subheader("Risk")

    if "returns_df" not in st.session_state:
        st.info("Run Analysis first to load data.")
        st.stop()

    returns_df = st.session_state["returns_df"]

    if "portfolio_weights" in st.session_state:
        w = np.array(st.session_state["portfolio_weights"])
    else:
        w = np.repeat(1 / returns_df.shape[1], returns_df.shape[1])

    w = w / w.sum()
    port_ret = (returns_df @ w)

    st.write("**Daily Return Distribution**")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=port_ret, nbinsx=50, name="Portfolio Returns"))
    fig.update_layout(bargap=0.1)
    st.plotly_chart(fig, use_container_width=True)

    st.write("**Rolling Volatility (30-day)**")
    rolling_vol = port_ret.rolling(30).std() * np.sqrt(252)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol.values, mode="lines", name="Rolling Vol"))
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------
# Tab 4 — Sectors
# ---------------------------------------------------------
with tab4:
    st.subheader("Sectors")

    if "fundamentals" not in st.session_state or "portfolio_weights" not in st.session_state:
        st.info("Run the Optimizer tab first to calculate portfolio weights.")
        st.stop()

    fundamentals = st.session_state["fundamentals"]
    weights = np.array(st.session_state["portfolio_weights"])

    if "Sector" not in fundamentals.columns:
        st.warning("Sector data not available in fundamentals.")
        st.stop()

    tickers = fundamentals.index.tolist()
    sector_series = fundamentals["Sector"]

    sector_weights = {}
    for i, t in enumerate(tickers):
        sector = sector_series.loc[t]
        sector_weights[sector] = sector_weights.get(sector, 0) + weights[i]

    sector_df = pd.DataFrame({
        "Sector": list(sector_weights.keys()),
        "Weight": list(sector_weights.values())
    }).sort_values("Weight", ascending=False)

    fig = go.Figure(data=[go.Pie(labels=sector_df["Sector"], values=sector_df["Weight"])])
    fig.update_layout(title="Sector Allocation")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(sector_df.reset_index(drop=True))

# ---------------------------------------------------------
# Tab 5 — Fundamentals
# ---------------------------------------------------------
with tab5:
    st.subheader("Fundamentals")

    if "fundamentals" not in st.session_state:
        st.info("Run Analysis first to load fundamentals.")
        st.stop()

    fundamentals = st.session_state["fundamentals"]
    st.dataframe(fundamentals)

# ---------------------------------------------------------
# Tab 6 — Weights
# ---------------------------------------------------------
with tab6:
    st.subheader("Weights")

    if "portfolio_weights" not in st.session_state or "tickers" not in st.session_state:
        st.info("Run the Optimizer tab first to calculate portfolio weights.")
        st.stop()

    tickers = st.session_state["tickers"]
    weights = list(st.session_state["portfolio_weights"])

    st.write("Adjust target weights (they will be normalized to sum to 1):")

    new_weights = []
    for t, w in zip(tickers, weights):
        val = st.slider(f"{t} weight", min_value=0.0, max_value=1.0, value=float(w), step=0.01)
        new_weights.append(val)

    new_weights = np.array(new_weights)
    if new_weights.sum() == 0:
        st.warning("All weights are zero; cannot normalize.")
    else:
        new_weights = new_weights / new_weights.sum()
        st.session_state["portfolio_weights"] = new_weights.tolist()

    weights_df = pd.DataFrame({"Ticker": tickers, "Weight": new_weights})
    st.dataframe(weights_df)

# ---------------------------------------------------------
# Tab 7 — AI Commentary
# ---------------------------------------------------------
with tab7:
    st.subheader("AI Commentary")

    if "portfolio_weights" not in st.session_state or "returns_df" not in st.session_state:
        st.info("Run the Optimizer tab first to calculate portfolio weights.")
        st.stop()

    # Simple rule-based commentary using metrics
    comments = []

    if not np.isnan(sharpe_ratio):
        if sharpe_ratio > 1.5:
            comments.append("The portfolio exhibits a strong risk-adjusted return profile (high Sharpe ratio).")
        elif sharpe_ratio > 1.0:
            comments.append("The portfolio has a reasonable risk-adjusted return profile.")
        else:
            comments.append("The portfolio's risk-adjusted performance is modest; consider improving diversification or tilting to higher-quality names.")

    if not np.isnan(max_drawdown):
        if max_drawdown < -0.3:
            comments.append("Historical drawdowns have been deep; risk management and position sizing are critical.")
        elif max_drawdown < -0.15:
            comments.append("Drawdowns are noticeable but within a typical equity risk range.")
        else:
            comments.append("Drawdowns have been relatively contained historically.")

    if not np.isnan(diversification_score):
        if diversification_score >= 7:
            comments.append("The portfolio appears well diversified across its holdings.")
        elif diversification_score >= 4:
            comments.append("Diversification is moderate; consider adding uncorrelated exposures.")
        else:
            comments.append("The portfolio seems concentrated; correlation risk may be elevated.")

    if not comments:
        st.write("Not enough data to generate commentary.")
    else:
        for c in comments:
            st.markdown(f"- {c}")

# ---------------------------------------------------------
# Tab 8 — Buy Analysis
# ---------------------------------------------------------
with tab8:
    st.subheader("Buy Analysis")

    required_keys = ["tickers", "fundamentals", "latest_prices", "portfolio_weights"]
    if not all(k in st.session_state for k in required_keys):
        st.info("Run the Optimizer tab first to calculate portfolio weights.")
        st.stop()

    tickers = st.session_state["tickers"]
    fundamentals = st.session_state["fundamentals"]
    latest_prices = st.session_state["latest_prices"]
    weights = np.array(st.session_state["portfolio_weights"])

    try:
        buy_results = run_buy_analysis(
            tickers=tickers,
            fundamentals=fundamentals,
            prices=latest_prices,
            weights=weights
        )

        st.write("**Buy Analysis Scores**")
        st.dataframe(buy_results)

        if {"Ticker", "Score"}.issubset(buy_results.columns):
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=buy_results["Ticker"],
                y=buy_results["Score"],
                name="Score"
            ))
            fig.update_layout(title="Buy Scores by Ticker", xaxis_title="Ticker", yaxis_title="Score")
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Buy analysis failed: {e}")

# ---------------------------------------------------------
# Tab 9 — Optimizer
# ---------------------------------------------------------
@st.cache_data(show_spinner=True)
def run_optimizer_cached(returns, cov):
    return run_optimizer(returns, cov)

with tab9:
    st.subheader("Optimizer")

    if "returns_df" not in st.session_state:
        st.info("Run Analysis first to load data.")
        st.stop()

    returns_df = st.session_state["returns_df"]

    cov_matrix = returns_df.cov()
    opt_results = run_optimizer_cached(returns_df, cov_matrix)

    st.session_state["model"] = opt_results
    st.session_state["portfolio_weights"] = opt_results["max_sharpe"]["weights"]
    st.session_state["returns_df"] = returns_df

    st.success("Optimization complete!")

    # Equal Weight Portfolio
    st.markdown("### Equal Weight Portfolio")
    ew = opt_results["equal_weight"]

    st.write(f"**Expected Return:** {ew['expected_return']:.2%}")
    st.write(f"**Volatility:** {ew['volatility']:.2%}")
    st.write(f"**Sharpe Ratio:** {ew['sharpe']:.2f}")

    ew_df = pd.DataFrame({
        "Ticker": opt_results["tickers"],
        "Weight": ew["weights"]
    })
    st.dataframe(ew_df, key="ew_df")

    # Maximum Sharpe Portfolio
    st.markdown("### Maximum Sharpe Portfolio")
    ms = opt_results["max_sharpe"]

    st.write(f"**Expected Return:** {ms['expected_return']:.2%}")
    st.write(f"**Volatility:** {ms['volatility']:.2%}")
    st.write(f"**Sharpe Ratio:** {ms['sharpe']:.2f}")

    ms_df = pd.DataFrame({
        "Ticker": opt_results["tickers"],
        "Weight": ms["weights"]
    })
    st.dataframe(ms_df, key="ms_df")

    # Correlation Heatmap
    st.markdown("### Correlation Heatmap")

    corr = returns_df.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale="RdBu",
        zmin=-1,
        zmax=1
    ))

    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
