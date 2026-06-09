import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor

st.set_option("client.showErrorDetails", True)
st.write("THIS IS THE FILE BEING EXECUTED")
st.cache_data.clear()
st.cache_resource.clear()

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
    right: 0;

    margin-top: 20px;

    background-color: white;
    padding: 10px 20px;

    font-size: 28px;
    font-weight: 900;
    color: #1E90FF;

    text-align: center;

    border-bottom: 1px solid #E5E5E5;
    box-shadow: 0 1px 4px rgba(0,0,0,0.10);

    z-index: 9999;
}

header[data-testid="stHeader"] {
    display: none !important;
}

div[data-testid="stAppViewContainer"] {
    background-color: transparent !important;
}

div[data-testid="stAppViewContainer"] .block-container {
    padding-top: 100px !important;
}

</style>

<div class="fixed-title">Portfolio Optimizer Dashboard</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Safe value cleaner
# ---------------------------------------------------------
def safe_val(x):
    if x in [None, "None", "nan", "NaN"]:
        return None
    try:
        if float(x) == 0:
            return None
    except:
        pass
    return x

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
# Load Price Data (robust loader)
# ---------------------------------------------------------
@st.cache_data
def load_price_data(tickers, start, end):
    try:
        raw = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False
        )

        if raw is None or raw.empty:
            return pd.DataFrame()

        # Multi-index (multi-ticker)
        if isinstance(raw.columns, pd.MultiIndex):
            if "Adj Close" in raw.columns.get_level_values(0):
                adj = raw["Adj Close"]
            elif "Adj Close" in raw.columns.get_level_values(1):
                adj = raw.xs("Adj Close", level=1, axis=1)
            elif "Close" in raw.columns.get_level_values(0):
                adj = raw["Close"]
            else:
                adj = raw.xs("Close", level=1, axis=1)

        # Single ticker
        else:
            if "Adj Close" in raw.columns:
                adj = raw[["Adj Close"]]
                adj.columns = [tickers[0]]
            else:
                adj = raw[["Close"]]
                adj.columns = [tickers[0]]

        if isinstance(adj, pd.Series):
            adj = adj.to_frame()

        adj = adj.dropna(axis=1, how="all")

        return adj

    except Exception as e:
        st.error(f"Price load failed: {e}")
        return pd.DataFrame()
# ---------------------------------------------------------
# STEP 1 — Load Prices + Validate
# ---------------------------------------------------------

prices = load_price_data(tickers, start_date, end_date)

if prices is None or prices.empty:
    st.error("Price data failed. Cannot continue.")
    st.stop()

# ---------------------------------------------------------
# STEP 1b — Compute Returns
# ---------------------------------------------------------

returns_df = prices.pct_change().dropna()

if returns_df is None or returns_df.empty:
    st.error("Return data unavailable after pct_change().")
    st.stop()

# ---------------------------------------------------------
# STEP 1c — Define valid_tickers
# ---------------------------------------------------------

valid_tickers = list(returns_df.columns)

if len(valid_tickers) == 0:
    st.error("No valid tickers after cleaning returns.")
    st.stop()

# ---------------------------------------------------------
# GLOBAL DEFAULT WEIGHTS (EQUAL WEIGHTS FAILSAFE)
# ---------------------------------------------------------

import numpy as np

global_weights = np.array([1 / len(valid_tickers)] * len(valid_tickers))

# ---------------------------------------------------------
# FUNDAMENTALS LOADER — FINAL FIXED VERSION
# ---------------------------------------------------------
import requests
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

FMP_API_KEY = "xxxxxxxxxxxxxxxxxxxx"

SECTOR_OVERRIDE = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "MU": "Technology", "PLTR": "Technology", "NOK": "Technology",
    "ARM": "Technology", "RDW": "Technology", "APP": "Communication Services",
    "GOOG": "Communication Services", "GOOGL": "Communication Services",
    "NFLX": "Communication Services", "AMZN": "Consumer Cyclical",
    "TSLA": "Consumer Cyclical"
}

def safe_float(x):
    try:
        if x in (None, "", "None", "NaN", "nan"):
            return np.nan
        x = float(x)
        return np.nan if np.isnan(x) or np.isinf(x) else x
    except:
        return np.nan

def load_single_fundamental(t):
    t = t.upper().strip()
    try:
        # PROFILE
        url_p = f"https://financialmodelingprep.com/api/v3/profile/{t}?apikey={FMP_API_KEY}"
        p = requests.get(url_p, timeout=6).json()
        p = p[0] if isinstance(p, list) and p else {}

        # KEY METRICS
        url_km = f"https://financialmodelingprep.com/api/v3/key-metrics/{t}?limit=1&apikey={FMP_API_KEY}"
        km = requests.get(url_km, timeout=6).json()
        km = km[0] if isinstance(km, list) and km else {}

        # RATIOS
        url_rt = f"https://financialmodelingprep.com/api/v3/ratios/{t}?limit=1&apikey={FMP_API_KEY}"
        rt = requests.get(url_rt, timeout=6).json()
        rt = rt[0] if isinstance(rt, list) and rt else {}

        sector = p.get("sector") or SECTOR_OVERRIDE.get(t, "Unknown")

        return t, {
            "PE": safe_float(km.get("peRatio")),
            "PB": safe_float(km.get("pbRatio")),
            "EPS": safe_float(km.get("eps")),
            "ROE": safe_float(rt.get("returnOnEquity")),
            "DividendYield": safe_float(rt.get("dividendYield")),
            "DebtToEquity": safe_float(rt.get("debtEquityRatio")),
            "Beta": safe_float(p.get("beta")),
            "MarketCap": safe_float(p.get("mktCap")),
            "Sector": sector,
        }

    except:
        return t, {
            "PE": np.nan, "PB": np.nan, "EPS": np.nan, "ROE": np.nan,
            "DividendYield": np.nan, "DebtToEquity": np.nan,
            "Beta": np.nan, "MarketCap": np.nan,
            "Sector": SECTOR_OVERRIDE.get(t, "Unknown"),
        }

@st.cache_data
def load_fundamentals_auto(tickers):
    fundamentals = {}
    tickers = [t.upper().strip() for t in tickers]

    with ThreadPoolExecutor(max_workers=10) as ex:
        for t, data in ex.map(load_single_fundamental, tickers):
            fundamentals[t] = data

    df = pd.DataFrame.from_dict(fundamentals, orient="index")
    return df

fundamentals_df = load_fundamentals_auto(valid_tickers)

if fundamentals_df.isna().all().all():
    st.error("No fundamentals returned. Check your FMP API key.")

# HARD GUARD — prevents fake commentary & zero scores
if fundamentals_df.isna().all().all():
    st.error("No fundamentals returned. Check your FMP API key.")

# -----------------------------------------
# OPTION B — Replace NaN with sector averages
# -----------------------------------------

numeric_cols = [
    "PE", "PB", "EPS", "ROE",
    "DividendYield", "DebtToEquity",
    "Beta", "MarketCap"
]

# Convert zeros back to NaN (because zeros came from missing data)
for col in numeric_cols:
    fundamentals_df[col] = fundamentals_df[col].replace(0, np.nan)

# Fill NaN with sector averages
for col in numeric_cols:
    fundamentals_df[col] = fundamentals_df.groupby("Sector")[col].transform(
        lambda x: x.fillna(x.mean())
    )

if "SPY" in fundamentals_df.index:
    fundamentals_df = fundamentals_df.drop("SPY")

if fundamentals_df is None or fundamentals_df.empty:
    st.error("FATAL: fundamentals_df is EMPTY — fundamentals loader returned no data.")
    st.write(fundamentals_df)
    st.stop()

st.write("DEBUG FUNDAMENTALS:", fundamentals_df)

# ---------------------------------------------------------
# STEP 3 — GLOBAL PORTFOLIO METRICS
# ---------------------------------------------------------

try:
    # Portfolio returns
    portfolio_returns = returns_df.dot(global_weights)

    # Max Drawdown
    max_drawdown = (portfolio_returns.cummax() - portfolio_returns).max()

    # Annualized return
    annual_return = portfolio_returns.mean() * 252

    # Annualized volatility
    annual_volatility = portfolio_returns.std() * (252 ** 0.5)

    # Sharpe ratio
    sharpe_ratio = (
        annual_return / annual_volatility
        if annual_volatility not in [0, None] else 0
    )

except Exception as e:
    st.error(f"Portfolio metrics failed: {e}")
    st.stop()

    # ---------------------------------------------------------
    # UI METRIC
    # ---------------------------------------------------------
    st.metric("Beta vs SPY", f"{portfolio_beta:.2f}")

except Exception as e:
    st.error(f"Portfolio metrics failed: {e}")
    st.stop()
# ---------------------------------------------------------
# STEP 4 — GLOBAL COMMENTARY INPUTS (LIST-BASED, FINAL)
# ---------------------------------------------------------

# Pull raw weights (may be None, dict, list, etc.)
raw_weights = st.session_state.get("weights", None)

# CASE 1 — Missing or None → reset
if raw_weights is None:
    stored_weights = [1/len(valid_tickers)] * len(valid_tickers)

# CASE 2 — Dict → convert to list in ticker order
elif isinstance(raw_weights, dict):
    stored_weights = [raw_weights.get(t, 1/len(valid_tickers)) for t in valid_tickers]

# CASE 3 — List → use it
elif isinstance(raw_weights, list):
    stored_weights = raw_weights.copy()

# CASE 4 — Anything else → reset
else:
    stored_weights = [1/len(valid_tickers)] * len(valid_tickers)

# Fix wrong length
if len(stored_weights) != len(valid_tickers):
    stored_weights = [1/len(valid_tickers)] * len(valid_tickers)

# Convert all values to float (fixes strings)
clean_weights = []
for w in stored_weights:
    try:
        clean_weights.append(float(w))
    except:
        clean_weights.append(1/len(valid_tickers))

stored_weights = clean_weights

# Normalize
total = sum(stored_weights)
if total == 0:
    stored_weights = [1/len(valid_tickers)] * len(valid_tickers)
else:
    stored_weights = [w/total for w in stored_weights]

# Save back
st.session_state["weights"] = stored_weights

# Build weights series
weights_series = pd.Series(stored_weights, index=valid_tickers)

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
# GLOBAL PORTFOLIO METRICS (MUST RUN BEFORE ANY TABS)
# ---------------------------------------------------------

try:
    # Compute portfolio returns
    portfolio_returns = returns_df.dot(global_weights)

    # Annualized return
    annual_return = portfolio_returns.mean() * 252

    # Annualized volatility
    annual_volatility = portfolio_returns.std() * (252 ** 0.5)

    # Sharpe ratio
    sharpe_ratio = (
        annual_return / annual_volatility
        if annual_volatility not in [0, None] else 0
    )

    # Portfolio beta vs SPY (if SPY exists)
    if "SPY" in returns_df.columns:
        spy_returns = returns_df["SPY"]
        covariance = portfolio_returns.cov(spy_returns)
        market_variance = spy_returns.var()
        portfolio_beta = covariance / market_variance if market_variance else None
    else:
        portfolio_beta = None

    # ---------------------------------------------------------
    # FINAL BETA GUARD CLAUSE (MUST BE LAST)
    # ---------------------------------------------------------
    if portfolio_beta is None:
        portfolio_beta = 0.0
    else:
        try:
            portfolio_beta = float(portfolio_beta)
        except Exception:
            portfolio_beta = 0.0

except Exception as e:
    st.error(f"Portfolio metrics failed: {e}")
    st.stop()

# ---------------------------------------------------------
# TAB 1 — OVERVIEW (FINAL CLEAN VERSION)
# ---------------------------------------------------------
with tab1:
    st.header("Portfolio Overview")

    # --- FIRST ROW OF METRICS ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Annual Return", f"{annual_return:.2%}")
    with col2:
        st.metric("Volatility", f"{annual_volatility:.2%}")
    with col3:
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

    # --- SECOND ROW OF METRICS ---
    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("Max Drawdown", f"{max_drawdown:.2%}")
    with col5:
        st.metric("Beta vs SPY", f"{portfolio_beta:.2f}")
    with col6:
        st.metric("Number of Holdings", len(valid_tickers))

    # ---------------------------------------------------------
    # PRICE CHART
    # ---------------------------------------------------------
    st.subheader("Price History")

    fig = go.Figure()
    for t in valid_tickers:
        fig.add_trace(go.Scatter(
            x=prices.index,
            y=prices[t],
            mode="lines",
            name=t
        ))

    fig.update_layout(
        height=400,
        title="Price History",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3)
    )

    st.plotly_chart(fig, use_container_width=True, key="plot_1_price_history")

# ---------------------------------------------------------
# TAB 2 — PERFORMANCE (ROBUST VERSION)
# ---------------------------------------------------------
with tab2:
    st.header("Performance Metrics")

    # ---------------------------------------------------------
    # GUARD CLAUSES
    # ---------------------------------------------------------
    if returns_df is None or returns_df.empty:
        st.warning("No return data available for the selected tickers/date range.")
        st.stop()

    # Drop columns that are entirely NaN
    returns_df = returns_df.dropna(how="all", axis=1)

    if returns_df.empty:
        st.warning("All selected tickers have missing returns in this period.")
        st.stop()

    # ---------------------------------------------------------
    # PORTFOLIO RETURNS (EQUAL-WEIGHT, ALWAYS VALID)
    # ---------------------------------------------------------
    portfolio_returns = returns_df.mean(axis=1)
    ret = pd.Series(portfolio_returns, name="Portfolio Return").dropna()

    if ret.empty:
        st.warning("No valid daily returns to analyze in this period.")
        st.stop()

    # ---------------------------------------------------------
    # CUMULATIVE RETURN
    # ---------------------------------------------------------
    cum_ret = (1 + ret).cumprod()

    # ---------------------------------------------------------
    # ROLLING METRICS
    # ---------------------------------------------------------
    rolling_vol = ret.rolling(30).std() * np.sqrt(252)
    rolling_sharpe = (ret.rolling(30).mean() * 252) / rolling_vol

    # ---------------------------------------------------------
    # DRAWDOWN
    # ---------------------------------------------------------
    cum_max = cum_ret.cummax()
    dd = (cum_ret - cum_max) / cum_max

    # ---------------------------------------------------------
    # PERFORMANCE STATISTICS
    # ---------------------------------------------------------
    mu = ret.mean() * 252
    vol = ret.std() * np.sqrt(252)
    sharpe_local = mu / vol if vol > 0 else 0
    neg_vol = ret[ret < 0].std() * np.sqrt(252)
    sortino = (mu / neg_vol) if neg_vol and neg_vol > 0 else 0
    calmar = mu / abs(dd.min()) if dd.min() != 0 else 0
    max_dd = dd.min()

    # ---------------------------------------------------------
    # DISPLAY METRICS
    # ---------------------------------------------------------
    colA, colB, colC, colD, colE, colF = st.columns(6)
    colA.metric("Expected Return", f"{mu:.2%}")
    colB.metric("Volatility", f"{vol:.2%}")
    colC.metric("Sharpe Ratio", f"{sharpe_local:.2f}")
    colD.metric("Sortino Ratio", f"{sortino:.2f}")
    colE.metric("Calmar Ratio", f"{calmar:.2f}")
    colF.metric("Max Drawdown", f"{max_dd:.2%}")

    # ---------------------------------------------------------
    # CHARTS
    # ---------------------------------------------------------
    st.markdown("### Cumulative Return")
    st.line_chart(cum_ret)

    st.markdown("### Rolling Volatility (30-day)")
    st.line_chart(rolling_vol)

    st.markdown("### Rolling Sharpe Ratio (30-day)")
    st.line_chart(rolling_sharpe)

    st.markdown("### Drawdown")
    st.area_chart(dd)

    # ---------------------------------------------------------
    # Distribution of Daily Returns (FIXED)
    # ---------------------------------------------------------
    st.markdown("### Distribution of Daily Returns")
    
    hist_data = ret.dropna()
    
    # FIX: Prevent empty histogram (the root cause of your issue)
    if hist_data.empty:
        st.info("No valid daily returns available to plot in this date range.")
    else:
        fig, ax = plt.subplots()
        ax.hist(hist_data, bins=40, alpha=0.7)
        ax.set_title("Histogram of Daily Returns")
        st.pyplot(fig)
    
# ---------------------------------------------------------
# TAB 3 — RISK & DRAWDOWN ANALYSIS (FINAL VERSION)
# ---------------------------------------------------------
with tab3:
    st.header("Risk & Drawdown Analysis")

    # ---------------------------------------------------------
    # SAFETY CHECKS
    # ---------------------------------------------------------
    if returns_df.empty:
        st.error("Return data unavailable. Check price loader.")
        st.stop()

    # ---------------------------------------------------------
    # PORTFOLIO RETURNS (EQUAL-WEIGHT, ALWAYS VALID)
    # ---------------------------------------------------------
    portfolio_returns = returns_df.mean(axis=1)
    ret = pd.Series(portfolio_returns, name="Portfolio Return")

    # ---------------------------------------------------------
    # DRAWDOWN
    # ---------------------------------------------------------
    cum_ret = (1 + ret).cumprod()
    running_max = cum_ret.cummax()
    drawdown = (cum_ret - running_max) / running_max
    max_dd = drawdown.min()

    # ---------------------------------------------------------
    # ROLLING VOLATILITY
    # ---------------------------------------------------------
    rolling_vol = ret.rolling(30).std() * np.sqrt(252)

    # ---------------------------------------------------------
    # BETA VS SPY (FROM STEP 4)
    # ---------------------------------------------------------
    beta_value = portfolio_beta if not np.isnan(portfolio_beta) else 0.0

    # ---------------------------------------------------------
    # VAR & CVAR
    # ---------------------------------------------------------
    var_95 = np.percentile(ret.dropna(), 5)
    cvar_95 = ret[ret <= var_95].mean() if len(ret[ret <= var_95]) > 0 else 0

    # ---------------------------------------------------------
    # DISPLAY METRICS
    # ---------------------------------------------------------
    colA, colB, colC, colD = st.columns(4)
    colA.metric("Max Drawdown", f"{max_dd:.2%}")
    colB.metric("Rolling Vol (30d)", f"{rolling_vol.iloc[-1]:.2%}")
    colC.metric("Beta vs SPY", f"{beta_value:.2f}")
    colD.metric("CVaR (95%)", f"{cvar_95:.2%}")

    # ---------------------------------------------------------
    # CHARTS
    # ---------------------------------------------------------
    st.markdown("### Drawdown")
    st.area_chart(drawdown)

    st.markdown("### Rolling Volatility (30-day)")
    st.line_chart(rolling_vol)

    st.markdown("### Distribution of Daily Returns (for VaR)")
    fig, ax = plt.subplots()
    ax.hist(ret.dropna(), bins=40, alpha=0.7)
    ax.axvline(var_95, color="red", linestyle="--", label=f"VaR 95%: {var_95:.2%}")
    ax.set_title("Return Distribution with VaR")
    ax.legend()
    st.pyplot(fig)

    # ---------------------------------------------------------
    # FULL DISTRIBUTION OF DAILY RETURNS (ALL TICKERS)
    # ---------------------------------------------------------
    st.markdown("### Distribution of Daily Returns (All Tickers)")
    
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    
    for col in returns_df.columns:
        ax3.hist(
            returns_df[col].dropna(),
            bins=40,
            alpha=0.4,
            label=col
        )
    
    ax3.set_title("Distribution of Daily Returns (All Tickers)")
    ax3.set_xlabel("Daily Return")
    ax3.set_ylabel("Frequency")
    ax3.legend()
    
    st.pyplot(fig3)

    # ---------------------------------------------------------
    # TRUE RISK CONTRIBUTION (MCTR-BASED)
    # ---------------------------------------------------------
    st.markdown("### Risk Contribution Breakdown")

    try:
        if "weights" not in st.session_state:
            st.info("Weights not set yet. Go to the Optimizer or Weights tab to set weights.")
        else:
            weights = np.array(st.session_state.weights, dtype=float)

            # Align weights with valid tickers
            if len(weights) != len(valid_tickers):
                st.warning("Weights length does not match number of valid tickers. Using equal weights instead.")
                weights = np.array([1/len(valid_tickers)] * len(valid_tickers))

            cov_matrix = returns_df[valid_tickers].cov().values

            # Portfolio volatility
            portfolio_volatility = np.sqrt(weights.T @ cov_matrix @ weights)

            if portfolio_volatility <= 0 or np.isnan(portfolio_volatility):
                st.warning("Portfolio volatility is zero or NaN. Cannot compute risk contribution.")
            else:
                # Marginal Contribution to Risk
                mctr = (cov_matrix @ weights) / portfolio_volatility

                # Risk Contribution
                risk_contribution = weights * mctr

                # Normalize to 100%
                risk_contribution = risk_contribution / risk_contribution.sum()

                # Display Pie Chart
                fig2, ax2 = plt.subplots()
                ax2.pie(
                    risk_contribution,
                    labels=valid_tickers,
                    autopct="%1.1f%%",
                    startangle=90
                )
                ax2.axis("equal")
                st.pyplot(fig2)

                # Display Table
                rc_df = pd.DataFrame({
                    "Ticker": valid_tickers,
                    "Risk Contribution %": (risk_contribution * 100).round(2)
                })
                st.dataframe(rc_df, use_container_width=True)

    except Exception as e:
        st.error(f"Risk Contribution failed: {e}")

# ---------------------------------------------------------
# TAB 4 — SECTOR EXPOSURE (FINAL FIXED VERSION)
# ---------------------------------------------------------
# ---------------------------------------------------------
# TAB 4 — SECTOR EXPOSURE (FINAL FIXED VERSION)
# ---------------------------------------------------------
with tab4:
    st.subheader("Sector Exposure")

    # ---------------------------------------------------------
    # SAFETY CHECKS
    # ---------------------------------------------------------
    if fundamentals_df.empty:
        st.warning("Sector data unavailable. Run analysis first.")
        st.stop()

    # Reindex fundamentals to match valid tickers
    fdf = fundamentals_df.reindex(valid_tickers).copy()

    # Ensure Sector column exists
    if "Sector" not in fdf.columns:
        fdf["Sector"] = "Unknown"

    # Clean sector values
    fdf["Sector"] = (
        fdf["Sector"]
        .fillna("Unknown")
        .replace("", "Unknown")
        .replace("None", "Unknown")
    )

    # ---------------------------------------------------------
    # FIX: Apply sector overrides (ensures no Unknown for known tickers)
    # ---------------------------------------------------------
    for t in fdf.index:
        if fdf.loc[t, "Sector"] == "Unknown" and t in SECTOR_OVERRIDE:
            fdf.loc[t, "Sector"] = SECTOR_OVERRIDE[t]

    # ---------------------------------------------------------
    # LOAD WEIGHTS
    # ---------------------------------------------------------
    if "weights" in st.session_state:
        w = st.session_state.weights
    else:
        w = np.array([1 / len(valid_tickers)] * len(valid_tickers))

    # FIX: Ensure weight length matches ticker count
    if len(w) != len(valid_tickers):
        w = np.array([1 / len(valid_tickers)] * len(valid_tickers))

    w_series = pd.Series(w, index=valid_tickers)

    # ---------------------------------------------------------
    # GROUP BY SECTOR
    # ---------------------------------------------------------
    sector_weights = (
        w_series.groupby(fdf["Sector"])
        .sum()
        .sort_values(ascending=False)
    )

    # ---------------------------------------------------------
    # DISPLAY TABLE
    # ---------------------------------------------------------
    st.markdown("### Sector Allocation Breakdown")
    st.dataframe(
        sector_weights.to_frame("Weight").style.format({"Weight": "{:.2%}"})
    )

    # ---------------------------------------------------------
    # DISPLAY CHART
    # ---------------------------------------------------------
    st.markdown("### Sector Chart")
    fig = go.Figure(
        go.Bar(
            x=sector_weights.index,
            y=sector_weights.values,
            marker_color="steelblue"
        )
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True, key="plot_714")

# ---------------------------------------------------------
# TAB 5 — FUNDAMENTALS (FINAL FIXED VERSION)
# ---------------------------------------------------------
with tab5:
    st.header("Fundamentals")

    # ---------------------------------------------------------
    # SAFETY CHECK
    # ---------------------------------------------------------
    if fundamentals_df.empty or fundamentals_df.isna().all().all():
        st.warning("No fundamentals available. Check API key or data coverage.")
        st.stop()

    # ---------------------------------------------------------
    # CLEAN FUNDAMENTALS TABLE
    # ---------------------------------------------------------
    fdf = fundamentals_df.copy()
    fdf = fdf.reindex(valid_tickers)

    # Normalize missing values
    fdf = fdf.replace({None: np.nan, "None": np.nan, "": np.nan})

    # Convert numeric columns to float
    numeric_cols = ["PE", "PB", "EPS", "ROE", "DividendYield", "DebtToEquity", "Beta", "MarketCap"]
    for col in numeric_cols:
        if col in fdf.columns:
            fdf[col] = pd.to_numeric(fdf[col], errors="coerce")

    fundamentals_display = fdf.drop(columns=["Sector"], errors="ignore")

    st.subheader("Raw Fundamentals")
    st.dataframe(
        fundamentals_display.style.format({
            "PE": "{:.2f}",
            "PB": "{:.2f}",
            "EPS": "{:.2f}",
            "ROE": "{:.2%}",
            "DividendYield": "{:.2%}",
            "DebtToEquity": "{:.2f}",
            "Beta": "{:.2f}",
            "MarketCap": "{:,.0f}"
        }),
        use_container_width=True
    )

    # ---------------------------------------------------------
    # FUNDAMENTALS RANKING
    # ---------------------------------------------------------
    st.subheader("Fundamentals Ranking")

    def score_fundamentals(row):
        score = 0

        # ROE (scaled)
        if pd.notna(row.get("ROE")):
            score += row["ROE"] * 10

        # EPS
        if pd.notna(row.get("EPS")):
            score += row["EPS"]

        # PE (lower is better)
        if pd.notna(row.get("PE")):
            score += max(0, 50 - row["PE"])

        # PB (lower is better)
        if pd.notna(row.get("PB")):
            score += max(0, 20 - row["PB"])

        # Dividend Yield
        if pd.notna(row.get("DividendYield")):
            score += row["DividendYield"] * 100

        return score

    fdf["score"] = fdf.apply(score_fundamentals, axis=1)
    ranked_df = fdf.sort_values("score", ascending=False)

    st.dataframe(ranked_df[["score"]], use_container_width=True)

    # ---------------------------------------------------------
    # AI FUNDAMENTALS COMMENTARY
    # ---------------------------------------------------------
    st.subheader("AI Fundamentals Commentary")

    def generate_fundamentals_commentary(ranked_df):
        lines = []
        tickers_sorted = ranked_df.index.tolist()
        if not tickers_sorted:
            return "No fundamentals available."

        best = tickers_sorted[0]
        worst = tickers_sorted[-1]

        # Best
        best_row = ranked_df.loc[best]
        best_reasons = []
        if pd.notna(best_row.get("ROE")) and best_row["ROE"] > 0:
            best_reasons.append("strong return on equity")
        if pd.notna(best_row.get("EPS")) and best_row["EPS"] > 0:
            best_reasons.append("solid earnings power")
        if pd.notna(best_row.get("PE")) and best_row["PE"] < 25:
            best_reasons.append("reasonable valuation")
        if pd.notna(best_row.get("DividendYield")) and best_row["DividendYield"] > 0:
            best_reasons.append("added income from dividends")

        best_reason_text = ", ".join(best_reasons) if best_reasons else "overall stronger fundamentals"
        lines.append(f"**{best}** ranks as the strongest fundamental name in the group, supported by {best_reason_text}.")

        # Middle
        if len(tickers_sorted) > 2:
            for t in tickers_sorted[1:-1]:
                row = ranked_df.loc[t]
                mid_reasons = []
                if pd.notna(row.get("ROE")) and row["ROE"] > 0:
                    mid_reasons.append("healthy ROE")
                if pd.notna(row.get("EPS")) and row["EPS"] > 0:
                    mid_reasons.append("stable earnings")
                if pd.notna(row.get("PE")) and row["PE"] < 40:
                    mid_reasons.append("fair valuation")

                reason_text = ", ".join(mid_reasons) if mid_reasons else "balanced fundamentals"
                lines.append(f"**{t}** shows {reason_text}.")

        # Worst
        worst_row = ranked_df.loc[worst]
        worst_reasons = []
        if pd.notna(worst_row.get("ROE")) and worst_row["ROE"] < 0.05:
            worst_reasons.append("weak ROE")
        if pd.notna(worst_row.get("PE")) and worst_row["PE"] > 50:
            worst_reasons.append("elevated valuation")
        if pd.notna(worst_row.get("PB")) and worst_row["PB"] > 10:
            worst_reasons.append("rich price-to-book ratio")

        worst_reason_text = ", ".join(worst_reasons) if worst_reasons else "weaker fundamentals overall"
        lines.append(f"**{worst}** ranks lowest, driven by {worst_reason_text}.")

        return "\n\n".join(lines)

    st.markdown(generate_fundamentals_commentary(ranked_df))

    # ---------------------------------------------------------
    # SIMPLE COMMENTARY LIST
    # ---------------------------------------------------------
    st.subheader("Commentary")
    commentary = [
        f"- **{ticker}**: score {row['score']:.1f}"
        for ticker, row in ranked_df.iterrows()
    ]
    st.markdown("\n".join(commentary))

# ---------------------------------------------------------
# TAB 6 — PORTFOLIO WEIGHTS (FULLY FIXED VERSION)
# ---------------------------------------------------------
with tab6:
    st.header("Portfolio Weights")

    if len(valid_tickers) == 0:
        st.warning("No valid tickers available to assign weights.")
        st.stop()

    # ---------------------------------------------------------
    # INITIALIZE SESSION STATE WEIGHTS (only once)
    # ---------------------------------------------------------
    if "weights" not in st.session_state or len(st.session_state.weights) != len(valid_tickers):
        st.session_state.weights = np.array(
            [1 / len(valid_tickers)] * len(valid_tickers),
            dtype=float
        )

    st.subheader("Adjust Weights")

    # ---------------------------------------------------------
    # SLIDERS — USER-ADJUSTABLE WEIGHTS
    # ---------------------------------------------------------
    new_weights = []
    for i, t in enumerate(valid_tickers):
        w_val = st.slider(
            f"{t} Weight",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.weights[i]),
            key=f"weight_slider_{t}"
        )
        new_weights.append(w_val)

    # ---------------------------------------------------------
    # NORMALIZE WEIGHTS (sum to 1)
    # ---------------------------------------------------------
    new_weights = np.array(new_weights, dtype=float)
    total = new_weights.sum()

    if total > 0:
        new_weights = new_weights / total
    else:
        new_weights = np.array([1 / len(valid_tickers)] * len(valid_tickers))

    # SAVE BACK TO SESSION STATE
    st.session_state.weights = new_weights

    # ---------------------------------------------------------
    # DISPLAY WEIGHTS TABLE
    # ---------------------------------------------------------
    st.subheader("Final Normalized Weights")
    weights_df = pd.DataFrame({
        "Ticker": valid_tickers,
        "Weight": st.session_state.weights
    })
    st.dataframe(weights_df, use_container_width=True)

# ---------------------------------------------------------
# TAB 7 — AI PORTFOLIO COMMENTARY (FINAL VERSION)
# ---------------------------------------------------------
with tab7:
    st.header("AI Portfolio Commentary")

    # ---------------------------------------------------------
    # SAFETY CHECKS
    # ---------------------------------------------------------
    if fundamentals_df.empty or fundamentals_df.isna().all().all():
        st.warning("Fundamentals not available. Run analysis first.")
        st.stop()

    if "weights" not in st.session_state:
        st.warning("Weights not set. Adjust weights in Tab 6.")
        st.stop()

    # ---------------------------------------------------------
    # PREPARE COMMENTARY DATAFRAME
    # ---------------------------------------------------------
    commentary_df = fundamentals_df.copy()
    commentary_df = commentary_df.reindex(valid_tickers)

    # Add weights
    weights = st.session_state.weights
    if len(weights) != len(valid_tickers):
        weights = np.array([1 / len(valid_tickers)] * len(valid_tickers))

    commentary_df["Weight"] = weights

    # ---------------------------------------------------------
    # FUNDAMENTALS SCORE (same logic as Tab 5)
    # ---------------------------------------------------------
    def score_fundamentals(row):
        score = 0

        # ROE (scaled)
        if pd.notna(row.get("ROE")):
            score += row["ROE"] * 10

        # EPS
        if pd.notna(row.get("EPS")):
            score += row["EPS"]

        # PE (lower is better)
        if pd.notna(row.get("PE")):
            score += max(0, 50 - row["PE"])

        # PB (lower is better)
        if pd.notna(row.get("PB")):
            score += max(0, 20 - row["PB"])

        # Dividend Yield
        if pd.notna(row.get("DividendYield")):
            score += row["DividendYield"] * 100

        return score

    commentary_df["score"] = commentary_df.apply(score_fundamentals, axis=1)

    # Remove SPY if present
    if "SPY" in commentary_df.index:
        commentary_df = commentary_df.drop("SPY")

    # Sort by score
    commentary_df = commentary_df.sort_values("score", ascending=False)

    # ---------------------------------------------------------
    # DISPLAY DATA USED FOR COMMENTARY
    # ---------------------------------------------------------
    st.subheader("Data Used for Commentary")
    st.dataframe(
        commentary_df[["score", "Weight", "Sector"]],
        use_container_width=True
    )

    # ---------------------------------------------------------
    # EXTRACT TOP & BOTTOM PICKS
    # ---------------------------------------------------------
    top_pick = commentary_df.index[0]
    bottom_pick = commentary_df.index[-1]

    top_score = commentary_df.iloc[0]["score"]
    bottom_score = commentary_df.iloc[-1]["score"]

    top_weight = commentary_df.iloc[0]["Weight"]
    bottom_weight = commentary_df.iloc[-1]["Weight"]

    # ---------------------------------------------------------
    # PORTFOLIO METRICS (from global model)
    # ---------------------------------------------------------
    ar = annual_return
    vol = annual_volatility
    shrp = sharpe_ratio
    beta = portfolio_beta
    mdd = max_drawdown

    # ---------------------------------------------------------
    # AI COMMENTARY TEXT
    # ---------------------------------------------------------
    commentary_text = f"""
    ### Portfolio Overview
    Your portfolio currently delivers an **annualized return of {ar:.2%}**, with volatility at **{vol:.2%}**  
    and a Sharpe ratio of **{shrp:.2f}**. The portfolio beta of **{beta:.2f}** indicates its sensitivity  
    to market movements, while the maximum drawdown of **{mdd:.2%}** reflects downside risk.
    
    ### Top Fundamental Pick
    **{top_pick}** leads the portfolio with a fundamentals score of **{top_score:.1f}**  
    and a portfolio weight of **{top_weight:.2%}**. This suggests strong underlying quality  
    and a meaningful contribution to long‑term performance.
    
    ### Weakest Fundamental Pick
    **{bottom_pick}** ranks lowest with a score of **{bottom_score:.1f}**  
    and a weight of **{bottom_weight:.2%}**. This may warrant monitoring or rebalancing  
    depending on your risk tolerance and investment horizon.
    
    ### Sector Positioning
    Your sector exposure reflects the combined influence of fundamentals strength  
    and your selected weight allocations. This helps maintain diversification  
    while emphasizing higher‑quality names.
    
    ### Final Thoughts
    Overall, the portfolio demonstrates balanced exposure with clear leaders and laggards.  
    Increasing exposure to high‑scoring names and trimming weaker positions  
    may improve risk‑adjusted performance going forward.
    """

    st.subheader("AI‑Generated Commentary")
    st.markdown(commentary_text)
# ---------------------------------------------------------
# TAB 8 — BUY ANALYSIS (FINAL FIXED VERSION)
# ---------------------------------------------------------
with tab8:
    st.header("AI Buy / Hold / Sell Analysis")

    # ---------------------------------------------------------
    # SAFETY CHECKS
    # ---------------------------------------------------------
    if fundamentals_df.empty or fundamentals_df.isna().all().all():
        st.warning("Fundamentals not available. Run analysis first.")
        st.stop()

    if "weights" not in st.session_state:
        st.warning("Weights not set. Adjust weights in Tab 6.")
        st.stop()

    # ---------------------------------------------------------
    # ALIGN FUNDAMENTALS WITH VALID TICKERS
    # ---------------------------------------------------------
    analysis_df = fundamentals_df.reindex(valid_tickers).copy()

    # Add weights with length check
    w = st.session_state.weights
    if len(w) != len(valid_tickers):
        w = np.array([1 / len(valid_tickers)] * len(valid_tickers))

    analysis_df["Weight"] = w

    # Remove SPY if present
    if "SPY" in analysis_df.index:
        analysis_df = analysis_df.drop("SPY")

    # ---------------------------------------------------------
    # ENSURE REQUIRED COLUMNS EXIST
    # ---------------------------------------------------------
    for col in ["PE", "PB", "DividendYield", "Beta"]:
        if col not in analysis_df.columns:
            analysis_df[col] = np.nan

    # ---------------------------------------------------------
    # MOMENTUM (63‑DAY RETURN) & RISK (ANNUALIZED VOL)
    # ---------------------------------------------------------
    if "returns_df" in globals() and not returns_df.empty:

        # Momentum: sum of last 63 daily returns
        momentum_series = returns_df.pct_change().tail(63).sum()
        analysis_df["Momentum"] = momentum_series.reindex(analysis_df.index)

        # Risk: annualized volatility
        common = [c for c in analysis_df.index if c in returns_df.columns]
        if common:
            risk_series = returns_df[common].pct_change().std() * np.sqrt(252)
            analysis_df.loc[common, "Risk"] = risk_series.reindex(common)
        else:
            analysis_df["Risk"] = np.nan

    else:
        analysis_df["Momentum"] = np.nan
        analysis_df["Risk"] = np.nan

    # ---------------------------------------------------------
    # CLEAN NUMERIC COLUMNS
    # ---------------------------------------------------------
    numeric_cols = ["PE", "PB", "DividendYield", "Beta", "Momentum", "Risk"]
    for col in numeric_cols:
        analysis_df[col] = pd.to_numeric(analysis_df[col], errors="coerce").fillna(0)

    # ---------------------------------------------------------
    # GENERATE SIGNALS
    # ---------------------------------------------------------
    signals = []
    for t, row in analysis_df.iterrows():
        score = 0
        conviction = 0

        # PE filter
        if 0 < row["PE"] < 40:
            score += 1
            conviction += 20

        # PB filter
        if 0 < row["PB"] < 8:
            score += 1
            conviction += 15

        # Dividend Yield
        if row["DividendYield"] > 0.005:
            score += 1
            conviction += 15

        # Beta
        if row["Beta"] < 1.3:
            score += 1
            conviction += 20

        # Momentum
        if row["Momentum"] > 0:
            score += 1
            conviction += 30

        rating = "Buy" if score >= 4 else "Hold" if score >= 2 else "Sell"
        conviction = min(100, max(0, conviction))

        signals.append({
            "Ticker": t,
            "PE": row["PE"],
            "PB": row["PB"],
            "DividendYield": row["DividendYield"],
            "Beta": row["Beta"],
            "Momentum": row["Momentum"],
            "Risk": row["Risk"],
            "Score": score,
            "Conviction": conviction,
            "Rating": rating
        })

    signals_df = pd.DataFrame(signals).sort_values("Conviction", ascending=False)
    signals_df["RatingColored"] = signals_df["Rating"].map({
        "Buy": "🟢 Buy",
        "Hold": "🟡 Hold",
        "Sell": "🔴 Sell"
    })

    # ---------------------------------------------------------
    # DISPLAY SIGNAL TABLE
    # ---------------------------------------------------------
    st.subheader("AI Buy / Hold / Sell Signals")
    st.dataframe(
        signals_df[
            ["Ticker", "PE", "PB", "DividendYield", "Beta", "Momentum", "Risk",
             "Score", "Conviction", "RatingColored"]
        ],
        use_container_width=True
    )

    # ---------------------------------------------------------
    # SIGNAL SUMMARY
    # ---------------------------------------------------------
    st.subheader("Signal Summary")
    buys = signals_df[signals_df["Rating"] == "Buy"]["Ticker"].tolist()
    holds = signals_df[signals_df["Rating"] == "Hold"]["Ticker"].tolist()
    sells = signals_df[signals_df["Rating"] == "Sell"]["Ticker"].tolist()

    if buys:
        st.success(f"**Buy signals:** {', '.join(buys)}")
    if holds:
        st.info(f"**Hold signals:** {', '.join(holds)}")
    if sells:
        st.error(f"**Sell signals:** {', '.join(sells)}")

    # ---------------------------------------------------------
    # PORTFOLIO‑LEVEL SIGNAL
    # ---------------------------------------------------------
    st.subheader("AI Portfolio‑Level Signal")
    buy_count = len(buys)
    sell_count = len(sells)

    if buy_count > sell_count:
        st.success("**AI Portfolio Signal: BUY** — Broad fundamental strength detected.")
    elif sell_count >= buy_count + 2:
        st.error("**AI Portfolio Signal: SELL** — Broad fundamental weakness detected.")
    else:
        st.warning("**AI Portfolio Signal: HOLD** — Mixed signals across the portfolio.")

    # ---------------------------------------------------------
    # RADAR CHART
    # ---------------------------------------------------------
    st.subheader("Fundamentals Radar Chart")
    radar_cols = ["PE", "PB", "DividendYield", "Momentum", "Risk"]

    fig = go.Figure()
    for _, row in signals_df.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row[c] for c in radar_cols],
            theta=radar_cols,
            fill='toself',
            name=row["Ticker"]
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # STRENGTHS & WEAKNESSES
    # ---------------------------------------------------------
    st.subheader("Top Strengths & Weaknesses")

    def strengths_weaknesses(row):
        strengths, weaknesses = [], []

        strengths.append("Positive momentum") if row["Momentum"] > 0 else weaknesses.append("Weak momentum")
        strengths.append("Low volatility") if row["Risk"] < 0.30 else weaknesses.append("High volatility")
        strengths.append("Reasonable PE ratio") if row["PE"] < 40 else weaknesses.append("Stretched PE ratio")
        strengths.append("Healthy PB ratio") if row["PB"] < 8 else weaknesses.append("Rich PB ratio")
        strengths.append("Dividend support") if row["DividendYield"] > 0.01 else weaknesses.append("Low or no dividend")

        return strengths, weaknesses

    for _, row in signals_df.iterrows():
        st.markdown(f"### {row['Ticker']}")
        strengths, weaknesses = strengths_weaknesses(row)

        st.markdown("**Strengths:**")
        for s in strengths:
            st.markdown(f"- {s}")

        st.markdown("**Weaknesses:**")
        for w in weaknesses:
            st.markdown(f"- {w}")

        st.markdown("---")
# ---------------------------------------------------------
# TAB 9 — OPTIMIZER (FINAL INSTITUTIONAL VERSION)
# ---------------------------------------------------------
with tab9:
    st.header("Portfolio Optimizer")

    # ---------------------------------------------------------
    # SAFETY CHECKS
    # ---------------------------------------------------------
    if returns_df.empty:
        st.warning("Return data unavailable. Cannot run optimizer.")
        st.stop()

    # ---------------------------------------------------------
    # PREPARE DATA
    # ---------------------------------------------------------
    tickers_opt = valid_tickers.copy()

    # Annualized mean returns & covariance
    mean_returns = returns_df.mean() * 252
    cov_matrix = returns_df.cov() * 252

    # Replace NaN with 0 to avoid optimizer crashes
    mean_returns = mean_returns.fillna(0)
    cov_matrix = cov_matrix.fillna(0)

    # Initial equal weights
    init_guess = np.array([1 / len(tickers_opt)] * len(tickers_opt))
    bounds = tuple((0, 1) for _ in tickers_opt)

    # Constraint: weights sum to 1
    def weight_constraint(weights):
        return np.sum(weights) - 1

    # ---------------------------------------------------------
    # PORTFOLIO PERFORMANCE FUNCTION
    # ---------------------------------------------------------
    def portfolio_performance(weights, mean_returns, cov_matrix):
        ret = np.dot(weights, mean_returns)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = ret / vol if vol > 0 else 0
        return ret, vol, sharpe

    # ---------------------------------------------------------
    # MINIMUM VARIANCE OPTIMIZER
    # ---------------------------------------------------------
    def min_variance():
        def objective(weights):
            return portfolio_performance(weights, mean_returns, cov_matrix)[1]

        result = minimize(
            objective,
            init_guess,
            method="SLSQP",
            bounds=bounds,
            constraints={"type": "eq", "fun": weight_constraint},
            options={"maxiter": 500}
        )

        return result.x if result.success else init_guess

    # ---------------------------------------------------------
    # MAXIMUM SHARPE OPTIMIZER
    # ---------------------------------------------------------
    def max_sharpe():
        def objective(weights):
            ret, vol, sharpe = portfolio_performance(weights, mean_returns, cov_matrix)
            return -sharpe

        result = minimize(
            objective,
            init_guess,
            method="SLSQP",
            bounds=bounds,
            constraints={"type": "eq", "fun": weight_constraint},
            options={"maxiter": 500}
        )

        return result.x if result.success else init_guess

    # ---------------------------------------------------------
    # RISK PARITY OPTIMIZER
    # ---------------------------------------------------------
    def risk_parity():
        def risk_contribution(weights):
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            mrc = np.dot(cov_matrix, weights) / port_vol
            rc = weights * mrc
            return rc

        def objective(weights):
            rc = risk_contribution(weights)
            return np.sum((rc - rc.mean()) ** 2)

        result = minimize(
            objective,
            init_guess,
            method="SLSQP",
            bounds=bounds,
            constraints={"type": "eq", "fun": weight_constraint},
            options={"maxiter": 500}
        )

        return result.x if result.success else init_guess

    # ---------------------------------------------------------
    # USER SELECTION
    # ---------------------------------------------------------
    st.subheader("Select Optimization Method")

    method = st.selectbox(
        "Optimization Method",
        ["Equal Weight", "Minimum Variance", "Maximum Sharpe", "Risk Parity"]
    )

    # ---------------------------------------------------------
    # RUN OPTIMIZER
    # ---------------------------------------------------------
    if st.button("Run Optimizer"):
        if method == "Equal Weight":
            opt_weights = init_guess
        elif method == "Minimum Variance":
            opt_weights = min_variance()
        elif method == "Maximum Sharpe":
            opt_weights = max_sharpe()
        elif method == "Risk Parity":
            opt_weights = risk_parity()

        # Normalize weights
        opt_weights = opt_weights / opt_weights.sum()

        # Save globally
        st.session_state.weights = opt_weights

        st.success("Optimizer completed successfully.")

        # ---------------------------------------------------------
        # DISPLAY RESULTS
        # ---------------------------------------------------------
        results_df = pd.DataFrame({
            "Ticker": tickers_opt,
            "Weight": opt_weights
        }).sort_values("Weight", ascending=False)

        st.subheader("Optimized Weights")
        st.dataframe(results_df, use_container_width=True)

        # Portfolio performance
        ret, vol, sharpe = portfolio_performance(opt_weights, mean_returns, cov_matrix)

        st.subheader("Optimized Portfolio Performance")
        st.write(f"**Expected Return:** {ret:.2%}")
        st.write(f"**Volatility:** {vol:.2%}")
        st.write(f"**Sharpe Ratio:** {sharpe:.2f}")

        # ---------------------------------------------------------
        # WEIGHTS BAR CHART
        # ---------------------------------------------------------
        fig = go.Figure(go.Bar(
            x=results_df["Ticker"],
            y=results_df["Weight"],
            marker_color="steelblue"
        ))
        fig.update_layout(height=400, title="Optimized Portfolio Weights")

        st.plotly_chart(fig, use_container_width=True, key="plot_1370")

