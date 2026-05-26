import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from scipy.optimize import minimize

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
# STEP 1 — Load Prices + Validate + Compute Returns
# ---------------------------------------------------------

prices = load_price_data(tickers, start_date, end_date)

if prices is None or prices.empty:
    st.error("Price data failed. Cannot continue.")
    st.stop()

# Compute returns
returns_df = prices.pct_change().dropna()

if returns_df.empty:
    st.error("Return data unavailable after pct_change().")
    st.stop()

valid_tickers = list(returns_df.columns)


# ---------------------------------------------------------
# STEP 2 — Load Fundamentals + Clean + Sector Extraction
# ---------------------------------------------------------

@st.cache_data
def load_fundamentals_auto(tickers):
    fundamentals = {}

    for t in tickers:
        try:
            yf_t = yf.Ticker(t)

            # Modern endpoints
            fast = yf_t.fast_info
            info = yf_t.get_info()  # safer than .info

            # Financial statements (fallbacks)
            fin = yf_t.financials
            bs = yf_t.balance_sheet

            # EPS fallback
            try:
                eps = fin.loc["Net Income"].iloc[0] / bs.loc["Common Stock"].iloc[0]
            except:
                eps = None

            fundamentals[t] = {
                "PE": fast.get("trailing_pe") or info.get("trailingPE"),
                "PB": fast.get("price_to_book") or info.get("priceToBook"),
                "DividendYield": fast.get("dividend_yield") or info.get("dividendYield"),
                "Beta": info.get("beta"),
                "MarketCap": fast.get("market_cap") or info.get("marketCap"),
                "EPS": eps,
                "Sector": info.get("sector") or "Unknown"
            }

        except Exception:
            fundamentals[t] = {
                "PE": None,
                "PB": None,
                "DividendYield": None,
                "Beta": None,
                "MarketCap": None,
                "EPS": None,
                "Sector": "Unknown"
            }

    return fundamentals

# Load fundamentals for valid tickers
fundamentals_raw = load_fundamentals_auto(valid_tickers)

# Convert to DataFrame
fundamentals_df = pd.DataFrame(fundamentals_raw).T

# Remove SPY — SPY should never appear in fundamentals
fundamentals_df = fundamentals_df[fundamentals_df.index != "SPY"]

# Validate fundamentals
if fundamentals_df.empty:
    st.error("Fundamentals could not be loaded. Cannot continue.")
    st.stop()

# Extract sectors cleanly
sector_map = fundamentals_df["Sector"].fillna("Unknown").to_dict()


# ---------------------------------------------------------
# STEP 3 — Fundamentals Scoring (GLOBAL)
# ---------------------------------------------------------

# Clean numeric columns
numeric_cols = ["PE", "PB", "DividendYield", "Beta", "MarketCap"]
for col in numeric_cols:
    if col in fundamentals_df.columns:
        fundamentals_df[col] = pd.to_numeric(fundamentals_df[col], errors="coerce")

# Replace missing values with median (prevents zeros)
fundamentals_df[numeric_cols] = fundamentals_df[numeric_cols].fillna(
    fundamentals_df[numeric_cols].median()
)

# Build scoring model
score_components = []

# Lower PE is better
if "PE" in fundamentals_df.columns:
    score_components.append(fundamentals_df["PE"].rank(pct=True, ascending=False))

# Lower PB is better
if "PB" in fundamentals_df.columns:
    score_components.append(fundamentals_df["PB"].rank(pct=True, ascending=False))

# Higher dividend yield is better
if "DividendYield" in fundamentals_df.columns:
    score_components.append(fundamentals_df["DividendYield"].rank(pct=True))

# Lower Beta is better
if "Beta" in fundamentals_df.columns:
    score_components.append(fundamentals_df["Beta"].rank(pct=True, ascending=False))

# Higher MarketCap is better
if "MarketCap" in fundamentals_df.columns:
    score_components.append(fundamentals_df["MarketCap"].rank(pct=True))

# Combine into a single score
if score_components:
    fundamentals_df["Score"] = sum(score_components)
else:
    fundamentals_df["Score"] = 0.0

# Normalize score to 0–100
fundamentals_df["Score"] = 100 * fundamentals_df["Score"] / fundamentals_df["Score"].max()


# ---------------------------------------------------------
# PORTFOLIO METRICS (GLOBAL)
# ---------------------------------------------------------

# Annualized return
annual_return = returns_df.mean().dot(
    np.array([1/len(valid_tickers)] * len(valid_tickers))
) * 252

# Annualized volatility
annual_volatility = returns_df.std().mean() * (252 ** 0.5)

# Sharpe ratio
sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0

# Portfolio beta (vs SPY)
try:
    spy = yf.download("SPY", start=start_date, end=end_date)["Adj Close"].pct_change().dropna()
    portfolio_returns = returns_df.mean(axis=1)
    portfolio_beta = np.cov(portfolio_returns, spy)[0][1] / np.var(spy)
except:
    portfolio_beta = 1.0

# Max drawdown
cum_returns = (1 + returns_df.mean(axis=1)).cumprod()
rolling_max = cum_returns.cummax()
drawdown = (cum_returns - rolling_max) / rolling_max
max_drawdown = drawdown.min()

# ---------------------------------------------------------
# STEP 4 — GLOBAL COMMENTARY INPUTS (FINAL + BULLETPROOF)
# ---------------------------------------------------------

# Ensure weights exist
if "weights" not in st.session_state:
    # initialize equal weights
    st.session_state["weights"] = {t: 1/len(valid_tickers) for t in valid_tickers}

# Convert stored weights to dict
stored_weights = st.session_state["weights"]

# Rebuild weights so they ALWAYS match valid_tickers
commentary_weights = []
for t in valid_tickers:
    commentary_weights.append(stored_weights.get(t, 1/len(valid_tickers)))

# Normalize again (safety)
total = sum(commentary_weights)
if total == 0:
    commentary_weights = [1/len(valid_tickers)] * len(valid_tickers)
else:
    commentary_weights = [w/total for w in commentary_weights]

# Save back to session_state as dict
st.session_state["weights"] = {t: commentary_weights[i] for i, t in enumerate(valid_tickers)}

# Build weights series
weights_series = pd.Series(commentary_weights, index=valid_tickers)

# Align fundamentals
commentary_df = fundamentals_df.reindex(valid_tickers).copy()
commentary_df["Weight"] = weights_series
commentary_df = commentary_df.dropna(subset=["Score"])
commentary_df = commentary_df.sort_values("Score", ascending=False)

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
    # PRICE CHART (OPTIONAL BUT USEFUL)
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

    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# TAB 2 — PERFORMANCE (FINAL VERSION)
# ---------------------------------------------------------
with tab2:
    st.header("Performance Metrics")

    # ---------------------------------------------------------
    # PORTFOLIO RETURNS (EQUAL-WEIGHT, ALWAYS VALID)
    # ---------------------------------------------------------
    portfolio_returns = returns_df.mean(axis=1)
    ret = pd.Series(portfolio_returns, name="Portfolio Return")

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
    sortino = (ret.mean() * 252) / (ret[ret < 0].std() * np.sqrt(252)) if ret[ret < 0].std() > 0 else 0
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

    st.markdown("### Distribution of Daily Returns")
    hist_data = ret.dropna()
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
with tab4:
    st.subheader("Sector Exposure")

    # Safety: ensure fundamentals_df exists
    if fundamentals_df.empty:
        st.warning("Sector data unavailable. Run analysis first.")
        st.stop()

    # Align fundamentals to valid tickers
    fdf = fundamentals_df.reindex(valid_tickers).copy()

    # Ensure Sector column exists
    if "Sector" not in fdf.columns:
        fdf["Sector"] = "Unknown"

    # Clean sector values
    fdf["Sector"] = fdf["Sector"].fillna("Unknown")

    # Load weights from session_state
    if "weights" in st.session_state:
        w = st.session_state.weights
    else:
        w = np.array([1/len(valid_tickers)] * len(valid_tickers))

    # Align weights to valid tickers
    w_series = pd.Series(w, index=valid_tickers)

    # Group by sector
    sector_weights = w_series.groupby(fdf["Sector"]).sum().sort_values(ascending=False)

    # Display table
    st.markdown("### Sector Allocation Breakdown")
    st.dataframe(sector_weights.to_frame("Weight").style.format({"Weight": "{:.2%}"}))

    # Display chart
    st.markdown("### Sector Chart")
    fig = go.Figure(go.Bar(
        x=sector_weights.index,
        y=sector_weights.values,
        marker_color="steelblue"
    ))
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# TAB 5 — FUNDAMENTALS (FINAL VERSION)
# ---------------------------------------------------------
with tab5:
    st.header("Fundamentals")

    # ---------------------------------------------------------
    # SAFETY CHECK
    # ---------------------------------------------------------
    if fundamentals_df.empty:
        st.warning("No fundamentals available. Run analysis first.")
        st.stop()

    # ---------------------------------------------------------
    # CLEAN FUNDAMENTALS TABLE
    # ---------------------------------------------------------
    # Drop SPY if present
    fdf = fundamentals_df.copy()
    fdf = fdf.reindex(valid_tickers)

    # Fill missing values
    fdf = fdf.fillna(0)

    # Do NOT show Sector here (your S2 choice)
    fundamentals_display = fdf.drop(columns=["Sector"], errors="ignore")
    st.subheader("Raw Fundamentals")
    st.dataframe(fundamentals_display, use_container_width=True)

    # ---------------------------------------------------------
    # FUNDAMENTALS RANKING
    # ---------------------------------------------------------
    st.subheader("Fundamentals Ranking")

    def score_fundamentals(row):
        score = 0
        if row.get("ROE"):
            score += row["ROE"] * 10
        if row.get("EPS"):
            score += row["EPS"]
        if row.get("PE"):
            score += max(0, 50 - row["PE"])
        if row.get("PB"):
            score += max(0, 20 - row["PB"])
        if row.get("DividendYield"):
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
        if best_row.get("ROE"):
            best_reasons.append("strong return on equity")
        if best_row.get("EPS"):
            best_reasons.append("solid earnings power")
        if best_row.get("PE") and best_row["PE"] < 25:
            best_reasons.append("reasonable valuation")
        if best_row.get("DividendYield"):
            best_reasons.append("added income from dividends")
        best_reason_text = ", ".join(best_reasons) if best_reasons else "overall stronger fundamentals"
        lines.append(f"**{best}** ranks as the strongest fundamental name in the group, supported by {best_reason_text}.")

        # Middle
        if len(tickers_sorted) > 2:
            middle = tickers_sorted[1:-1]
            for t in middle:
                row = ranked_df.loc[t]
                mid_reasons = []
                if row.get("ROE"):
                    mid_reasons.append("healthy ROE")
                if row.get("EPS"):
                    mid_reasons.append("stable earnings")
                if row.get("PE") and row["PE"] < 40:
                    mid_reasons.append("fair valuation")
                reason_text = ", ".join(mid_reasons) if mid_reasons else "balanced fundamentals"
                lines.append(f"**{t}** shows {reason_text}, placing it in the middle of the group.")

        # Worst
        worst_row = ranked_df.loc[worst]
        worst_reasons = []
        if worst_row.get("ROE") and worst_row["ROE"] < 0.05:
            worst_reasons.append("weak ROE")
        if worst_row.get("PE") and worst_row["PE"] > 50:
            worst_reasons.append("elevated valuation")
        if worst_row.get("PB") and worst_row["PB"] > 10:
            worst_reasons.append("rich price-to-book ratio")
        worst_reason_text = ", ".join(worst_reasons) if worst_reasons else "weaker fundamentals overall"
        lines.append(f"**{worst}** ranks lowest, driven by {worst_reason_text}.")

        return "\n\n".join(lines)

    st.markdown(generate_fundamentals_commentary(ranked_df))

    # ---------------------------------------------------------
    # SIMPLE COMMENTARY LIST
    # ---------------------------------------------------------
    st.subheader("Commentary")
    commentary = [f"- **{ticker}**: score {row['score']:.1f}" for ticker, row in ranked_df.iterrows()]
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
# TAB 7 — AI COMMENTARY (FINAL FIXED VERSION)
# ---------------------------------------------------------
with tab7:
    st.header("AI Portfolio Commentary")

    # Safety checks
    if fundamentals_df.empty:
        st.warning("Fundamentals not available. Run analysis first.")
        st.stop()

    if "weights" not in st.session_state:
        st.warning("Weights not set. Adjust weights in Tab 6.")
        st.stop()

    # ---------------------------------------------------------
    # PREPARE COMMENTARY DATAFRAME
    # ---------------------------------------------------------
    commentary_df = fundamentals_df.copy()

    # Add weights
    commentary_df["Weight"] = st.session_state.weights

    # Compute fundamentals score (same logic as Tab 5)
    def score_fundamentals(row):
        score = 0
        if row.get("ROE"):
            score += row["ROE"] * 10
        if row.get("EPS"):
            score += row["EPS"]
        if row.get("PE"):
            score += max(0, 50 - row["PE"])
        if row.get("PB"):
            score += max(0, 20 - row["PB"])
        if row.get("DividendYield"):
            score += row["DividendYield"] * 100
        return score

    commentary_df["score"] = commentary_df.apply(score_fundamentals, axis=1)

    # Remove SPY if present
    commentary_df = commentary_df[commentary_df.index != "SPY"]

    # Sort by score (lowercase)
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

    # Portfolio metrics (already computed globally)
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
# TAB 8 — BUY ANALYSIS (FINAL ROBUST VERSION)
# ---------------------------------------------------------
with tab8:
    st.header("AI Buy / Hold / Sell Analysis")

    # Safety checks
    if fundamentals_df.empty:
        st.warning("Fundamentals not available. Run analysis first.")
        st.stop()

    if "weights" not in st.session_state:
        st.warning("Weights not set. Adjust weights in Tab 6.")
        st.stop()

    # Align fundamentals with valid tickers
    analysis_df = fundamentals_df.reindex(valid_tickers).copy()

    # Add weights
    analysis_df["Weight"] = st.session_state.weights

    # Remove SPY if present
    analysis_df = analysis_df[analysis_df.index != "SPY"]

    # Ensure required columns exist
    for col in ["PE", "PB", "DividendYield", "Beta"]:
        if col not in analysis_df.columns:
            analysis_df[col] = np.nan

    # Compute momentum (3‑month return approx)
    if not returns_df.empty:
        momentum_series = returns_df.tail(63).sum()
        analysis_df["Momentum"] = momentum_series.reindex(analysis_df.index)
        # Risk (annualized volatility)
        common = [c for c in analysis_df.index if c in returns_df.columns]
        if common:
            analysis_df.loc[common, "Risk"] = (
                returns_df[common].std() * np.sqrt(252)
            ).reindex(common)
        else:
            analysis_df["Risk"] = np.nan
    else:
        analysis_df["Momentum"] = np.nan
        analysis_df["Risk"] = np.nan

    # Clean missing numeric values
    analysis_df = analysis_df.fillna(analysis_df.median(numeric_only=True))

    # Generate Buy/Hold/Sell signals
    signals = []
    for t, row in analysis_df.iterrows():
        score = 0
        conviction = 0

        # PE filter
        if row["PE"] > 0 and row["PE"] < 40:
            score += 1
            conviction += 20

        # PB filter
        if row["PB"] > 0 and row["PB"] < 8:
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

    def rating_color(val):
        return {"Buy": "🟢 Buy", "Hold": "🟡 Hold", "Sell": "🔴 Sell"}[val]

    signals_df["RatingColored"] = signals_df["Rating"].apply(rating_color)

    st.subheader("AI Buy / Hold / Sell Signals")
    st.dataframe(
        signals_df[
            ["Ticker", "PE", "PB", "DividendYield", "Beta", "Momentum", "Risk",
             "Score", "Conviction", "RatingColored"]
        ],
        use_container_width=True
    )

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

    st.subheader("AI Portfolio‑Level Signal")
    buy_count = len(buys)
    sell_count = len(sells)

    if buy_count > sell_count:
        st.success("**AI Portfolio Signal: BUY** — Broad fundamental strength detected.")
    elif sell_count >= buy_count + 2:
        st.error("**AI Portfolio Signal: SELL** — Broad fundamental weakness detected.")
    else:
        st.warning("**AI Portfolio Signal: HOLD** — Mixed signals across the portfolio.")

    st.subheader("Fundamentals Radar Chart")
    radar_cols = ["PE", "PB", "DividendYield", "Momentum", "Risk"]
    fig = go.Figure()
    for _, row in signals_df.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row[c] if pd.notna(row[c]) else 0 for c in radar_cols],
            theta=radar_cols,
            fill='toself',
            name=row["Ticker"]
        ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True, height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Strengths & Weaknesses")

    def strengths_weaknesses(row):
        strengths, weaknesses = [], []

        if row["Momentum"] > 0:
            strengths.append("Positive momentum")
        else:
            weaknesses.append("Weak momentum")

        if row["Risk"] < 0.30:
            strengths.append("Low volatility")
        else:
            weaknesses.append("High volatility")

        if row["PE"] > 40:
            weaknesses.append("Stretched PE ratio")
        else:
            strengths.append("Reasonable PE ratio")

        if row["PB"] > 8:
            weaknesses.append("Rich PB ratio")
        else:
            strengths.append("Healthy PB ratio")

        if row["DividendYield"] > 0.01:
            strengths.append("Dividend support")
        else:
            weaknesses.append("Low or no dividend")

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

    if returns_df.empty:
        st.warning("Return data unavailable. Cannot run optimizer.")
        st.stop()

    # Annualized stats
    mean_returns = returns_df.mean() * 252
    cov_matrix = returns_df.cov() * 252

    # Safety: remove NaNs
    mean_returns = mean_returns.fillna(0)
    cov_matrix = cov_matrix.fillna(0)

    # Align tickers
    tickers_opt = valid_tickers.copy()

    # Helper: portfolio performance
    def portfolio_performance(weights, mean_returns, cov_matrix):
        ret = np.dot(weights, mean_returns)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = ret / vol if vol > 0 else 0
        return ret, vol, sharpe

    def weight_constraint(weights):
        return np.sum(weights) - 1

    bounds = tuple((0, 1) for _ in tickers_opt)
    init_guess = np.array([1/len(tickers_opt)] * len(tickers_opt))

    # -----------------------------
    # Minimum Variance
    # -----------------------------
    def min_variance():
        def objective(weights):
            return portfolio_performance(weights, mean_returns, cov_matrix)[1]

        result = minimize(
            objective, init_guess, method="SLSQP",
            bounds=bounds, constraints={"type": "eq", "fun": weight_constraint}
        )

        if not result.success:
            st.warning("Min-variance optimizer failed. Using equal weights.")
            return init_guess

        return result.x

    # -----------------------------
    # Maximum Sharpe
    # -----------------------------
    def max_sharpe():
        def objective(weights):
            ret, vol, sharpe = portfolio_performance(weights, mean_returns, cov_matrix)
            return -sharpe

        result = minimize(
            objective, init_guess, method="SLSQP",
            bounds=bounds, constraints={"type": "eq", "fun": weight_constraint}
        )

        if not result.success:
            st.warning("Max-sharpe optimizer failed. Using equal weights.")
            return init_guess

        return result.x

    # -----------------------------
    # Risk Parity
    # -----------------------------
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
            objective, init_guess, method="SLSQP",
            bounds=bounds, constraints={"type": "eq", "fun": weight_constraint}
        )

        if not result.success:
            st.warning("Risk parity optimizer failed. Using equal weights.")
            return init_guess

        return result.x

    # -----------------------------
    # Run optimizer
    # -----------------------------
    st.subheader("Select Optimization Method")

    method = st.selectbox(
        "Optimization Method",
        ["Equal Weight", "Minimum Variance", "Maximum Sharpe", "Risk Parity"]
    )

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

        results_df = pd.DataFrame({
            "Ticker": tickers_opt,
            "Weight": opt_weights
        }).sort_values("Weight", ascending=False)

        st.subheader("Optimized Weights")
        st.dataframe(results_df, use_container_width=True)

        ret, vol, sharpe = portfolio_performance(opt_weights, mean_returns, cov_matrix)

        st.subheader("Optimized Portfolio Performance")
        st.write(f"**Expected Return:** {ret:.2%}")
        st.write(f"**Volatility:** {vol:.2%}")
        st.write(f"**Sharpe Ratio:** {sharpe:.2f}")

        fig = go.Figure(go.Bar(
            x=results_df["Ticker"],
            y=results_df["Weight"],
            marker_color="steelblue"
        ))
        fig.update_layout(height=400, title="Optimized Portfolio Weights")
        st.plotly_chart(fig, use_container_width=True)
