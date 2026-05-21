import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

from datetime import datetime, timedelta

from utils.data_loader import load_price_data
from utils.fundamentals_loader import load_fundamentals
from utils.optimizer_core import run_optimizer
from utils.buy_analysis import run_buy_analysis
from utils.analytics import run_monte_carlo_simulation
import plotly.graph_objects as go

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

    text-align: center;   /* ⭐ CENTER THE TITLE */

    border-bottom: 1px solid #E5E5E5;
    box-shadow: 0 1px 4px rgba(0,0,0,0.10);

    z-index: 9999;
}

/* Remove Streamlit header */
header[data-testid="stHeader"] {
    display: none !important;
}

/* Remove white overlay */
div[data-testid="stAppViewContainer"] {
    background-color: transparent !important;
}

/* Push content down */
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
import yfinance as yf

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

prices = load_price_data(tickers, start_date, end_date)

st.write("DEBUG — prices shape:", prices.shape)
st.write(prices.head())

if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

st.sidebar.subheader("Analysis Controls")
run_button = st.sidebar.button("Run Analysis")

mc_sims = st.sidebar.slider("Monte Carlo Simulations", 200, 3000, 500)
mc_horizon = st.sidebar.slider("Monte Carlo Horizon (days)", 50, 500, 252)

def load_price_data(tickers, start, end):
    try:
        raw = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            group_by="ticker"
        )

        if raw is None or raw.empty:
            return pd.DataFrame()

        # ---------------------------------------------------------
        # MULTI-TICKER (MultiIndex)
        # ---------------------------------------------------------
        if isinstance(raw.columns, pd.MultiIndex):

            # Case 1: Level 0 contains fields (Adj Close, Close, etc.)
            if "Adj Close" in raw.columns.get_level_values(0):
                adj = raw["Adj Close"]

            # Case 2: Level 1 contains fields
            elif "Adj Close" in raw.columns.get_level_values(1):
                adj = raw.xs("Adj Close", level=1, axis=1)

            # Fallback to Close
            elif "Close" in raw.columns.get_level_values(1):
                adj = raw.xs("Close", level=1, axis=1)

            else:
                return pd.DataFrame()

        # ---------------------------------------------------------
        # SINGLE-TICKER (flat columns)
        # ---------------------------------------------------------
        else:
            if "Adj Close" in raw.columns:
                adj = raw["Adj Close"]
            elif "Close" in raw.columns:
                adj = raw["Close"]
            else:
                return pd.DataFrame()

        # Ensure DataFrame
        if isinstance(adj, pd.Series):
            adj = adj.to_frame()

        # Keep only requested tickers
        adj = adj[[t for t in tickers if t in adj.columns]]

        return adj

    except Exception as e:
        st.error(f"Price load failed: {e}")
        return pd.DataFrame()

# ---------------------------------------------------------
# SINGLE‑TICKER FUNDAMENTALS LOADER (MODERN + RELIABLE)
# ---------------------------------------------------------
def load_fundamentals(ticker):
    try:
        t = yf.Ticker(ticker)
        fi = t.fast_info

        pe = fi.get("pe_ratio")
        pb = fi.get("pb_ratio")
        mcap = fi.get("market_cap")
        beta = fi.get("beta")
        div = fi.get("dividend_yield")

        if div is not None:
            div = div * 100

        try:
            sector = t.info.get("sector")
        except:
            sector = "Unknown"

        return pd.DataFrame([{
            "PE": pe or 0,
            "PB": pb or 0,
            "DividendYield": div or 0,
            "Beta": beta or 0,
            "MarketCap": mcap or 0,
            "Sector": sector or "Unknown"
        }], index=[ticker])

    except Exception:
        return pd.DataFrame([{
            "PE": 0,
            "PB": 0,
            "DividendYield": 0,
            "Beta": 0,
            "MarketCap": 0,
            "Sector": "Unknown"
        }], index=[ticker])

# ---------------------------------------------------------
# MULTI‑TICKER FUNDAMENTALS LOADER
# ---------------------------------------------------------
def load_fundamentals_multi(tickers):
    frames = []

    for t in tickers:
        try:
            df = load_fundamentals(t)
            if isinstance(df, pd.DataFrame):
                frames.append(df)
            else:
                raise ValueError("Single‑ticker loader returned non‑DataFrame")
        except Exception:
            frames.append(pd.DataFrame([{
                "PE": None,
                "PB": None,
                "DividendYield": None,
                "Beta": None,
                "MarketCap": None,
                "Sector": "Unknown"
            }], index=[t]))

    if len(frames) == 0:
        return pd.DataFrame()

    df_all = pd.concat(frames)
    df_all["Sector"] = df_all["Sector"].fillna("Unknown").replace("", "Unknown")
    return df_all


# ---------------------------------------------------------
# GLOBAL DATA PIPELINE (REQUIRED FOR ALL TABS)
# ---------------------------------------------------------

# 1. Load prices
if prices is None or prices.empty:
    st.error("Price data could not be loaded.")
    st.stop()

# 2. Load or create weights
if "weights" in st.session_state and len(st.session_state.weights) == len(tickers):
    weights = np.array(st.session_state.weights, dtype=float)
else:
    weights = np.array([1 / len(tickers)] * len(tickers), dtype=float) if len(tickers) > 0 else np.array([])

# 3. Remove SPY from portfolio tickers
portfolio_tickers = [t for t in tickers if t in prices.columns and t != "SPY"]

# 4. Compute returns
returns_full = prices.pct_change().dropna()
returns = prices[portfolio_tickers].pct_change().dropna() if len(portfolio_tickers) > 0 else pd.DataFrame()

# 5. Align weights
ticker_to_weight = dict(zip(tickers, weights))
valid_weights = np.array([ticker_to_weight[t] for t in portfolio_tickers]) if len(portfolio_tickers) > 0 else np.array([])

# 6. Compute portfolio returns
portfolio_returns = returns @ valid_weights if len(portfolio_tickers) > 0 else pd.Series(dtype=float)

# 7. Load fundamentals (NOW WORKS)
fundamentals = load_fundamentals_multi(portfolio_tickers)

# SAFETY CHECK
if not isinstance(fundamentals, pd.DataFrame):
    st.error("Fundamentals loader returned invalid data.")
    st.stop()

# 8. Store valid tickers globally
#valid_tickers = portfolio_tickers
valid_tickers = portfolio_tickers

# Ensure SPY exists for Beta
if "SPY" not in prices.columns:
    spy_data = load_price_data(["SPY"], start_date, end_date)
    if spy_data is not None and not spy_data.empty:
        prices["SPY"] = spy_data["SPY"]
    else:
        st.warning("SPY could not be loaded. Beta vs SPY unavailable.")

# Clean returns
returns_df = prices.pct_change().ffill().bfill()

# ---------------------------------------------------------
# REMOVE SPY FROM RETURNS_DF (CRITICAL)
# ---------------------------------------------------------
if "SPY" in returns_df.columns:
    returns_df = returns_df.drop(columns=["SPY"])

# ---------------------------------------------------------
# REALIGN WEIGHTS TO MATCH RETURNS_DF
# ---------------------------------------------------------
aligned_tickers = [t for t in portfolio_tickers if t in returns_df.columns]

aligned_weights = np.array([ticker_to_weight[t] for t in aligned_tickers], dtype=float)

# Normalize
if aligned_weights.sum() > 0:
    aligned_weights = aligned_weights / aligned_weights.sum()
else:
    aligned_weights = np.array([1 / len(aligned_tickers)] * len(aligned_tickers))

# Replace global tickers + weights
valid_tickers = aligned_tickers
weights = aligned_weights


# DO NOT OVERWRITE valid_tickers
# valid_tickers must remain portfolio_tickers

if len(valid_tickers) == 0:
    st.error("No valid tickers after cleaning returns.")
    st.stop()

# ---------------------------------------------------------
# Auto Sector Detection (no manual map needed)
# ---------------------------------------------------------
sector_map = {
    t: fundamentals.loc[t, "Sector"] if t in fundamentals.index else "Unknown"
    for t in valid_tickers
}
#sector_map = {t: fundamentals[t].get("Sector", "Unknown") for t in valid_tickers}

# ---------------------------------------------------------
# PREP FOR METRICS
# ---------------------------------------------------------
weights = np.array([1 / len(valid_tickers)] * len(valid_tickers))

# ---------------------------------------------------------
# BUILD TICKER → WEIGHT MAP
# ---------------------------------------------------------
ticker_to_weight = {t: 1 / len(portfolio_tickers) for t in portfolio_tickers}

# ---------------------------------------------------------
# REALIGN RETURNS_DF AND WEIGHTS BEFORE PORTFOLIO METRICS
# ---------------------------------------------------------

# Clean returns
returns_df = prices[portfolio_tickers].pct_change().ffill().bfill()

# Remove SPY if it sneaks in
if "SPY" in returns_df.columns:
    returns_df = returns_df.drop(columns=["SPY"])

# Align tickers
aligned_tickers = [t for t in portfolio_tickers if t in returns_df.columns]

# Align weights
aligned_weights = np.array([ticker_to_weight[t] for t in aligned_tickers], dtype=float)

# Normalize
if aligned_weights.sum() > 0:
    aligned_weights = aligned_weights / aligned_weights.sum()
else:
    aligned_weights = np.array([1 / len(aligned_tickers)] * len(aligned_tickers))

# Replace global tickers + weights
valid_tickers = aligned_tickers
weights = aligned_weights

# ---------------------------------------------------------
# PORTFOLIO METRICS (GLOBAL)
# ---------------------------------------------------------
try:
    portfolio_returns = returns_df.dot(weights)

    annual_return = portfolio_returns.mean() * 252
    annual_volatility = portfolio_returns.std() * (252 ** 0.5)
    sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0

    max_drawdown = (portfolio_returns.cummax() - portfolio_returns).max()

    corr_matrix = returns_df.corr()
    avg_corr = corr_matrix.where(~np.eye(corr_matrix.shape[0], dtype=bool)).mean().mean()
    diversification_score = max(0, min(10, (1 - avg_corr) * 10))

    # Compute portfolio beta vs SPY (safe version)
    if "SPY" in prices.columns:
        spy_returns = prices["SPY"].pct_change().dropna()
        common_index = portfolio_returns.index.intersection(spy_returns.index)

        if len(common_index) > 0:
            covariance = portfolio_returns.loc[common_index].cov(spy_returns.loc[common_index])
            market_variance = spy_returns.loc[common_index].var()
            portfolio_beta = covariance / market_variance if market_variance > 0 else 0
        else:
            portfolio_beta = 0
    else:
        portfolio_beta = 0

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
# TAB 1 — OVERVIEW
# ---------------------------------------------------------
with tab1:
    st.subheader("Portfolio Overview")

    # --- FIRST ROW OF METRICS (col1, col2, col3) ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Annual Return", f"{annual_return:.2%}")
    with col2:
        st.metric("Volatility", f"{annual_volatility:.2%}")
    with col3:
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

    # --- SECOND ROW OF METRICS (col4, col5, col6) ---
    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("Max Drawdown", f"{max_drawdown:.2%}")
    with col5:
        st.metric("Beta vs SPY", f"{portfolio_beta:.2f}")
    with col6:
        st.metric("Diversification Score", f"{diversification_score:.1f}/10")

# ---------------------------------------------------------
# TAB 2 — PERFORMANCE
# ---------------------------------------------------------
with tab2:
    st.subheader("Performance Analysis")

    # === Portfolio Return Series (required for rolling metrics) ===
    ret = pd.Series(portfolio_returns, name="Portfolio Return")

    # === Cumulative Returns ===
    st.markdown("### Cumulative Returns")
    cum_returns = (1 + ret).cumprod()
    st.line_chart(cum_returns)

    # === Rolling Sharpe (30-day) ===
    st.markdown("### Rolling Sharpe Ratio (30-day)")
    rolling_sharpe = (
        ret.rolling(30).mean() /
        ret.rolling(30).std().replace(0, np.nan)
    ) * np.sqrt(252)
    st.line_chart(rolling_sharpe)

    # === Rolling Beta vs SPY ===
    st.markdown("### Rolling Beta vs SPY")

    if "SPY" in prices.columns:
        spy_ret = prices["SPY"].pct_change().dropna()
        rolling_beta = (
            ret.rolling(60).cov(spy_ret) /
            spy_ret.rolling(60).var()
        )
        st.line_chart(rolling_beta)
    else:
        st.info("SPY data unavailable — cannot compute rolling beta.")

# ---------------------------------------------------------
# Tab 3 — Portfolio Weights & Shares
# ---------------------------------------------------------
with tab3:
    st.subheader("Portfolio Weights & Shares")

    # ⭐ GUARD CLAUSE — prevents empty sliders ⭐
    if "portfolio_weights" not in st.session_state:
        st.info("Run the Optimizer tab first to calculate portfolio weights.")
        st.stop()

    portfolio_weights = st.session_state["portfolio_weights"]

    # Display weights
    st.markdown("### Optimized Portfolio Weights")
    weights_df = pd.DataFrame({
        "Ticker": tickers,
        "Weight": portfolio_weights
    })
    st.dataframe(weights_df, use_container_width=True)

    # Shares calculation (if price data exists)
    if "latest_prices" in st.session_state:
        latest_prices = st.session_state["latest_prices"]

        st.markdown("### Shares Based on Portfolio Value")
        portfolio_value = st.number_input(
            "Enter total portfolio value ($)",
            min_value=1000,
            value=100000,
            step=1000
        )

        shares = (portfolio_value * portfolio_weights) / latest_prices
        shares_df = pd.DataFrame({
            "Ticker": tickers,
            "Price": latest_prices,
            "Weight": portfolio_weights,
            "Shares": shares
        })

        st.dataframe(shares_df, use_container_width=True)
    else:
        st.warning("Price data missing — shares cannot be calculated.")

# ---------------------------------------------------------
# Tab 4 — Sector Exposure
# ---------------------------------------------------------
with tab4:
    st.subheader("Sector Exposure")

    # ⭐ GUARD CLAUSE ⭐
    if "portfolio_weights" not in st.session_state:
        st.info("Run the Optimizer tab first to calculate portfolio weights.")
        st.stop()

    portfolio_weights = st.session_state["portfolio_weights"]

    # Fundamentals must exist
    if "fundamentals" not in st.session_state:
        st.warning("Fundamentals data missing — cannot compute sector exposure.")
        st.stop()

    fundamentals = st.session_state["fundamentals"]

    # Ensure Sector column exists
    if "Sector" not in fundamentals.columns:
        fundamentals["Sector"] = "Unknown"

    # Build sector mapping
    sector_map = fundamentals["Sector"]

    # Compute weighted sector exposure
    sector_weights = pd.Series(portfolio_weights, index=tickers).groupby(sector_map).sum()

    # Display table
    st.markdown("### Sector Allocation Table")
    st.dataframe(sector_weights.to_frame("Weight").style.format("{:.2%}"))

    # Pie chart
    fig = go.Figure(data=[go.Pie(
        labels=sector_weights.index,
        values=sector_weights.values,
        hole=0.4
    )])
    fig.update_layout(title="Sector Allocation")
    st.plotly_chart(fig, use_container_width=True)
# ---------------------------------------------------------
# Tab 5 — Fundamentals
# ---------------------------------------------------------
with tab5:
    st.subheader("Fundamentals")

    # Guard clause
    if "portfolio_weights" not in st.session_state:
        st.info("Run the Optimizer tab first to calculate portfolio weights.")
        st.stop()

    portfolio_weights = st.session_state["portfolio_weights"]

    # Load fundamentals
    if "fundamentals" not in st.session_state:
        st.error("Fundamentals data missing — cannot display fundamentals.")
        st.stop()

    fundamentals = st.session_state["fundamentals"]

    # Ensure DataFrame
    if isinstance(fundamentals, dict):
        fundamentals = pd.DataFrame(fundamentals).T

    if not isinstance(fundamentals, pd.DataFrame):
        st.error("Fundamentals loader returned invalid data.")
        st.stop()

    st.write("DEBUG — fundamentals head:")
    st.write(fundamentals.head())

    # Clean fundamentals
    fundamentals_df = fundamentals.copy()
    fundamentals_df = fundamentals_df.drop("full_prices", errors="ignore")
    fundamentals_df = fundamentals_df.fillna(0)

    # Display fundamentals (hide sector)
    fundamentals_display = fundamentals_df.drop(columns=["Sector"], errors="ignore")
    st.dataframe(fundamentals_display)

    # ---------------------------------------------------------
    # Fundamentals Ranking
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

    fundamentals_df["score"] = fundamentals_df.apply(score_fundamentals, axis=1)
    ranked_df = fundamentals_df.sort_values("score", ascending=False)
    st.dataframe(ranked_df[["score"]])

    # ---------------------------------------------------------
    # AI Fundamentals Commentary
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

    # Simple commentary list
    st.subheader("Commentary")
    commentary = [f"- **{ticker}**: score {row['score']:.1f}" for ticker, row in ranked_df.iterrows()]
    st.markdown("\n".join(commentary))

# ---------------------------------------------------------
# Tab 6 — Weights
# ---------------------------------------------------------
with tab6:
    st.subheader("Weights")

    if "portfolio_weights" not in st.session_state:
        st.info("Run the Optimizer tab first to calculate portfolio weights.")
        st.stop()

    portfolio_weights = st.session_state["portfolio_weights"]

    if len(valid_tickers) == 0:
        st.warning("No valid tickers available to assign weights.")
        st.stop()

    # ---------------------------------------------------------
    # INITIAL WEIGHTS = optimizer weights
    # ---------------------------------------------------------
    weights_dict = {t: float(portfolio_weights[i]) for i, t in enumerate(valid_tickers)}

    st.subheader("Adjust Weights")

    # ---------------------------------------------------------
    # SLIDERS — USER-ADJUSTABLE WEIGHTS
    # ---------------------------------------------------------
    for t in valid_tickers:
        weights_dict[t] = st.slider(
            f"{t} Weight",
            min_value=0.0,
            max_value=1.0,
            value=float(weights_dict[t]),
            key=f"weight_{t}"
        )

    # ---------------------------------------------------------
    # NORMALIZE WEIGHTS
    # ---------------------------------------------------------
    total = sum(weights_dict.values())
    if total > 0:
        weights_dict = {t: weight/total for t, weight in weights_dict.items()}

    # ---------------------------------------------------------
    # CONVERT TO NUMPY ARRAY
    # ---------------------------------------------------------
    w = np.array([weights_dict[t] for t in valid_tickers])

    # ⭐ SAVE UPDATED WEIGHTS ⭐
    st.session_state["portfolio_weights"] = w

    # ---------------------------------------------------------
    # DISPLAY WEIGHTS TABLE
    # ---------------------------------------------------------
    st.subheader("Final Normalized Weights")
    weights_df = pd.DataFrame({
        "Ticker": valid_tickers,
        "Weight": [weights_dict[t] for t in valid_tickers]
    })
    st.dataframe(weights_df, use_container_width=True)

# ---------------------------------------------------------
# Tab 7 — AI Commentary + Signals
# ---------------------------------------------------------
with tab7:
    st.subheader("AI Commentary")

    # Guard clause
    if "portfolio_weights" not in st.session_state:
        st.info("Run the Optimizer tab first to calculate portfolio weights.")
        st.stop()

    portfolio_weights = st.session_state["portfolio_weights"]

    # Load model safely
    model = st.session_state.get("model", {})

    # SAFE FALLBACKS for missing model keys
    perf = model.get("performance", {
        "expected_return": None,
        "volatility": None,
        "sharpe": None
    })

    fundamentals_model = model.get("fundamentals", {})
    tickers_model = model.get("tickers", tickers)

    drawdown_model = model.get("drawdown", None)
    sector_weights_model = model.get("sector_weights", {})
    mc_model = model.get("monte_carlo", None)
    momentum_model = model.get("momentum", {})

    # If performance is missing → stop gracefully
    if perf.get("expected_return") is None:
        st.warning("Performance metrics missing — cannot generate AI commentary.")
        st.stop()

    # Extract metrics
    er = perf["expected_return"]
    vol = perf["volatility"]
    sharpe_m = perf["sharpe"]

    # Drawdown
    if isinstance(drawdown_model, pd.DataFrame) and not drawdown_model.empty:
        max_dd_m = float(drawdown_model["Drawdown"].min())
    else:
        max_dd_m = None

    max_dd_text = f"{max_dd_m:.2%}" if isinstance(max_dd_m, (int, float, np.floating)) else "N/A"

    # Sector weights
    sector_text = ""
    if isinstance(sector_weights_model, dict) and sector_weights_model:
        sector_text = ", ".join([f"{s}: {w:.1%}" for s, w in sector_weights_model.items()])

    # Fundamentals summary
    fund_summary = []
    for t in tickers_model:
        f = fundamentals_model.get(t, {})
        fund_summary.append({
            "Ticker": t,
            "PE": f.get("PE"),
            "PB": f.get("PB"),
            "DividendYield": f.get("DividendYield"),
            "Beta": f.get("Beta"),
            "MarketCap": f.get("MarketCap"),
            "Sector": f.get("Sector", "Unknown"),
        })
    fund_df = pd.DataFrame(fund_summary)

    # Portfolio grade
    grade = "C"
    if sharpe_m and er:
        if sharpe_m > 1.2 and er > 0.12:
            grade = "A"
        elif sharpe_m > 0.8 and er > 0.08:
            grade = "B"
        elif sharpe_m < 0.3 or er < 0.03:
            grade = "D"

    # Risk bucket
    if vol is None:
        risk_bucket = "Unknown"
    elif vol < 0.12:
        risk_bucket = "Low Risk"
    elif vol < 0.20:
        risk_bucket = "Moderate Risk"
    else:
        risk_bucket = "High Risk"

    # Monte Carlo commentary
    mc_comment = ""
    if isinstance(mc_model, pd.DataFrame) and not mc_model.empty:
        final_vals = mc_model.iloc[-1]
        p5 = np.percentile(final_vals, 5)
        p50 = np.percentile(final_vals, 50)
        p95 = np.percentile(final_vals, 95)
        mc_comment = (
            f"Simulations show a **5% worst-case outcome of {p5:.2f}x**, "
            f"a **median outcome of {p50:.2f}x**, and a **best-case outcome of {p95:.2f}x**."
        )

    # -----------------------------
    # Portfolio Overview
    # -----------------------------
    st.markdown("### Portfolio Overview")
    st.write(
        f"""
**Portfolio Grade:** {grade}  
**Risk Bucket:** {risk_bucket}  
**Expected Annual Return:** {er:.2%}  
**Annualized Volatility:** {vol:.2%}  
**Sharpe Ratio:** {sharpe_m:.2f}  
**Max Drawdown:** {max_dd_text}  
"""
    )

    st.markdown("---")
    st.markdown("### AI Commentary")

    # Expected return commentary
    if er > 0.15:
        st.write("• Strong expected returns suggest meaningful upside potential.")
    elif er > 0.05:
        st.write("• Expected returns are moderate and consistent with balanced equity exposure.")
    else:
        st.write("• Expected returns appear muted, likely due to defensive or low-growth names.")

    # Volatility commentary
    if vol > 0.25:
        st.write("• Volatility is high, indicating exposure to high-beta or momentum stocks.")
    elif vol > 0.15:
        st.write("• Volatility is moderate, typical for diversified portfolios.")
    else:
        st.write("• Volatility is low, suggesting defensive or mega-cap concentration.")

    # Sharpe commentary
    if sharpe_m > 1.0:
        st.write("• Strong Sharpe ratio indicates efficient risk-adjusted performance.")
    elif sharpe_m > 0.5:
        st.write("• Sharpe ratio is acceptable but could be improved.")
    else:
        st.write("• Weak Sharpe ratio suggests the portfolio may not be compensated for its risk.")

    # Drawdown commentary
    if isinstance(max_dd_m, (int, float, np.floating)):
        if max_dd_m < -0.40:
            st.write("• Deep drawdowns indicate vulnerability during market stress.")
        elif max_dd_m < -0.20:
            st.write("• Drawdowns are moderate and typical for equities.")
        else:
            st.write("• Shallow drawdowns indicate strong downside resilience.")

    # Sector commentary
    if sector_text:
        st.markdown("### Sector Exposure")
        st.write(f"**Sector Weights:** {sector_text}")

    # Monte Carlo commentary
    if mc_comment:
        st.markdown("### Monte Carlo Outlook")
        st.write(mc_comment)

    # -----------------------------
    # AI Buy/Hold/Sell Signals
    # -----------------------------
    st.markdown("### AI Buy / Hold / Sell Signals")

    signals = []
    for _, row in fund_df.iterrows():
        t = row["Ticker"]
        pe = row["PE"]
        pb = row["PB"]
        dy = row["DividendYield"]
        beta = row["Beta"]
        momentum_val = momentum_model.get(t, 0)

        score = 0
        conviction = 0

        if pe and pe < 40:
            score += 1
            conviction += 20
        if pb and pb < 8:
            score += 1
            conviction += 15
        if dy and dy > 0.005:
            score += 1
            conviction += 15
        if beta and beta < 1.3:
            score += 1
            conviction += 20
        if momentum_val and momentum_val > 0:
            score += 1
            conviction += 30

        rating = "Buy" if score >= 4 else "Hold" if score >= 2 else "Sell"
        conviction = min(100, max(0, conviction))

        signals.append({
            "Ticker": t,
            "Score": score,
            "Conviction": conviction,
            "Rating": rating
        })

    signals_df = pd.DataFrame(signals)
    st.dataframe(signals_df)

    # Portfolio-level signal
    st.markdown("### AI Portfolio-Level Signal")
    buy_count = len(signals_df[signals_df["Rating"] == "Buy"])
    sell_count = len(signals_df[signals_df["Rating"] == "Sell"])

    if buy_count > sell_count:
        st.success("**AI Portfolio Signal: BUY** — Strong aggregate fundamentals.")
    elif sell_count >= buy_count + 2:
        st.error("**AI Portfolio Signal: SELL** — Broad fundamental weakness.")
    else:
        st.warning("**AI Portfolio Signal: HOLD** — Mixed signals across the portfolio.")

# ---------------------------------------------------------
# Tab 8 — Buy Analysis
# ---------------------------------------------------------
with tab8:
    st.subheader("Buy Analysis")

    # Guard clause
    if "portfolio_weights" not in st.session_state:
        st.info("Run the Optimizer tab first to calculate portfolio weights.")
        st.stop()

    portfolio_weights = st.session_state["portfolio_weights"]

    # Load fundamentals + prices
    if "fundamentals" not in st.session_state:
        st.error("Fundamentals missing — cannot run buy analysis.")
        st.stop()

    if "prices" not in st.session_state:
        st.error("Price data missing — cannot run buy analysis.")
        st.stop()

    fundamentals = st.session_state["fundamentals"]
    prices = st.session_state["prices"]

    # Run buy analysis
    buy_results = run_buy_analysis(tickers, fundamentals, prices)

    # Clean numeric columns
    numeric_cols = ["PE", "PB", "DividendYield", "Momentum", "Risk", "Score"]
    for col in numeric_cols:
        if col in buy_results.columns:
            buy_results[col] = buy_results[col].apply(safe_val)

    # Rating colors
    def rating_color(val):
        return {"Buy": "🟢 Buy", "Hold": "🟡 Hold", "Sell": "🔴 Sell"}[val]

    buy_results["RatingColored"] = buy_results["Rating"].apply(rating_color)

    st.dataframe(
        buy_results[
            ["Ticker", "Momentum", "Risk", "PE", "PB", "DividendYield", "Score", "RatingColored"]
        ]
    )

    # Commentary
    st.subheader("AI Buy Analysis Commentary")

    def generate_buy_commentary(df):
        if df.empty:
            return "No buy analysis available."
        lines = []
        best = df.sort_values("Score", ascending=False).iloc[0]
        lines.append(f"**{best['Ticker']}** leads with a score of {best['Score']}.")
        worst = df.sort_values("Score", ascending=True).iloc[0]
        lines.append(f"**{worst['Ticker']}** ranks weakest with a score of {worst['Score']}.")
        return "\n\n".join(lines)

    st.markdown(generate_buy_commentary(buy_results))

    # Radar chart
    st.subheader("Fundamentals Radar Chart")
    radar_cols = ["PE", "PB", "DividendYield", "Momentum", "Risk"]
    fig = go.Figure()
    for _, row in buy_results.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row[c] if row[c] is not None else 0 for c in radar_cols],
            theta=radar_cols,
            fill='toself',
            name=row["Ticker"]
        ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True, height=500)
    st.plotly_chart(fig, use_container_width=True, key="radar_chart")

    # Strengths & Weaknesses
    st.subheader("Top Strengths & Weaknesses")

    def strengths_weaknesses(row):
        strengths, weaknesses = [], []

        if row["Momentum"] and row["Momentum"] > 0:
            strengths.append("Positive momentum")
        else:
            weaknesses.append("Weak momentum")

        if row["Risk"] and row["Risk"] < 0.30:
            strengths.append("Low volatility")
        else:
            weaknesses.append("High volatility")

        if row["PE"] and row["PE"] > 40:
            weaknesses.append("Stretched PE ratio")
        elif row["PE"]:
            strengths.append("Reasonable PE ratio")

        if row["PB"] and row["PB"] > 8:
            weaknesses.append("Rich PB ratio")
        elif row["PB"]:
            strengths.append("Healthy PB ratio")

        if row["DividendYield"] and row["DividendYield"] > 0.01:
            strengths.append("Dividend support")
        else:
            weaknesses.append("Low or no dividend")

        return strengths, weaknesses

    for _, row in buy_results.iterrows():
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
# Tab 9 — Optimizer
# ---------------------------------------------------------
@st.cache_data(show_spinner=True)
def run_optimizer_cached(returns, cov):
    return run_optimizer(returns, cov)

with tab9:
    st.subheader("Optimizer")

    # Correct guard clause
    if "returns_df" not in st.session_state:
        st.info("Run Analysis first to load data.")
        st.stop()

    returns_df = st.session_state["returns_df"]

    # Run optimizer
    cov_matrix = returns_df.cov()
    opt_results = run_optimizer_cached(returns_df, cov_matrix)

    # Save results
    st.session_state["model"] = opt_results
    st.session_state["portfolio_weights"] = opt_results["max_sharpe"]["weights"]
    st.session_state["returns_df"] = returns_df

    st.success("Optimization complete!")
    
    # -----------------------------------------------------
    # EQUAL WEIGHT PORTFOLIO
    # -----------------------------------------------------
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

    # -----------------------------------------------------
    # MAXIMUM SHARPE PORTFOLIO
    # -----------------------------------------------------
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

    # -----------------------------------------------------
    # CORRELATION HEATMAP
    # -----------------------------------------------------
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
