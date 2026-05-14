import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import load_price_data
from utils.fundamentals_loader import load_fundamentals
from utils.optimizer_core import run_optimizer
from utils.buy_analysis import run_buy_analysis
from utils.analytics import run_monte_carlo_simulation


# ---------------------------------------------------------
# FUNCTION: LOAD TICKERS BASED ON UNIVERSE
# ---------------------------------------------------------
def get_available_tickers(universe):
    if universe == "S&P 500":
        df = pd.read_csv("data/sp500.csv")
        return df["Symbol"].tolist()

    elif universe == "Nasdaq 100":
        df = pd.read_csv("data/nasdaq100.csv")
        return df["Symbol"].tolist()

    elif universe == "Dow 30":
        df = pd.read_csv("data/dow30.csv")
        return df["Symbol"].tolist()

    elif universe == "Mega Caps":
        return ["AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META", "TSLA"]

    elif universe == "Custom":
        custom_input = st.sidebar.text_input(
            "Enter custom tickers (comma-separated):",
            placeholder="AAPL, MSFT, TSLA"
        )
        return [t.strip().upper() for t in custom_input.split(",") if t.strip()]

    return []

# ---------------------------------------------------------
# SIDEBAR – MAIN CONTROLS
# ---------------------------------------------------------
st.sidebar.header("Configuration")

# 1. Universe selector
universe = st.sidebar.selectbox(
    "Select a ticker universe:",
    ["S&P 500", "Nasdaq 100", "Dow 30", "Mega Caps", "Custom"]
)

# 2. Load tickers dynamically (includes custom input)
available_tickers = get_available_tickers(universe)

# 3. Ticker multiselect
tickers = st.sidebar.multiselect(
    "Select tickers:",
    options=available_tickers,
    default=available_tickers[:5] if len(available_tickers) > 5 else available_tickers
)

# 4. Date range
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))

# 5. Monte Carlo settings
mc_sims = st.sidebar.number_input(
    "Monte Carlo Simulations",
    min_value=100, max_value=50000, value=5000
)

mc_horizon = st.sidebar.number_input(
    "Monte Carlo Horizon (days)",
    min_value=30, max_value=252*5, value=252
)

# 6. Run button
run_button = st.sidebar.button("Run Analysis")


# ---------------------------------------------------------
# MAIN PAGE TITLE
# ---------------------------------------------------------
st.title("Portfolio Optimizer Dashboard")

# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------
def safe_val(x):
    try:
        if x in [None, "None", "nan", "NaN", "", "-", "N/A"]:
            return np.nan
        return float(x)
    except Exception:
        return np.nan

# ---------------------------------------------------------
# CORE DATA LOAD (using sidebar tickers)
# ---------------------------------------------------------
prices = load_price_data(tickers, start_date, end_date)

if prices is None or prices.empty:
    st.error("Price data could not be loaded.")
    st.stop()

close = prices.copy()
returns_df = close.pct_change().dropna()

valid_tickers = list(returns_df.columns)
if len(valid_tickers) == 0:
    st.error("No valid tickers after cleaning returns.")
    st.stop()

# SPY for beta
spy_data = load_price_data(["SPY"], start_date, end_date)
if spy_data is not None and not spy_data.empty:
    spy_returns = spy_data["SPY"].pct_change().dropna()
else:
    spy_returns = None

# ---------------------------------------------------------
# LOAD FUNDAMENTALS FOR EACH VALID TICKER
# ---------------------------------------------------------

# Normalize tickers
valid_tickers = [t.strip().upper() for t in valid_tickers]

fundamentals = []

for t in valid_tickers:
    f = load_fundamentals([t])   # returns DataFrame indexed by ticker

    # --- SAFETY: force DataFrame format ---
    if isinstance(f, dict):
        f = pd.DataFrame([f], index=[t])

    elif isinstance(f, pd.DataFrame):
        if t not in f.index:
            f.index = [t]

    # Now safe
    row = f.loc[t]

    fundamentals.append({
    "Ticker": t,
    "PE": float(row.get("PE")) if pd.notna(row.get("PE")) else None,
    "PB": float(row.get("PB")) if pd.notna(row.get("PB")) else None,
    "EPS": float(row.get("EPS")) if pd.notna(row.get("EPS")) else None,
    "ROE": float(row.get("ROE")) if pd.notna(row.get("ROE")) else None,
    "DividendYield": float(row.get("DividendYield")) if pd.notna(row.get("DividendYield")) else None,
    "DebtToEquity": float(row.get("DebtToEquity")) if pd.notna(row.get("DebtToEquity")) else None,
    "Beta": float(row.get("Beta")) if pd.notna(row.get("Beta")) else None,
    "MarketCap": float(row.get("MarketCap")) if pd.notna(row.get("MarketCap")) else None,
    "Sector": str(row.get("Sector")) if row.get("Sector") else "Unknown"
})

# Convert to DataFrame
fundamentals_df = pd.DataFrame(fundamentals).set_index("Ticker")

# Clean sector labels
fundamentals_df["Sector"] = fundamentals_df["Sector"].fillna("Unknown").replace("", "Unknown")

st.write(fundamentals)

st.write("Shape:", fundamentals_df.shape)
st.write("Columns:", fundamentals_df.columns.tolist())
st.dataframe(fundamentals_df)

st.write(fundamentals)  # show raw list

st.write("Shape:", fundamentals_df.shape)
st.write("Columns:", fundamentals)

# SAFE CHECK
if fundamentals_df.empty:
    st.error("No fundamentals data available for the selected tickers.")
    st.stop()

# STEP 4 — Fix missing sectors
if "Sector" not in fundamentals_df.columns:
    fundamentals_df["Sector"] = "Unknown"

fundamentals_df["Sector"] = fundamentals_df["Sector"].fillna("Unknown")

if not fundamentals or len(fundamentals) == 0:
    st.error("No fundamentals data returned for the selected tickers.")
    st.stop()

# Convert dict → DataFrame
fundamentals_df = pd.DataFrame(fundamentals)
    
# ---------------------------------------------------------
# FIX: Ensure weights_dict contains ALL valid tickers
# ---------------------------------------------------------
if "weights" not in st.session_state:
    st.session_state["weights"] = {}

weights_dict = st.session_state["weights"]

# Add missing tickers with equal weights
for t in valid_tickers:
    if t not in weights_dict:
        weights_dict[t] = 1 / len(valid_tickers)

# Remove tickers that no longer exist
weights_dict = {t: weights_dict[t] for t in valid_tickers}

# Normalize weights
total_w = sum(weights_dict.values())
if total_w > 0:
    weights_dict = {t: w / total_w for t, w in weights_dict.items()}

st.session_state["weights"] = weights_dict
global_weights = np.array([weights_dict[t] for t in valid_tickers])
st.session_state["global_weights"] = global_weights

# Global portfolio metrics
portfolio_returns = returns_df[valid_tickers].dot(global_weights)
annual_return = portfolio_returns.mean() * 252
annual_volatility = portfolio_returns.std() * np.sqrt(252)
sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else np.nan

drawdown = (1 + portfolio_returns).cumprod()
drawdown = (drawdown / drawdown.cummax()) - 1
max_drawdown = drawdown.min() if not drawdown.empty else np.nan

corr_matrix = returns_df.corr()
avg_corr = corr_matrix.where(~np.eye(corr_matrix.shape[0], dtype=bool)).mean().mean()
diversification_score = max(0, min(10, (1 - avg_corr) * 10))

if spy_returns is not None and not spy_returns.empty:
    aligned = pd.concat([portfolio_returns, spy_returns], axis=1).dropna()
    if not aligned.empty:
        pr = aligned.iloc[:, 0]
        mr = aligned.iloc[:, 1]
        covariance = pr.cov(mr)
        market_variance = mr.var()
        portfolio_beta = covariance / market_variance if market_variance > 0 else np.nan
    else:
        portfolio_beta = np.nan
else:
    portfolio_beta = np.nan

# ---------------------------------------------------------
# LOAD FUNDAMENTALS ONCE (FOR ALL VALID TICKERS)
# ---------------------------------------------------------
fundamentals = []
for t in valid_tickers:
    f = load_fundamentals(t)
    fundamentals.append({
        "Ticker": t,
        "PE": f["PE"],
        "PB": f["PB"],
        "DividendYield": f["DividendYield"],
        "Beta": f["Beta"],
        "MarketCap": f["MarketCap"],
        "Sector": f.get("Sector", "Unknown")
    })

fundamentals_df = pd.DataFrame(fundamentals).set_index("Ticker")

# ---------------------------------------------------------
# SAFE SECTOR WEIGHTS (REPLACE OLD BLOCK)
# ---------------------------------------------------------

# Clean sector labels
fundamentals_df["Sector"] = (
    fundamentals_df["Sector"]
    .astype(str)              # <-- MUST COME FIRST
    .replace("None", "Unknown")
    .replace("nan", "Unknown")
    .replace("", "Unknown")
    .fillna("Unknown")
)


# Ensure weights align with fundamentals_df
weights_series = pd.Series(global_weights, index=valid_tickers)
weights_series = weights_series.reindex(fundamentals_df.index).fillna(0)

# Safe groupby
sector_weights = (
    weights_series
    .groupby(fundamentals_df["Sector"])
    .sum()
    .sort_values(ascending=False)
)

# ---------------------------------------------------------
# RUN OPTIMIZER / BUY / MC
# ---------------------------------------------------------
optimizer_results = None
buy_results = None
mc_results = None

if run_button:
    cov_matrix = returns_df[valid_tickers].cov()
    optimizer_results = run_optimizer(returns_df[valid_tickers], cov_matrix)
    buy_results = run_buy_analysis(valid_tickers, fundamentals_df, close)
    mc_results = run_monte_carlo_simulation(returns_df[valid_tickers], mc_sims, mc_horizon)

    st.session_state["model"] = {
        "performance": {
            "expected_return": annual_return,
            "volatility": annual_volatility,
            "sharpe": sharpe_ratio,
        },
        "fundamentals": fundamentals,
        "tickers": valid_tickers,
        "drawdown": drawdown.to_frame("Drawdown"),
        "sector_weights": sector_weights,
        "monte_carlo": mc_results,
        "optimizer": optimizer_results,
        "buy_analysis": buy_results,
        "momentum": {},
    }
# ---------------------------------------------------------
# Tabs
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
# Tab 1 
# ---------------------------------------------------------
# ---------------------------------------------------------
# TAB 2 — PERFORMANCE
# ---------------------------------------------------------
with tab2:
    st.subheader("Performance Metrics")

    ret = pd.Series(portfolio_returns, name="Portfolio Return").dropna()

    # Cumulative return
    cum_ret = (1 + ret).cumprod()

    # Rolling metrics
    rolling_vol = ret.rolling(30).std() * np.sqrt(252)
    rolling_sharpe = (ret.rolling(30).mean() * 252) / rolling_vol

    # Drawdown
    cum_max = cum_ret.cummax()
    dd = (cum_ret - cum_max) / cum_max

    # Performance stats
    mu = ret.mean() * 252
    vol = ret.std() * np.sqrt(252)
    sharpe_local = mu / vol if vol > 0 else 0
    sortino = (ret.mean() * 252) / (ret[ret < 0].std() * np.sqrt(252)) if ret[ret < 0].std() > 0 else 0
    calmar = mu / abs(dd.min()) if dd.min() != 0 else 0
    max_dd = dd.min()

    # Display metrics
    colA, colB, colC, colD, colE, colF = st.columns(6)
    colA.metric("Expected Return", f"{mu:.2%}")
    colB.metric("Volatility", f"{vol:.2%}")
    colC.metric("Sharpe Ratio", f"{sharpe_local:.2f}")
    colD.metric("Sortino Ratio", f"{sortino:.2f}")
    colE.metric("Calmar Ratio", f"{calmar:.2f}")
    colF.metric("Max Drawdown", f"{max_dd:.2%}")

    # Charts
    st.markdown("### Cumulative Return")
    st.line_chart(cum_ret)

    st.markdown("### Rolling Volatility (30-day)")
    st.line_chart(rolling_vol)

    st.markdown("### Rolling Sharpe Ratio (30-day)")
    st.line_chart(rolling_sharpe)

    st.markdown("### Drawdown")
    st.area_chart(dd)

    st.markdown("### Distribution of Daily Returns")
    fig, ax = plt.subplots()
    ax.hist(ret, bins=40, alpha=0.7)
    ax.set_title("Histogram of Daily Returns")
    st.pyplot(fig)

# ---------------------------------------------------------
# TAB 3 — RISK & DRAWDOWN ANALYSIS
# ---------------------------------------------------------
with tab3:
    st.subheader("Risk & Drawdown Analysis")

    ret = pd.Series(portfolio_returns, name="Portfolio Return").dropna()
    if ret.empty:
        st.warning("Not enough return data to compute risk metrics.")
        st.stop()

    cum_ret = (1 + ret).cumprod()
    running_max = cum_ret.cummax()
    dd = (cum_ret - running_max) / running_max
    max_dd = dd.min() if not dd.empty else np.nan

    rolling_vol = ret.rolling(30).std() * np.sqrt(252)
    last_vol = rolling_vol.dropna().iloc[-1] if not rolling_vol.dropna().empty else np.nan

    beta_value = portfolio_beta if not np.isnan(portfolio_beta) else 0.0

    var_95 = np.percentile(ret, 5) if len(ret) > 0 else np.nan
    if np.isnan(var_95):
        cvar_95 = np.nan
    else:
        tail = ret[ret <= var_95]
        cvar_95 = tail.mean() if len(tail) > 0 else np.nan

    colA, colB, colC, colD = st.columns(4)
    colA.metric("Max Drawdown", f"{max_dd:.2%}" if not np.isnan(max_dd) else "N/A")
    colB.metric("Rolling Vol (30d)", f"{last_vol:.2%}" if not np.isnan(last_vol) else "N/A")
    colC.metric("Beta vs SPY", f"{beta_value:.2f}")
    colD.metric("CVaR (95%)", f"{cvar_95:.2%}" if not np.isnan(cvar_95) else "N/A")

    st.markdown("### Drawdown")
    st.area_chart(dd)

    st.markdown("### Rolling Volatility (30-day)")
    st.line_chart(rolling_vol)

    st.markdown("### Distribution of Daily Returns (for VaR)")
    fig, ax = plt.subplots()
    ax.hist(ret, bins=40, alpha=0.7)
    if not np.isnan(var_95):
        ax.axvline(var_95, color="red", linestyle="--", label=f"VaR 95%: {var_95:.2%}")
    ax.set_title("Return Distribution with VaR")
    ax.legend()
    st.pyplot(fig)

    # Risk contribution
    st.markdown("### Risk Contribution Breakdown")
    if len(valid_tickers) == 0:
        st.info("No valid tickers available.")
    else:
        w = st.session_state.get("global_weights", np.array([1/len(valid_tickers)]*len(valid_tickers)))
        cov = returns_df[valid_tickers].cov().values
        port_var = float(w @ cov @ w.T)
        if port_var <= 0 or np.isnan(port_var):
            st.info("Unable to compute risk contribution (invalid variance).")
        else:
            mrc = cov @ w
            rc = w * mrc
            rc_pct = rc / port_var
            fig2, ax2 = plt.subplots()
            ax2.pie(rc_pct, labels=valid_tickers, autopct="%1.1f%%", startangle=90)
            ax2.axis("equal")
            st.pyplot(fig2)

# ---------------------------------------------------------
# TAB 4 — SECTOR EXPOSURE (FINAL FIXED)
# ---------------------------------------------------------
with tab4:
    st.subheader("Sector Exposure")

    # fundamentals_df must already exist
    if fundamentals_df is None or fundamentals_df.empty:
        st.info("Sector data unavailable. Run analysis first.")
        st.stop()

    # Ensure Sector column exists
    if "Sector" not in fundamentals_df.columns:
        fundamentals_df["Sector"] = "Unknown"

    # Clean sector values
    fundamentals_df["Sector"] = fundamentals_df["Sector"].fillna("Unknown")

    # Use sidebar tickers (NOT valid_tickers)
    fdf = fundamentals_df.reindex(tickers)

    # -----------------------------
    # WEIGHTING LOGIC (unchanged)
    # -----------------------------
    if "global_weights" in st.session_state:
        w = st.session_state["global_weights"]
        if len(w) != len(tickers):
            w = np.array([1/len(tickers)] * len(tickers))
    else:
        w = np.array([1/len(tickers)] * len(tickers))

    w_series = pd.Series(w, index=tickers)

    # -----------------------------
    # GROUP BY SECTOR (corrected)
    # -----------------------------
    sector_map = fdf["Sector"].to_dict()

    # Group weights by sector
    sector_weights = (
        w_series.groupby(sector_map)
        .sum()
        .sort_values(ascending=False)
    )

    # -----------------------------
    # DISPLAY TABLE
    # -----------------------------
    st.markdown("### Sector Allocation Breakdown")
    st.dataframe(
        sector_weights.to_frame("Weight")
        .style.format({"Weight": "{:.2%}"})
    )

    # -----------------------------
    # DISPLAY CHART
    # -----------------------------
    st.markdown("### Sector Chart")
    fig = go.Figure(go.Bar(
        x=sector_weights.index,
        y=sector_weights.values,
        marker_color="steelblue"
    ))
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# TAB 5 — FUNDAMENTALS
# ---------------------------------------------------------
with tab5:
    st.subheader("Fundamentals")

    # Use the sidebar tickers, NOT valid_tickers
    fundamentals_df = (
        pd.DataFrame(fundamentals)
        .set_index("Ticker")
        .reindex(tickers)
    )

    st.dataframe(fundamentals_df)
    st.subheader("Fundamentals")

    num_cols = ["ROE", "EPS", "PE", "PB", "DividendYield"]
    for c in num_cols:
        if c not in fundamentals_df.columns:
            fundamentals_df[c] = np.nan

    display_df = fundamentals_df.copy()
    display_df[num_cols] = display_df[num_cols].fillna(0)

    # Do NOT drop Sector unless it exists
    fundamentals_display = display_df.drop(columns=["Sector"], errors="ignore")
    st.dataframe(fundamentals_display)

    st.subheader("Fundamentals Ranking")

def score_fundamentals(row):
    score = 0

    pe = row.get("PE")
    pb = row.get("PB")
    eps = row.get("EPS")
    roe = row.get("ROE")
    dy = row.get("DividendYield")
    dte = row.get("DebtToEquity")

    # PE (lower is better)
    if pd.notna(pe):
        if pe < 15: score += 2
        elif pe < 25: score += 1

    # PB (lower is better)
    if pd.notna(pb):
        if pb < 3: score += 2
        elif pb < 6: score += 1

    # EPS (higher is better)
    if pd.notna(eps):
        if eps > 5: score += 2
        elif eps > 1: score += 1

    # ROE (higher is better)
    if pd.notna(roe):
        if roe > 0.25: score += 2
        elif roe > 0.10: score += 1

    # Dividend Yield (higher is better)
    if pd.notna(dy):
        if dy > 2: score += 2
        elif dy > 1: score += 1

    # Debt-to-Equity (lower is better)
    if pd.notna(dte):
        if dte < 40: score += 2
        elif dte < 80: score += 1

    return score

    fundamentals_df["score"] = fundamentals_df.apply(score_fundamentals, axis=1)
    ranked_df = fundamentals_df.sort_values("score", ascending=False)
    st.dataframe(ranked_df[["score"]])

    st.subheader("AI Fundamentals Commentary")

    def generate_fundamentals_commentary(rdf):
        if rdf.empty:
            return "No fundamentals available."

        lines = []
        tickers_sorted = rdf.index.tolist()
        best = tickers_sorted[0]
        worst = tickers_sorted[-1]

        best_row = rdf.loc[best]
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

        if len(tickers_sorted) > 2:
            middle = tickers_sorted[1:-1]
            for t in middle:
                row = rdf.loc[t]
                mid_reasons = []
                if pd.notna(row.get("ROE")) and row["ROE"] > 0:
                    mid_reasons.append("healthy ROE")
                if pd.notna(row.get("EPS")) and row["EPS"] > 0:
                    mid_reasons.append("stable earnings")
                if pd.notna(row.get("PE")) and row["PE"] < 40:
                    mid_reasons.append("fair valuation")
                reason_text = ", ".join(mid_reasons) if mid_reasons else "balanced fundamentals"
                lines.append(f"**{t}** shows {reason_text}, placing it in the middle of the group.")

        worst_row = rdf.loc[worst]
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

    st.subheader("Commentary")
    commentary = [f"- **{ticker}**: score {row['score']:.1f}" for ticker, row in ranked_df.iterrows()]
    st.markdown("\n".join(commentary))

# ---------------------------------------------------------
# TAB 6 — WEIGHTS
# ---------------------------------------------------------
with tab6:
    st.header("Portfolio Weights")

    if len(valid_tickers) == 0:
        st.warning("No valid tickers available to assign weights.")
        st.stop()

    if "weights" not in st.session_state:
        st.session_state["weights"] = {t: 1/len(valid_tickers) for t in valid_tickers}

    weights_dict = st.session_state["weights"]

    st.subheader("Adjust Weights")
    for t in valid_tickers:
        weights_dict[t] = st.slider(
            f"{t} Weight",
            min_value=0.0,
            max_value=1.0,
            value=float(weights_dict[t]),
            key=f"weight_{t}"
        )

    total = sum(weights_dict.values())
    if total > 0:
        weights_dict = {t: w/total for t, w in weights_dict.items()}
    st.session_state["weights"] = weights_dict

    global_weights = np.array([weights_dict[t] for t in valid_tickers])
    st.session_state["global_weights"] = global_weights

    st.subheader("Final Normalized Weights")
    weights_df = pd.DataFrame({
        "Ticker": valid_tickers,
        "Weight": [weights_dict[t] for t in valid_tickers]
    })
    st.dataframe(weights_df, use_container_width=True)
# ---------------------------------------------------------
# TAB 7 — AI COMMENTARY + SIGNALS (FINAL FIXED)
# ---------------------------------------------------------
with tab7:
    st.subheader("AI Portfolio Commentary")

    if fundamentals_df is None or fundamentals_df.empty:
        st.warning("Not enough data to generate commentary.")
        st.stop()

    # -----------------------------
    # PORTFOLIO WEIGHTS
    # -----------------------------
    if "global_weights" in st.session_state:
        w = st.session_state["global_weights"]
        if len(w) != len(tickers):
            w = np.array([1/len(tickers)] * len(tickers))
    else:
        w = np.array([1/len(tickers)] * len(tickers))

    w_series = pd.Series(w, index=tickers)

    # Sector weights
    sector_map = fundamentals_df["Sector"].fillna("Unknown").to_dict()
    sector_weights_model = w_series.groupby(sector_map).sum().sort_values(ascending=False)

    # -----------------------------
    # OPTIMIZER RESULTS
    # -----------------------------
    optimizer = st.session_state.get("model", {}).get("optimizer")

    if optimizer is None:
        st.warning("Optimizer results unavailable.")
        st.stop()

    perf = optimizer["max_sharpe"]
    er = perf["expected_return"]
    vol = perf["volatility"]
    sharpe_m = perf["sharpe"]

    # -----------------------------
    # DRAWDOWN
    # -----------------------------
    drawdown_model = st.session_state.get("drawdown")
    if isinstance(drawdown_model, pd.DataFrame) and not drawdown_model.empty:
        max_dd_m = float(drawdown_model["Drawdown"].min())
    else:
        max_dd_m = None
    max_dd_text = f"{max_dd_m:.2%}" if isinstance(max_dd_m, (int, float, np.floating)) else "N/A"

    # -----------------------------
    # MONTE CARLO
    # -----------------------------
    mc_model = st.session_state.get("monte_carlo")
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
    # PORTFOLIO OVERVIEW
    # -----------------------------
    st.markdown("### Portfolio Overview")
    st.write(
        f"""
**Portfolio Grade:** {"A" if sharpe_m > 1.2 else "B" if sharpe_m > 0.8 else "C" if sharpe_m > 0.3 else "D"}  
**Risk Bucket:** {"Low Risk" if vol < 0.12 else "Moderate Risk" if vol < 0.20 else "High Risk"}  
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
    st.markdown("### Sector Exposure")
    sector_text = ", ".join([f"{s}: {w:.1%}" for s, w in sector_weights_model.items()])
    st.write(f"**Sector Weights:** {sector_text}")

    # Monte Carlo commentary
    if mc_comment:
        st.markdown("### Monte Carlo Outlook")
        st.write(mc_comment)

    # -----------------------------
    # FUNDAMENTALS SUMMARY
    # -----------------------------
    fund_df = fundamentals_df[["PE", "PB", "DividendYield", "Beta", "MarketCap", "Sector"]].copy()
    fund_df["Ticker"] = fund_df.index
    st.markdown("### Fundamentals Summary")
    st.dataframe(fund_df)

    # -----------------------------
    # AI BUY / HOLD / SELL SIGNALS
    # -----------------------------
    st.markdown("### AI Buy / Hold / Sell Signals")

    momentum_model = st.session_state.get("momentum", {})

    signals = []
    for t in tickers:
        row = fund_df.loc[t]

        pe = row["PE"]
        pb = row["PB"]
        dy = row["DividendYield"]
        beta = row["Beta"]
        momentum_val = momentum_model.get(t, 0)

        score = 0
        conviction = 0

        if pe is not None and not pd.isna(pe) and 0 < pe < 40:
            score += 1; conviction += 20
        if pb is not None and not pd.isna(pb) and 0 < pb < 8:
            score += 1; conviction += 15
        if dy is not None and not pd.isna(dy) and dy > 0.005:
            score += 1; conviction += 15
        if beta is not None and not pd.isna(beta) and beta < 1.3:
            score += 1; conviction += 20
        if momentum_val is not None and momentum_val > 0:
            score += 1; conviction += 30

        rating = "Buy" if score >= 4 else "Hold" if score >= 2 else "Sell"
        conviction = min(100, max(0, conviction))

        signals.append({
            "Ticker": t,
            "PE": pe,
            "PB": pb,
            "DividendYield": dy,
            "Beta": beta,
            "Momentum": momentum_val,
            "Score": score,
            "Conviction": conviction,
            "Rating": rating
        })

    signals_df = pd.DataFrame(signals)
    st.dataframe(signals_df)

    # -----------------------------
    # SIGNAL SUMMARY
    # -----------------------------
    st.markdown("### AI Signal Summary")
    buys = signals_df[signals_df["Rating"] == "Buy"]["Ticker"].tolist()
    holds = signals_df[signals_df["Rating"] == "Hold"]["Ticker"].tolist()
    sells = signals_df[signals_df["Rating"] == "Sell"]["Ticker"].tolist()

    if buys:
        st.write(f"• **Buy signals:** {', '.join(buys)} show strong valuation and risk-adjusted characteristics.")
    if holds:
        st.write(f"• **Hold signals:** {', '.join(holds)} appear fairly valued with balanced fundamentals.")
    if sells:
        st.write(f"• **Sell signals:** {', '.join(sells)} exhibit weaker fundamentals or elevated risk.")

    # -----------------------------
    # PORTFOLIO-LEVEL SIGNAL
    # -----------------------------
    st.markdown("### AI Portfolio-Level Signal")
    buy_count = len(buys)
    sell_count = len(sells)

    if buy_count > sell_count:
        st.success("**AI Portfolio Signal: BUY** — The portfolio shows strong aggregate fundamentals.")
    elif sell_count >= buy_count + 2:
        st.error("**AI Portfolio Signal: SELL** — The portfolio shows broad fundamental weakness.")
    else:
        st.warning("**AI Portfolio Signal: HOLD** — Mixed signals across the portfolio.")

# ---------------------------------------------------------
# TAB 8 — BUY ANALYSIS
# ---------------------------------------------------------
with tab8:
    st.subheader("Buy Analysis")

    if not run_button:
        st.info("Run Analysis to generate buy analysis.")
        st.stop()

    # Use sidebar tickers, NOT valid_tickers
    buy_results = run_buy_analysis(tickers, fundamentals_df, close)
    #buy_results = run_buy_analysis(tickers, fundamentals, close)

    numeric_cols = ["PE", "PB", "DividendYield", "Momentum", "Risk", "Score"]
    for col in numeric_cols:
        if col in buy_results.columns:
            buy_results[col] = buy_results[col].apply(safe_val)

    def rating_color(val):
        return {"Buy": "🟢 Buy", "Hold": "🟡 Hold", "Sell": "🔴 Sell"}.get(val, "N/A")

    buy_results["RatingColored"] = buy_results["Rating"].apply(rating_color)

    st.dataframe(
        buy_results[
            ["Ticker", "Momentum", "Risk", "PE", "PB", "DividendYield", "Score", "RatingColored"]
        ],
        use_container_width=True
    )

    st.subheader("AI Buy Analysis Commentary")

    def generate_buy_commentary(df):
        if df.empty:
            return "No buy analysis available."
        lines = []
        best = df.sort_values("Score", ascending=False).iloc[0]
        lines.append(f"**{best['Ticker']}** leads with a score of {best['Score']:.2f}.")
        worst = df.sort_values("Score", ascending=True).iloc[0]
        lines.append(f"**{worst['Ticker']}** ranks weakest with a score of {worst['Score']:.2f}.")
        return "\n\n".join(lines)

    st.markdown(generate_buy_commentary(buy_results))

    st.subheader("Fundamentals Radar Chart")
    radar_cols = ["PE", "PB", "DividendYield", "Momentum", "Risk"]
    fig = go.Figure()
    for _, row in buy_results.iterrows():
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
        if pd.notna(row["Momentum"]) and row["Momentum"] > 0:
            strengths.append("Positive momentum")
        else:
            weaknesses.append("Weak momentum")
        if pd.notna(row["Risk"]) and row["Risk"] < 0.30:
            strengths.append("Low volatility")
        else:
            weaknesses.append("High volatility")
        if pd.notna(row["PE"]) and row["PE"] > 40:
            weaknesses.append("Stretched PE ratio")
        elif pd.notna(row["PE"]):
            strengths.append("Reasonable PE ratio")
        if pd.notna(row["PB"]) and row["PB"] > 8:
            weaknesses.append("Rich PB ratio")
        elif pd.notna(row["PB"]):
            strengths.append("Healthy PB ratio")
        if pd.notna(row["DividendYield"]) and row["DividendYield"] > 0.01:
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
# TAB 9 — OPTIMIZER
# ---------------------------------------------------------
@st.cache_data(show_spinner=True)
def run_optimizer_cached(returns, cov):
    return run_optimizer(returns, cov)

with tab9:
    st.subheader("Optimizer")

    if not run_button:
        st.info("Run Analysis to generate optimizer results.")
        st.stop()

    # Use sidebar tickers, NOT valid_tickers
    cov_matrix = returns_df[tickers].cov()
    opt_results = run_optimizer_cached(returns_df[tickers], cov_matrix)

    st.success("Optimization complete!")

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

    st.markdown("### Correlation Heatmap")
    corr = returns_df[tickers].corr()
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

    # Save optimizer results
    if "model" in st.session_state:
        st.session_state["model"]["optimizer"] = opt_results
    else:
        st.session_state["model"] = {"optimizer": opt_results}
