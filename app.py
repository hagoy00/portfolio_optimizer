import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import yfinance as yf

from datetime import datetime, timedelta

# === NEW UTILS (institutional‑grade) ===
from utils.optimizer_core import run_optimizer
from utils.buy_analysis import run_buy_analysis
from utils.analytics import run_monte_carlo_simulation
from utils.data_loader import load_price_data
from utils.fundamentals_loader import load_fundamentals

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(page_title="Portfolio Optimizer Dashboard", layout="wide")

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

run_button = st.sidebar.button("Run Analysis")
mc_sims = st.sidebar.slider("Monte Carlo Simulations", 200, 3000, 500)
mc_horizon = st.sidebar.slider("Monte Carlo Horizon (days)", 50, 500, 252)

# ---------------------------------------------------------
# LOAD PRICE DATA (NEW FLAT FORMAT)
# ---------------------------------------------------------
from utils.data_loader import load_price_data
from utils.fundamentals_loader import load_fundamentals
from utils.optimizer_core import run_optimizer
from utils.buy_analysis import run_buy_analysis
from utils.analytics import run_monte_carlo_simulation

# Load prices
prices = load_price_data(tickers, start_date, end_date)

if prices is None or prices.empty:
    st.error("Price data could not be loaded.")
    st.stop()

# Prices = Adjusted Close only
close = prices.copy()

# ---------------------------------------------------------
# Clean returns (user tickers only)
# ---------------------------------------------------------
returns_df = close.pct_change().dropna()

valid_tickers = list(returns_df.columns)

if len(valid_tickers) == 0:
    st.error("No valid tickers after cleaning returns.")
    st.stop()

# ---------------------------------------------------------
# Load SPY separately for beta (DO NOT add to tickers)
# ---------------------------------------------------------
spy_data = load_price_data(["SPY"], start_date, end_date)
if spy_data is not None and not spy_data.empty:
    spy_returns = spy_data["SPY"].pct_change().dropna()
else:
    spy_returns = None

# ---------------------------------------------------------
# Fundamentals Loader (NEW)
# ---------------------------------------------------------
fundamentals = load_fundamentals(valid_tickers)

# Sector map
sector_map = {t: fundamentals[t].get("Sector", "Unknown") for t in valid_tickers}

# ---------------------------------------------------------
# Global Metrics
# ---------------------------------------------------------
weights = np.array([1 / len(valid_tickers)] * len(valid_tickers))

portfolio_returns = returns_df.dot(weights)
annual_return = portfolio_returns.mean() * 252
annual_volatility = portfolio_returns.std() * np.sqrt(252)
sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else np.nan
max_drawdown = (portfolio_returns.cummax() - portfolio_returns).max()

corr_matrix = returns_df.corr()
avg_corr = corr_matrix.where(~np.eye(corr_matrix.shape[0], dtype=bool)).mean().mean()
diversification_score = max(0, min(10, (1 - avg_corr) * 10))

# Beta vs SPY
if spy_returns is not None:
    covariance = portfolio_returns.cov(spy_returns)
    market_variance = spy_returns.var()
    portfolio_beta = covariance / market_variance if market_variance > 0 else np.nan
else:
    portfolio_beta = np.nan

# ---------------------------------------------------------
# Optimizer + Buy Analysis (RUN ONLY WHEN BUTTON CLICKED)
# ---------------------------------------------------------
optimizer_results = None
buy_results = None
mc_results = None

if run_button:
    cov_matrix = returns_df.cov()

    optimizer_results = run_optimizer(returns_df, cov_matrix)
    buy_results = run_buy_analysis(valid_tickers, fundamentals, close)
    mc_results = run_monte_carlo_simulation(returns_df, mc_sims, mc_horizon)

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
        #st.metric("Volatility", f"{portfolio_volatility:.2%}")
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
    # RISK CONTRIBUTION PIE CHART — CRASH‑PROOF VERSION
    # ---------------------------------------------------------
    st.subheader("Risk Contribution Breakdown")

    if len(valid_tickers) == 0:
        st.info("No valid tickers available.")
    else:
        # Equal contribution for each ticker (always valid)
        n = len(valid_tickers)
        risk_contribution = np.array([1.0 / n] * n, dtype=float)

        # Safety: ensure non-negative and normalized
        risk_contribution = np.clip(risk_contribution, 0, None)
        total = risk_contribution.sum()
        if total == 0 or np.isnan(total):
            risk_contribution = np.array([1.0 / n] * n, dtype=float)
        else:
            risk_contribution = risk_contribution / total

        # Force labels to match wedge count
        valid_tickers = valid_tickers[:len(risk_contribution)]

        # Render pie chart
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(
            risk_contribution,
            labels=valid_tickers,
            autopct="%1.1f%%",
            startangle=90
        )
        ax.axis("equal")
        st.pyplot(fig)
# ---------------------------------------------------------
# Performance Tab
# ---------------------------------------------------------
with tab2:
    st.subheader("Performance Metrics")

    # Use aligned portfolio returns
    ret = pd.Series(portfolio_returns, name="Portfolio Return")

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
    hist_data = pd.Series(portfolio_returns).dropna()
    fig, ax = plt.subplots()
    ax.hist(hist_data, bins=40, alpha=0.7)
    ax.set_title("Histogram of Daily Returns")
    st.pyplot(fig)

# ---------------------------------------------------------
# TAB 3 — RISK & DRAWDOWN ANALYSIS (CRASH-PROOF VERSION)
# ---------------------------------------------------------
with tab3:
    st.subheader("Risk & Drawdown Analysis")

    # Use aligned portfolio returns
    ret = pd.Series(portfolio_returns, name="Portfolio Return")
    clean_ret = ret.dropna()

    # === Guard clause: no return data ===
    if clean_ret.empty:
        st.warning("Not enough return data to compute risk metrics.")

        colA, colB, colC, colD = st.columns(4)
        colA.metric("Max Drawdown", "N/A")
        colB.metric("Rolling Vol (30d)", "N/A")
        colC.metric("Beta vs SPY", f"{portfolio_beta:.2f}")
        colD.metric("CVaR (95%)", "N/A")

        st.markdown("### Drawdown")
        st.area_chart(pd.Series(dtype=float))

        st.markdown("### Rolling Volatility (30-day)")
        st.line_chart(pd.Series(dtype=float))

        st.markdown("### Distribution of Daily Returns (for VaR)")
        fig, ax = plt.subplots()
        ax.hist([], bins=40, alpha=0.7)
        st.pyplot(fig)

        st.markdown("### Risk Contribution Breakdown")
        st.info("No valid tickers available.")

        st.stop()

    # === Drawdown ===
    cum_ret = (1 + clean_ret).cumprod()
    running_max = cum_ret.cummax()
    drawdown = (cum_ret - running_max) / running_max
    max_dd = drawdown.min() if not drawdown.empty else np.nan

    # === Rolling Volatility ===
    rolling_vol = clean_ret.rolling(30).std() * np.sqrt(252)
    last_vol = rolling_vol.iloc[-1] if len(rolling_vol.dropna()) > 0 else np.nan

    # === Portfolio Beta (use global value) ===
    beta_value = portfolio_beta if not np.isnan(portfolio_beta) else 0.0

    # === Value at Risk (95%) ===
    var_95 = np.percentile(clean_ret, 5) if len(clean_ret) > 0 else np.nan

    # === Conditional VaR (CVaR) ===
    if np.isnan(var_95):
        cvar_95 = np.nan
    else:
        tail = clean_ret[clean_ret <= var_95]
        cvar_95 = tail.mean() if len(tail) > 0 else np.nan

    # === Display Metrics ===
    colA, colB, colC, colD = st.columns(4)
    colA.metric("Max Drawdown", f"{max_dd:.2%}" if not np.isnan(max_dd) else "N/A")
    colB.metric("Rolling Vol (30d)", f"{last_vol:.2%}" if not np.isnan(last_vol) else "N/A")
    colC.metric("Beta vs SPY", f"{beta_value:.2f}")
    colD.metric("CVaR (95%)", f"{cvar_95:.2%}" if not np.isnan(cvar_95) else "N/A")

    # === Drawdown Chart ===
    st.markdown("### Drawdown")
    st.area_chart(drawdown)

    # === Rolling Volatility Chart ===
    st.markdown("### Rolling Volatility (30-day)")
    st.line_chart(rolling_vol)

    # === VaR Distribution Chart ===
    st.markdown("### Distribution of Daily Returns (for VaR)")
    fig, ax = plt.subplots()
    ax.hist(clean_ret, bins=40, alpha=0.7)
    if not np.isnan(var_95):
        ax.axvline(var_95, color="red", linestyle="--", label=f"VaR 95%: {var_95:.2%}")
    ax.set_title("Return Distribution with VaR")
    ax.legend()
    st.pyplot(fig)

        # ---------------------------------------------------------
    # RISK CONTRIBUTION — TRUE RISK-BASED VERSION
    # ---------------------------------------------------------
    st.markdown("### Risk Contribution Breakdown")

    if len(valid_tickers) == 0:
        st.info("No valid tickers available.")
    else:
        # Use only the valid tickers in returns_df
        ret_mat = returns_df[valid_tickers]

        # Equal weights (or later: user-defined)
        w = np.array([1.0 / len(valid_tickers)] * len(valid_tickers))

        # Covariance matrix
        cov = ret_mat.cov().values

        # Portfolio variance
        port_var = float(w @ cov @ w.T)
        if port_var <= 0 or np.isnan(port_var):
            st.info("Unable to compute risk contribution (invalid variance).")
        else:
            # Marginal contribution to risk
            mrc = cov @ w  # shape (n,)

            # Risk contribution per asset
            rc = w * mrc   # absolute contribution
            rc_pct = rc / port_var  # percentage of total risk

            fig2, ax2 = plt.subplots()
            ax2.pie(
                rc_pct,
                labels=valid_tickers,
                autopct="%1.1f%%",
                startangle=90
            )
            ax2.axis("equal")
            st.pyplot(fig2)
# ---------------------------------------------------------
# Tab 4 Sector exposure
# ---------------------------------------------------------
with tab4:
    st.subheader("Sector Exposure")

    # Safety: ensure fundamentals exist
    if not fundamentals or len(fundamentals) == 0:
        st.info("Sector data unavailable. Run analysis first.")
        st.stop()

    # Build DataFrame from fundamentals
    fdf = pd.DataFrame(fundamentals).T

    # Ensure Sector column exists
    if "Sector" not in fdf.columns:
        fdf["Sector"] = "Unknown"

    # Restrict to valid tickers
    fdf = fdf.reindex(valid_tickers)

    # Sector map from fundamentals
    sector_map = fdf["Sector"].fillna("Unknown").to_dict()

    # Equal weights for now (optimizer can override later)
    w = np.array([1 / len(valid_tickers)] * len(valid_tickers))
    w_series = pd.Series(w, index=valid_tickers)

    # Compute sector weights
    sector_weights = w_series.groupby(sector_map).sum().sort_values(ascending=False)

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
# Tab 5 Fundamentals
# ---------------------------------------------------------
with tab5:
    st.subheader("Fundamentals")

    # Build DataFrame from fundamentals
    fundamentals_df = pd.DataFrame(fundamentals).T

    # Keep only valid tickers
    fundamentals_df = fundamentals_df.reindex(valid_tickers)

    # Numeric columns we care about
    num_cols = ["ROE", "EPS", "PE", "PB", "DividendYield"]
    for c in num_cols:
        if c not in fundamentals_df.columns:
            fundamentals_df[c] = np.nan

    # For display: fill NaN with 0, but keep original for scoring
    display_df = fundamentals_df.copy()
    display_df[num_cols] = display_df[num_cols].fillna(0)

    # Do NOT show Sector here
    fundamentals_display = display_df.drop(columns=["Sector"], errors="ignore")
    st.dataframe(fundamentals_display)

    st.subheader("Fundamentals Ranking")

    def score_fundamentals(row):
        score = 0.0

        roe = row.get("ROE")
        eps = row.get("EPS")
        pe = row.get("PE")
        pb = row.get("PB")
        dy = row.get("DividendYield")

        if pd.notna(roe):
            score += roe * 10
        if pd.notna(eps):
            score += eps
        if pd.notna(pe):
            score += max(0, 50 - pe)
        if pd.notna(pb):
            score += max(0, 20 - pb)
        if pd.notna(dy):
            score += dy * 100

        return score

    fundamentals_df["score"] = fundamentals_df.apply(score_fundamentals, axis=1)
    ranked_df = fundamentals_df.sort_values("score", ascending=False)
    st.dataframe(ranked_df[["score"]])

    st.subheader("AI Fundamentals Commentary")

    def generate_fundamentals_commentary(ranked_df):
        if ranked_df.empty:
            return "No fundamentals available."

        lines = []
        tickers_sorted = ranked_df.index.tolist()

        best = tickers_sorted[0]
        worst = tickers_sorted[-1]

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

        if len(tickers_sorted) > 2:
            middle = tickers_sorted[1:-1]
            for t in middle:
                row = ranked_df.loc[t]
                mid_reasons = []
                if pd.notna(row.get("ROE")) and row["ROE"] > 0:
                    mid_reasons.append("healthy ROE")
                if pd.notna(row.get("EPS")) and row["EPS"] > 0:
                    mid_reasons.append("stable earnings")
                if pd.notna(row.get("PE")) and row["PE"] < 40:
                    mid_reasons.append("fair valuation")
                reason_text = ", ".join(mid_reasons) if mid_reasons else "balanced fundamentals"
                lines.append(f"**{t}** shows {reason_text}, placing it in the middle of the group.")

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

    st.subheader("Commentary")
    commentary = [f"- **{ticker}**: score {row['score']:.1f}" for ticker, row in ranked_df.iterrows()]
    st.markdown("\n".join(commentary))
# ---------------------------------------------------------
# Weights
# ---------------------------------------------------------
with tab6:
    st.header("Portfolio Weights")

    if len(valid_tickers) == 0:
        st.warning("No valid tickers available to assign weights.")
        st.stop()

    # ---------------------------------------------------------
    # DEFAULT WEIGHTS (equal weight for all valid tickers)
    # ---------------------------------------------------------
    weights_dict = {t: 1/len(valid_tickers) for t in valid_tickers}

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
    # NORMALIZE WEIGHTS (so they sum to 1)
    # ---------------------------------------------------------
    total = sum(weights_dict.values())
    if total > 0:
        weights_dict = {t: weight/total for t, weight in weights_dict.items()}

    # ---------------------------------------------------------
    # CONVERT TO NUMPY ARRAY (for portfolio math)
    # ---------------------------------------------------------
    w = np.array([weights_dict[t] for t in valid_tickers])

    # ---------------------------------------------------------
    # DISPLAY WEIGHTS TABLE
    # ---------------------------------------------------------
    st.subheader("Final Normalized Weights")
    weights_df = pd.DataFrame({
        "Ticker": valid_tickers,
        "Weight": [weights_dict[t] for t in valid_tickers]
    })

    st.dataframe(weights_df, use_container_width=True
# ---------------------------------------------------------
# AI Commentary + Signals
# ---------------------------------------------------------
with tab7:
    st.subheader("AI Portfolio Commentary")

    # Pull model from session_state (built after run_button)
    model = st.session_state.get("model")
    if not model:
        st.info("Run the analysis first to generate optimizer results.")
        st.stop()

    # Unpack model components
    perf = model.get("performance", {})
    fundamentals_model = model.get("fundamentals", {})
    tickers_model = model.get("tickers", valid_tickers)
    drawdown_model = model.get("drawdown")
    sector_weights_model = model.get("sector_weights", {})
    mc_model = model.get("monte_carlo")
    momentum_model = model.get("momentum", {})

    # Guard: need performance metrics
    if not perf or perf.get("expected_return") is None:
        st.warning("Not enough data to generate commentary.")
        st.stop()

    er = perf["expected_return"]
    vol = perf["volatility"]
    sharpe_m = perf["sharpe"]

    # Max drawdown from model
    if isinstance(drawdown_model, pd.DataFrame) and not drawdown_model.empty:
        max_dd_m = float(drawdown_model["Drawdown"].min())
    else:
        max_dd_m = None
    max_dd_text = f"{max_dd_m:.2%}" if isinstance(max_dd_m, (int, float, np.floating)) else "N/A"

    # Sector weights text
    sector_text = ""
    if isinstance(sector_weights_model, dict) and sector_weights_model:
        sector_text = ", ".join([f"{s}: {w:.1%}" for s, w in sector_weights_model.items()])

    # Fundamentals summary table
    fund_summary = []
    for t in tickers_model:
        f = fundamentals_model.get(t, {})
        fund_summary.append({
            "Ticker": t,
            "PE": f.get("PE"),
            "PB": f.get("PB"),
            "Dividend Yield": f.get("DividendYield"),
            "Beta": f.get("Beta"),
            "MarketCap": f.get("MarketCap"),
            "Sector": f.get("Sector", "Unknown"),
        })
    fund_df = pd.DataFrame(fund_summary)

    # Portfolio grade
    grade = "C"
    if sharpe_m > 1.2 and er > 0.12:
        grade = "A"
    elif sharpe_m > 0.8 and er > 0.08:
        grade = "B"
    elif sharpe_m < 0.3 or er < 0.03:
        grade = "D"

    # Risk bucket
    if vol < 0.12:
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
    # Portfolio overview
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
        if "Technology" in sector_weights_model and sector_weights_model["Technology"] > 0.45:
            st.write("• Heavy concentration in Technology increases sensitivity to interest rates.")

    # Monte Carlo commentary
    if mc_comment:
        st.markdown("### Monte Carlo Outlook")
        st.write(mc_comment)

    # Sector profile from fundamentals
    if "Sector" in fund_df.columns:
        sectors_from_fund = fund_df["Sector"].fillna("Unknown").value_counts().to_dict()
        if sectors_from_fund:
            st.markdown("### Sector Profile (Fundamentals)")
            sector_lines = [f"{s}: {c} name(s)" for s, c in sectors_from_fund.items()]
            st.write("• " + "; ".join(sector_lines))

    # -----------------------------
    # AI Buy/Hold/Sell Signals
    # -----------------------------
    st.markdown("### AI Buy / Hold / Sell Signals")

    signals = []
    for _, row in fund_df.iterrows():
        t = row["Ticker"]
        pe = row.get("PE")
        pb = row.get("PB")
        dy = row.get("Dividend Yield")
        beta = row.get("Beta")
        momentum_val = momentum_model.get(t, 0)

        score = 0
        conviction = 0

        if pe is not None and not pd.isna(pe) and 0 < pe < 40:
            score += 1
            conviction += 20

        if pb is not None and not pd.isna(pb) and 0 < pb < 8:
            score += 1
            conviction += 15

        if dy is not None and not pd.isna(dy) and dy > 0.005:
            score += 1
            conviction += 15

        if beta is not None and not pd.isna(beta) and beta < 1.3:
            score += 1
            conviction += 20

        if momentum_val is not None and momentum_val > 0:
            score += 1
            conviction += 30

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

    # Signal summary
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

    # Portfolio-level signal
    st.markdown("### AI Portfolio-Level Signal")
    buy_count = len(buys)
    sell_count = len(sells)

    if buy_count > sell_count:
        portfolio_signal = "Buy"
        st.success("**AI Portfolio Signal: BUY** — The portfolio shows strong aggregate fundamentals.")
    elif sell_count >= buy_count + 2:
        portfolio_signal = "Sell"
        st.error("**AI Portfolio Signal: SELL** — The portfolio shows broad fundamental weakness.")
    else:
        portfolio_signal = "Hold"
        st.warning("**AI Portfolio Signal: HOLD** — Mixed signals across the portfolio.")

    # Commentary on signals
    st.markdown("### AI Commentary on Signals")
    if portfolio_signal == "Buy":
        st.write(
            "The portfolio demonstrates broad fundamental strength, with multiple tickers showing "
            "attractive valuation, healthy risk profiles, and supportive momentum."
        )
    elif portfolio_signal == "Sell":
        st.write(
            "The portfolio exhibits widespread fundamental weakness. Several names show elevated risk, "
            "poor valuation, or weak momentum. Rebalancing may be warranted."
        )
    else:
        st.write(
            "The portfolio presents a balanced but indecisive signal profile. Monitoring key metrics "
            "and maintaining diversification is recommended."
        )

# ---------------------------------------------------------
# Buy Analysis (Corrected for New Architecture)
# ---------------------------------------------------------
with tab8:
    st.subheader("Buy Analysis")

    # Safety check — model must exist
    model = st.session_state.get("model")
    if model is None:
        st.info("Run the analysis first to generate optimizer results.")
        st.stop()

    if not run_button:
        st.info("Run Analysis to generate buy analysis.")
        st.stop()

    # Run buy analysis using CLEAN inputs
    buy_results = run_buy_analysis(valid_tickers, fundamentals, close)

    # Clean numeric columns
    numeric_cols = ["PE", "PB", "DividendYield", "Momentum", "Risk", "Score"]
    for col in numeric_cols:
        if col in buy_results.columns:
            buy_results[col] = buy_results[col].apply(safe_val)

    # Rating colors
    def rating_color(val):
        return {"Buy": "🟢 Buy", "Hold": "🟡 Hold", "Sell": "🔴 Sell"}.get(val, "N/A")

    buy_results["RatingColored"] = buy_results["Rating"].apply(rating_color)

    st.dataframe(
        buy_results[
            ["Ticker", "Momentum", "Risk", "PE", "PB", "DividendYield", "Score", "RatingColored"]
        ],
        use_container_width=True
    )

    # Commentary
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

    # Radar chart
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
    st.plotly_chart(fig, use_container_width=True, key="radar_chart")

    # Strengths & Weaknesses
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
# Optimizer (Corrected for New Architecture)
# ---------------------------------------------------------
@st.cache_data(show_spinner=True)
def run_optimizer_cached(returns, cov):
    return run_optimizer(returns, cov)

with tab9:
    st.subheader("Optimizer")

    # Safety check — run button must be pressed
    if not run_button:
        st.info("Run Analysis to generate optimizer results.")
        st.stop()

    # Compute covariance
    cov_matrix = returns_df.cov()

    # Run optimizer
    opt_results = run_optimizer_cached(returns_df, cov_matrix)
    st.success("Optimization complete!")

    # ---------------------------------------------------------
    # Equal Weight Portfolio
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # Maximum Sharpe Portfolio
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # Correlation Heatmap
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # Store optimizer results inside the existing model
    # ---------------------------------------------------------
    if "model" in st.session_state:
        st.session_state["model"]["optimizer"] = opt_results
    else:
        st.session_state["model"] = {"optimizer": opt_results}
