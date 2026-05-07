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
    df = pd.DataFrame({"Ticker": tickers, "Weight": weights, "Sector": sectors})
    return df.groupby("Sector")["Weight"].sum().to_dict()

sector_weights = compute_sector_weights(weights, tickers)

# ---------------------------------------------------------
# Momentum for AI model
# ---------------------------------------------------------
def compute_momentum_series(returns, window=60):
    momentum = {}
    for t in returns.columns:
        r = returns[t].dropna()
        if len(r) >= window:
            momentum[t] = float(r.tail(window).mean() * 252)
        elif len(r) > 0:
            momentum[t] = float(r.mean() * 252)
        else:
            momentum[t] = 0.0
    return momentum

momentum_dict = compute_momentum_series(returns)

# ---------------------------------------------------------
# Model for AI commentary
# ---------------------------------------------------------
model = {
    "performance": performance,
    "fundamentals": fundamentals,
    "tickers": tickers,
    "drawdown": drawdown_df,
    "sector_weights": sector_weights,
    "monte_carlo": None,
    "momentum": momentum_dict,
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
# Overview
# ---------------------------------------------------------
with tab1:
    st.markdown("<h2 style='color:#1E90FF;'>Optimizer Dashboard Report</h2>", unsafe_allow_html=True)
    st.subheader("Overview")
    st.dataframe(prices.tail())
# ---------------------------------------------------------
# Performance Tab (Upgraded Institutional Version)
# ---------------------------------------------------------
with tab2:
    st.subheader("Performance Metrics")

    # Compute portfolio returns
    port_ret = returns.dot(weights)
    cum_ret = (1 + port_ret).cumprod()

    # Rolling metrics
    rolling_vol = port_ret.rolling(30).std() * np.sqrt(252)
    rolling_sharpe = (port_ret.rolling(30).mean() * 252) / rolling_vol

    # Drawdown
    cum_max = cum_ret.cummax()
    dd = (cum_ret - cum_max) / cum_max

    # Summary metrics
    mu = port_ret.mean() * 252
    vol = port_ret.std() * np.sqrt(252)
    sharpe = mu / vol if vol > 0 else 0
    sortino = (port_ret.mean() * 252) / (port_ret[port_ret < 0].std() * np.sqrt(252))
    calmar = mu / abs(dd.min()) if dd.min() != 0 else 0
    max_dd = dd.min()

    # Display metrics
    st.metric("Expected Return", f"{mu:.2%}")
    st.metric("Volatility", f"{vol:.2%}")
    st.metric("Sharpe Ratio", f"{sharpe:.2f}")
    st.metric("Sortino Ratio", f"{sortino:.2f}")
    st.metric("Calmar Ratio", f"{calmar:.2f}")
    st.metric("Max Drawdown", f"{max_dd:.2%}")

    # Charts
    st.markdown("### Cumulative Return")
    st.line_chart(cum_ret)

    st.markdown("### Rolling 30-Day Volatility")
    st.line_chart(rolling_vol)

    st.markdown("### Rolling 30-Day Sharpe Ratio")
    st.line_chart(rolling_sharpe)

    st.markdown("### Drawdown")
    st.area_chart(dd)

    st.markdown("### Distribution of Daily Returns")
    st.bar_chart(port_ret.hist(bins=40).to_frame())

# ---------------------------------------------------------
# Risk
# ---------------------------------------------------------
with tab3:
    st.subheader("Risk & Drawdown")
    st.line_chart(drawdown_df)

# ---------------------------------------------------------
# Sectors
# ---------------------------------------------------------
with tab4:
    st.subheader("Sector Exposure")
    sector_df = pd.DataFrame.from_dict(sector_weights, orient="index", columns=["Weight"])
    st.bar_chart(sector_df)

# ---------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------
with tab5:
    st.subheader("Fundamentals")

    fundamentals_df = pd.DataFrame(fundamentals).T.drop("full_prices", errors="ignore")
    st.dataframe(fundamentals_df)

    st.subheader("Fundamentals Ranking")

    def score_fundamentals(row):
        score = 0
        if row.get("gross_margins"): score += row["gross_margins"] * 10
        if row.get("profit_margins"): score += row["profit_margins"] * 10
        if row.get("revenue"): score += (row["revenue"] / 1e9)
        if row.get("PE"): score += max(0, 50 - row["PE"])
        if row.get("PB"): score += max(0, 20 - row["PB"])
        if row.get("DividendYield"): score += row["DividendYield"] * 100
        return score

    fundamentals_df["score"] = fundamentals_df.apply(score_fundamentals, axis=1)
    ranked_df = fundamentals_df.sort_values("score", ascending=False)
    st.dataframe(ranked_df[["score"]])

    st.subheader("AI Fundamentals Commentary")

    def generate_fundamentals_commentary(ranked_df):
        lines = []
        tickers_sorted = ranked_df.index.tolist()
        if not tickers_sorted:
            return "No fundamentals available."

        best = tickers_sorted[0]
        worst = tickers_sorted[-1]

        best_row = ranked_df.loc[best]
        best_reasons = []
        if best_row.get("gross_margins"):
            best_reasons.append("strong gross margins")
        if best_row.get("profit_margins"):
            best_reasons.append("solid profitability")
        if best_row.get("revenue"):
            best_reasons.append("healthy revenue base")
        if best_row.get("PE") and best_row["PE"] < 25:
            best_reasons.append("reasonable valuation")
        if best_row.get("DividendYield"):
            best_reasons.append("added income from dividends")
        best_reason_text = ", ".join(best_reasons) if best_reasons else "overall stronger fundamentals"
        lines.append(f"**{best}** ranks as the strongest fundamental name in the group, supported by {best_reason_text}.")

        if len(tickers_sorted) > 2:
            middle = tickers_sorted[1:-1]
            for t in middle:
                row = ranked_df.loc[t]
                mid_reasons = []
                if row.get("gross_margins"):
                    mid_reasons.append("solid margins")
                if row.get("profit_margins"):
                    mid_reasons.append("healthy profitability")
                if row.get("revenue"):
                    mid_reasons.append("stable revenue")
                if row.get("PE") and row["PE"] < 40:
                    mid_reasons.append("fair valuation")
                reason_text = ", ".join(mid_reasons) if mid_reasons else "balanced fundamentals"
                lines.append(f"**{t}** shows {reason_text}, placing it in the middle of the group.")

        worst_row = ranked_df.loc[worst]
        worst_reasons = []
        if worst_row.get("gross_margins") and worst_row["gross_margins"] < 0.2:
            worst_reasons.append("thin margins")
        if worst_row.get("profit_margins") and worst_row["profit_margins"] < 0.1:
            worst_reasons.append("weak profitability")
        if worst_row.get("PE") and worst_row["PE"] > 50:
            worst_reasons.append("elevated valuation")
        if worst_row.get("PB") and worst_row["PB"] > 10:
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
    st.subheader("Weights")
    weights_df = pd.DataFrame({"Ticker": tickers, "Weight": weights})
    st.dataframe(weights_df)
# ---------------------------------------------------------
# AI Commentary
# ---------------------------------------------------------
with tab7:
    st.subheader("AI Portfolio Commentary")

    perf = model.get("performance", {})
    fundamentals_model = model.get("fundamentals", {})
    tickers_model = model.get("tickers", tickers)
    drawdown_model = model.get("drawdown", drawdown_df)
    sector_weights_model = model.get("sector_weights", sector_weights)
    mc = model.get("monte_carlo")

    # -----------------------------
    # Guard: No performance → stop
    # -----------------------------
    if not perf or perf.get("expected_return") is None:
        st.warning("Not enough data to generate commentary.")
        st.stop()

    er = perf["expected_return"]
    vol = perf["volatility"]
    sharpe = perf["sharpe"]

    # -----------------------------
    # Drawdown
    # -----------------------------
    if isinstance(drawdown_model, pd.DataFrame) and not drawdown_model.empty:
        max_dd = float(drawdown_model["Drawdown"].min())
    else:
        max_dd = None
    max_dd_text = f"{max_dd:.2%}" if isinstance(max_dd, (int, float, np.floating)) else "N/A"

    # -----------------------------
    # Sector weights
    # -----------------------------
    sector_text = ""
    if sector_weights_model:
        sector_text = ", ".join([f"{s}: {w:.1%}" for s, w in sector_weights_model.items()])

    # -----------------------------
    # Fundamentals summary (safe)
    # -----------------------------
    fund_summary = []
    for t in tickers_model:
        f = fundamentals_model.get(t, {})
        fund_summary.append({
            "Ticker": t,
            "PE": f.get("PE") if f.get("PE") not in [None, 0] else None,
            "PB": f.get("PB") if f.get("PB") not in [None, 0] else None,
            "Dividend Yield": f.get("DividendYield") if f.get("DividendYield") not in [None, 0] else None,
            "Beta": f.get("beta") if f.get("beta") not in [None, 0] else None,
        })
    fund_df = pd.DataFrame(fund_summary)

    # -----------------------------
    # Portfolio Grade
    # -----------------------------
    grade = "C"
    if sharpe > 1.2 and er > 0.12:
        grade = "A"
    elif sharpe > 0.8 and er > 0.08:
        grade = "B"
    elif sharpe < 0.3 or er < 0.03:
        grade = "D"

    # -----------------------------
    # Risk Bucket
    # -----------------------------
    if vol < 0.12:
        risk_bucket = "Low Risk"
    elif vol < 0.20:
        risk_bucket = "Moderate Risk"
    else:
        risk_bucket = "High Risk"

    # -----------------------------
    # Monte Carlo Commentary
    # -----------------------------
    mc_comment = ""
    if mc is not None and isinstance(mc, pd.DataFrame) and not mc.empty:
        final_vals = mc.iloc[-1]
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
    **Sharpe Ratio:** {sharpe:.2f}  
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
    if sharpe > 1.0:
        st.write("• Strong Sharpe ratio indicates efficient risk-adjusted performance.")
    elif sharpe > 0.5:
        st.write("• Sharpe ratio is acceptable but could be improved.")
    else:
        st.write("• Weak Sharpe ratio suggests the portfolio may not be compensated for its risk.")

    # Drawdown commentary
    if isinstance(max_dd, (int, float, np.floating)):
        if max_dd < -0.40:
            st.write("• Deep drawdowns indicate vulnerability during market stress.")
        elif max_dd < -0.20:
            st.write("• Drawdowns are moderate and typical for equities.")
        else:
            st.write("• Shallow drawdowns indicate strong downside resilience.")

    # Sector commentary
    if sector_text:
        st.markdown("### Sector Exposure")
        st.write(f"**Sector Weights:** {sector_text}")
        if "Technology" in sector_weights_model and sector_weights_model["Technology"] > 0.45:
            st.write("• Heavy concentration in Technology increases sensitivity to interest rates.")

    # Monte Carlo
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
        dy = row["Dividend Yield"]
        beta = row["Beta"]
        momentum = momentum_dict.get(t, 0)

        score = 0
        conviction = 0

        if pe and 0 < pe < 40:
            score += 1
            conviction += 20
        if pb and 0 < pb < 8:
            score += 1
            conviction += 15
        if dy and dy > 0.005:
            score += 1
            conviction += 15
        if beta and beta < 1.3:
            score += 1
            conviction += 20
        if momentum > 0:
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
            "Momentum": momentum,
            "Score": score,
            "Conviction": conviction,
            "Rating": rating
        })

    signals_df = pd.DataFrame(signals)
    st.dataframe(signals_df)

    # -----------------------------
    # Signal Summary
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
    # Portfolio-Level Signal (fixed)
    # -----------------------------
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

    # -----------------------------
    # Commentary on Signals
    # -----------------------------
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
# Buy Analysis
# ---------------------------------------------------------
with tab8:
    st.subheader("Buy / Hold / Sell Analysis")

    if not run_button:
        st.info("Run Analysis to generate buy analysis.")
    else:
        buy_results = run_buy_analysis(tickers, fundamentals, prices)

        numeric_cols = ["PE", "PB", "DividendYield", "Momentum", "Risk", "Score"]
        for col in numeric_cols:
            if col in buy_results.columns:
                buy_results[col] = buy_results[col].fillna(0)

        def rating_color(val):
            if val == "Buy":
                return "🟢 Buy"
            elif val == "Hold":
                return "🟡 Hold"
            else:
                return "🔴 Sell"

        buy_results["RatingColored"] = buy_results["Rating"].apply(rating_color)

        st.dataframe(
            buy_results[
                ["Ticker", "Momentum", "Risk", "PE", "PB", "DividendYield", "Score", "RatingColored"]
            ]
        )

        st.subheader("AI Buy Analysis Commentary")

        def generate_buy_commentary(df):
            if df.empty:
                return "No buy analysis available."
            lines = []
            best = df.sort_values("Score", ascending=False).iloc[0]
            lines.append(
                f"**{best['Ticker']}** leads the group with a score of {best['Score']}."
            )
            worst = df.sort_values("Score", ascending=True).iloc[0]
            lines.append(
                f"**{worst['Ticker']}** ranks weakest with a score of {worst['Score']}."
            )
            return "\n\n".join(lines)

        st.markdown(generate_buy_commentary(buy_results))

        import plotly.graph_objects as go

        st.subheader("Fundamentals Radar Chart")
        radar_cols = ["PE", "PB", "DividendYield", "Momentum", "Risk"]
        fig = go.Figure()
        for _, row in buy_results.iterrows():
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

        st.subheader("Top Strengths & Weaknesses")

        def strengths_weaknesses(row):
            strengths, weaknesses = [], []
            strengths.append("Positive momentum") if row["Momentum"] > 0 else weaknesses.append("Weak momentum")
            strengths.append("Low volatility") if row["Risk"] < 0.30 else weaknesses.append("High volatility")
            strengths.append("Reasonable PE ratio") if 0 < row["PE"] < 30 else weaknesses.append("Stretched PE ratio")
            strengths.append("Healthy PB ratio") if 0 < row["PB"] < 5 else weaknesses.append("Rich PB ratio")
            strengths.append("Dividend support") if row["DividendYield"] > 0.01 else weaknesses.append("Low or no dividend")
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
# Optimizer
# ---------------------------------------------------------
@st.cache_data(show_spinner=True)
def run_optimizer_cached(returns, cov):
    return run_optimizer(returns, cov)

with tab9:
    st.subheader("Optimizer & Monte Carlo")

    if not run_button:
        st.info("Run Analysis to generate optimizer and Monte Carlo results.")
    else:
        opt_results = run_optimizer_cached(returns, cov)
        st.success("Optimization complete!")

        # -----------------------------
        # Equal Weight Portfolio
        # -----------------------------
        st.markdown("### Equal Weight Portfolio")
        ew = opt_results["equal_weight"]
        st.write(f"**Expected Return:** {ew['expected_return']:.2%}")
        st.write(f"**Volatility:** {ew['volatility']:.2%}")
        st.write(f"**Sharpe Ratio:** {ew['sharpe']:.2f}")

        ew_df = pd.DataFrame({
            "Ticker": opt_results["tickers"],
            "Weight": ew["weights"]
        })
        st.dataframe(ew_df)

        # -----------------------------
        # Minimum Volatility Portfolio
        # -----------------------------
        st.markdown("### Minimum Volatility Portfolio")
        mv = opt_results["min_volatility"]
        st.write(f"**Expected Return:** {mv['expected_return']:.2%}")
        st.write(f"**Volatility:** {mv['volatility']:.2%}")
        st.write(f"**Sharpe Ratio:** {mv['sharpe']:.2f}")

        mv_df = pd.DataFrame({
            "Ticker": opt_results["tickers"],
            "Weight": mv["weights"]
        })
        st.dataframe(mv_df)

        # -----------------------------
        # Maximum Sharpe Portfolio
        # -----------------------------
        st.markdown("### Maximum Sharpe Portfolio")
        ms = opt_results["max_sharpe"]
        st.write(f"**Expected Return:** {ms['expected_return']:.2%}")
        st.write(f"**Volatility:** {ms['volatility']:.2%}")
        st.write(f"**Sharpe Ratio:** {ms['sharpe']:.2f}")

        ms_df = pd.DataFrame({
            "Ticker": opt_results["tickers"],
            "Weight": ms["weights"]
        })
        st.dataframe(ms_df)

        # -----------------------------
        # Monte Carlo Simulation
        # -----------------------------
        st.markdown("### Monte Carlo Simulation")
        mc_df = run_monte_carlo_simulation(returns, mc_sims, mc_horizon)
        st.line_chart(mc_df)

