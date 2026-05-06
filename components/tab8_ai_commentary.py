import streamlit as st
import pandas as pd
import numpy as np

def render_ai_commentary_tab(tab, prices, model):

    with tab:

        st.subheader("AI Portfolio Commentary")

        perf = model.get("performance", {})
        fundamentals = model.get("fundamentals", {})
        tickers = model.get("tickers", [])
        drawdown = model.get("drawdown")
        sector_weights = model.get("sector_weights", None)
        mc = model.get("monte_carlo")

        if not perf or perf.get("expected_return") is None:
            st.warning("Not enough data to generate commentary.")
            return

        er = perf["expected_return"]
        vol = perf["volatility"]
        sharpe = perf["sharpe"]

        # ---- Drawdown ----
        max_dd = None
        if isinstance(drawdown, pd.DataFrame) and not drawdown.empty:
            dd_series = model["drawdown"]["Drawdown"]
            max_dd = dd_series.min() if dd_series is not None else None

        # ---- Sector Exposure ----
        sector_text = ""
        if sector_weights is not None:
            sector_text = ", ".join([f"{s}: {w:.1%}" for s, w in sector_weights.items()])

        # ---- Fundamentals Table ----
        fund_summary = []
        for t in tickers:
            f = fundamentals.get(t, {})
            fund_summary.append({
                "Ticker": t,
                "PE": f.get("pe"),
                "PS": f.get("ps"),
                "PB": f.get("pb"),
                "Rating": f.get("recommendation"),
                "Target Price": f.get("target_mean_price"),
                "Dividend Yield": f.get("dividend_yield"),
                "Beta": f.get("beta"),
            })
        fund_df = pd.DataFrame(fund_summary)

        # ---- Portfolio Grade ----
        grade = "C"
        if sharpe > 1.2 and er > 0.12:
            grade = "A"
        elif sharpe > 0.8 and er > 0.08:
            grade = "B"
        elif sharpe < 0.3 or er < 0.03:
            grade = "D"

        # ---- Risk Bucket ----
        if vol < 0.12:
            risk_bucket = "Low Risk"
        elif vol < 0.20:
            risk_bucket = "Moderate Risk"
        else:
            risk_bucket = "High Risk"

        # ---- Monte Carlo ----
        mc_comment = ""
        if mc is not None and isinstance(mc, pd.DataFrame):
            final_vals = mc.iloc[-1]
            p5 = np.percentile(final_vals, 5)
            p50 = np.percentile(final_vals, 50)
            p95 = np.percentile(final_vals, 95)

            mc_comment = (
                f"Simulations show a **5% worst-case outcome of {p5:.2f}x**, "
                f"a **median outcome of {p50:.2f}x**, and a **best-case outcome of {p95:.2f}x**."
            )

        # ---- Display Metrics ----
        st.markdown("### 📌 Portfolio Overview")

        st.write(f"""
        **Portfolio Grade:** {grade}  
        **Risk Bucket:** {risk_bucket}  
        **Expected Annual Return:** {er:.2%}  
        **Annualized Volatility:** {vol:.2%}  
        **Sharpe Ratio:** {sharpe:.2f}  
        **Max Drawdown:** {max_dd:.2% if max_dd is not None else "N/A"}  
        """)

        st.markdown("---")

        # ---- AI Commentary ----
        st.markdown("### 🧠 AI Commentary")

        # Return Commentary
        if er > 0.15:
            st.write("• Strong expected returns suggest meaningful upside potential.")
        elif er > 0.05:
            st.write("• Expected returns are moderate and consistent with balanced equity exposure.")
        else:
            st.write("• Expected returns appear muted, likely due to defensive or low-growth names.")

        # Volatility Commentary
        if vol > 0.25:
            st.write("• Volatility is high, indicating exposure to high-beta or momentum stocks.")
        elif vol > 0.15:
            st.write("• Volatility is moderate, typical for diversified portfolios.")
        else:
            st.write("• Volatility is low, suggesting defensive or mega-cap concentration.")

        # Sharpe Commentary
        if sharpe > 1.0:
            st.write("• Strong Sharpe ratio indicates efficient risk-adjusted performance.")
        elif sharpe > 0.5:
            st.write("• Sharpe ratio is acceptable but could be improved.")
        else:
            st.write("• Weak Sharpe ratio suggests the portfolio may not be compensated for its risk.")

        # Drawdown Commentary
        if max_dd is not None:
            if max_dd < -0.40:
                st.write("• Deep drawdowns indicate vulnerability during market stress.")
            elif max_dd < -0.20:
                st.write("• Drawdowns are moderate and typical for equities.")
            else:
                st.write("• Shallow drawdowns indicate strong downside resilience.")

        # Sector Commentary
        if sector_text:
            st.markdown("### 🏢 Sector Exposure")
            st.write(f"**Sector Weights:** {sector_text}")

            if "Technology" in sector_weights and sector_weights["Technology"] > 0.45:
                st.write("• Heavy concentration in Technology increases sensitivity to interest rates.")

        # Monte Carlo Commentary
        if mc_comment:
            st.markdown("### 📈 Monte Carlo Outlook")
            st.write(mc_comment)

        # ---- Buy/Hold/Sell Scoring ----
        st.markdown("### 📊 AI Buy / Hold / Sell Signals")

        signals = []
        for _, row in fund_df.iterrows():
            score = 0
            t = row["Ticker"]

            # Valuation
            if row["PE"] and row["PE"] < 15:
                score += 1
            if row["PE"] and row["PE"] > 40:
                score -= 1

            # Analyst Rating
            if row["Rating"] in ["strong_buy", "buy"]:
                score += 1
            if row["Rating"] in ["sell", "strong_sell"]:
                score -= 1

            # Beta
            if row["Beta"] and row["Beta"] < 1:
                score += 0.5
            if row["Beta"] and row["Beta"] > 1.5:
                score -= 0.5

            # Final Signal
            if score >= 1.5:
                signal = "BUY"
            elif score <= -1:
                signal = "SELL"
            else:
                signal = "HOLD"

            signals.append({"Ticker": t, "Signal": signal, "Score": score})

        st.dataframe(pd.DataFrame(signals), use_container_width=True)

        # ---- Optimization Suggestions ----
        st.markdown("### 🛠 Optimization Suggestions")

        suggestions = []

        if sharpe < 0.5:
            suggestions.append("• Improve Sharpe ratio by reducing high-volatility names.")
        if vol > 0.25:
            suggestions.append("• Consider lowering exposure to high-beta stocks.")
        if er < 0.05:
            suggestions.append("• Expected return is low — consider adding growth or momentum names.")
        if sector_weights is not None and "Technology" in sector_weights and sector_weights["Technology"] > 0.45:
            suggestions.append("• Reduce Technology concentration to improve diversification.")

        if not suggestions:
            suggestions.append("• Portfolio is well-balanced with no major red flags.")

        for s in suggestions:
            st.write(s)

        # ---- Fundamentals Table ----
        st.markdown("### 📑 Fundamentals Snapshot")
        st.dataframe(fund_df, use_container_width=True)
