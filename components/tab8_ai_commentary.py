import streamlit as st
import pandas as pd
import numpy as np

def render_ai_commentary_tab(tab, prices, model):

    with tab:

        st.subheader("AI Portfolio Commentary")

        perf = model.get("performance", {})
        fundamentals = model.get("fundamentals", {})
        tickers = model.get("tickers", [])
        drawdown_df = model.get("drawdown")
        sector_weights = model.get("sector_weights", None)
        mc = model.get("monte_carlo")

        if not perf or perf.get("expected_return") is None:
            st.warning("Not enough data to generate commentary.")
            return

        er = perf["expected_return"]
        vol = perf["volatility"]
        sharpe = perf["sharpe"]

        # ---- Drawdown ----
        dd = drawdown_df
        if isinstance(dd, pd.DataFrame) and not dd.empty:
            dd_series = dd["Drawdown"]
            max_dd = float(dd_series.min())
        else:
            max_dd = None

        max_dd_text = (
            f"{max_dd:.2%}"
            if isinstance(max_dd, (int, float, np.floating))
            else "N/A"
        )

        # ---- Sector Exposure ----
        sector_text = ""
        if sector_weights is not None:
            sector_text = ", ".join(
                [f"{s}: {w:.1%}" for s, w in sector_weights.items()]
            )

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
        if mc is not None and isinstance(mc, pd.DataFrame) and not mc.empty:
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
        if isinstance(max_dd, (int, float, np.floating)):
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

            if (
                sector_weights is not None
                and "Technology" in sector_weights
                and sector_weights["Technology"] > 0.45
            ):
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
                score
