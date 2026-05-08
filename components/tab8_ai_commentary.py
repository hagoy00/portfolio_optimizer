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

        # ---------------------------------------------------------
        # Validate performance
        # ---------------------------------------------------------
        if not perf or perf.get("expected_return") is None:
            st.warning("Not enough data to generate commentary.")
            return

        er = perf["expected_return"]
        vol = perf["volatility"]
        sharpe = perf["sharpe"]

        # ---------------------------------------------------------
        # Drawdown
        # ---------------------------------------------------------
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

        # ---------------------------------------------------------
        # Monte Carlo Summary
        # ---------------------------------------------------------
        if mc is not None and not mc.empty:
            mc_mean = mc.iloc[-1].mean()
            mc_p10 = mc.iloc[-1].quantile(0.10)
            mc_p90 = mc.iloc[-1].quantile(0.90)
        else:
            mc_mean = mc_p10 = mc_p90 = None

        # ---------------------------------------------------------
        # Fundamentals Table (FIX #3 APPLIED)
        # ---------------------------------------------------------
        fund_summary = []
        for t in tickers:
            f = fundamentals.get(t, {})

            fund_summary.append({
                "Ticker": t,
                "PE": f.get("pe"),
                "PB": f.get("pb"),
                "PS": f.get("ps"),
                "Forward PE": f.get("forward_pe"),
                "Dividend Yield": f.get("dividend_yield"),
                "Beta": f.get("beta"),
                "EPS": f.get("eps"),
                "Sector": f.get("sector"),
                "Gross Margins": f.get("gross_margins"),
                "Profit Margins": f.get("profit_margins"),
                "Revenue": f.get("revenue"),
                "Rating": f.get("recommendation"),
                "Target Price": f.get("target_mean_price"),
            })

        fund_df = pd.DataFrame(fund_summary)

        st.subheader("Fundamentals Overview")
        st.dataframe(fund_df, use_container_width=True)

        # ---------------------------------------------------------
        # AI Commentary Section
        # ---------------------------------------------------------
        st.subheader("AI‑Generated Commentary")

        commentary = []

        # Performance commentary
        commentary.append(
            f"Your portfolio shows an expected annual return of **{er:.2%}** "
            f"with volatility of **{vol:.2%}**, resulting in a Sharpe ratio of **{sharpe:.2f}**."
        )

        # Drawdown commentary
        commentary.append(
            f"The maximum historical drawdown observed is **{max_dd_text}**, "
            "indicating the worst peak‑to‑trough decline during the period."
        )

        # Monte Carlo commentary
        if mc_mean is not None:
            commentary.append(
                f"Monte Carlo simulations project an average ending value of **{mc_mean:.2f}**, "
                f"with a 10th percentile outcome of **{mc_p10:.2f}** and a 90th percentile outcome of **{mc_p90:.2f}**."
            )

        # Sector commentary
        if sector_weights:
            top_sector = max(sector_weights, key=sector_weights.get)
            commentary.append(
                f"Your largest sector exposure is **{top_sector}**, "
                f"representing **{sector_weights[top_sector]:.2%}** of the portfolio."
            )

        # Fundamentals commentary
        commentary.append(
            "Fundamentals across your holdings have been analyzed, including valuation ratios, "
            "profitability margins, revenue levels, and analyst sentiment."
        )

        # Display commentary
        for line in commentary:
            st.write("• " + line)
