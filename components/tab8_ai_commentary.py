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

        # ---- Safety Checks ----
        if not perf or perf.get("expected_return") is None:
            st.warning("Not enough data to generate commentary.")
            return

        er = perf["expected_return"]
        vol = perf["volatility"]
        sharpe = perf["sharpe"]

        # ---- Drawdown Stats ----
        max_dd = None
        if isinstance(drawdown, pd.DataFrame) and not drawdown.empty:
            max_dd = drawdown["Drawdown"].min()

        # ---- Sector Exposure ----
        sector_text = ""
        if sector_weights is not None:
            sector_text = ", ".join([f"{s}: {w:.1%}" for s, w in sector_weights.items()])

        # ---- Fundamentals Summary ----
        fund_summary = []
        for t in tickers:
            f = fundamentals.get(t, {})
            fund_summary.append({
                "Ticker": t,
                "PE": f.get("pe"),
                "PS": f.get("ps"),
                "PB": f.get("pb"),
                "Rating": f.get("recommendation"),
                "Target Price": f.get("target_mean_price")
            })
        fund_df = pd.DataFrame(fund_summary)

        # ---- Commentary Generation ----
        st.markdown("### 📌 Portfolio Overview")

        st.write(f"""
        **Expected Annual Return:** {er:.2%}  
        **Annualized Volatility:** {vol:.2%}  
        **Sharpe Ratio:** {sharpe:.2f}  
        **Max Drawdown:** {max_dd:.2% if max_dd is not None else "N/A"}  
        """)

        st.markdown("---")

        st.markdown("### 🧠 AI Commentary")

        # ---- Return Commentary ----
        if er > 0.15:
            st.write("• The portfolio shows **strong expected returns**, suggesting meaningful upside potential.")
        elif er > 0.05:
            st.write("• Expected returns are **moderate**, consistent with a balanced risk profile.")
        else:
            st.write("• Expected returns appear **muted**, likely due to low‑growth or defensive components.")

        # ---- Volatility Commentary ----
        if vol > 0.25:
            st.write("• Volatility is **elevated**, indicating exposure to high‑beta or momentum names.")
        elif vol > 0.15:
            st.write("• Volatility is **moderate**, typical for diversified equity portfolios.")
        else:
            st.write("• Volatility is **low**, suggesting defensive or mega‑cap concentration.")

        # ---- Sharpe Commentary ----
        if sharpe > 1.0:
            st.write("• The Sharpe ratio is **strong**, indicating efficient risk‑adjusted performance.")
        elif sharpe > 0.5:
            st.write("• The Sharpe ratio is **acceptable**, though there is room for optimization.")
        else:
            st.write("• The Sharpe ratio is **weak**, implying the portfolio may not be compensated for its risk.")

        # ---- Drawdown Commentary ----
        if max_dd is not None:
            if max_dd < -0.40:
                st.write("• Historical drawdowns are **deep**, suggesting vulnerability during market stress.")
            elif max_dd < -0.20:
                st.write("• Drawdowns are **moderate**, consistent with typical equity risk.")
            else:
                st.write("• Drawdowns are **shallow**, indicating strong downside resilience.")

        # ---- Sector Commentary ----
        if sector_text:
            st.markdown("### 🏢 Sector Exposure")
            st.write(f"**Sector Weights:** {sector_text}")

        # ---- Fundamentals Commentary ----
        st.markdown("### 📊 Fundamentals Snapshot")
        st.dataframe(fund_df, use_container_width=True)

        st.write("• Analyst ratings and valuation multiples provide additional context for upside/downside potential.")
