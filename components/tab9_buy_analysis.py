import streamlit as st
import pandas as pd

def render_buy_analysis_tab(tab, prices, model):

    with tab:

        st.subheader("Buy / Hold / Sell Analysis")

        fundamentals = model.get("fundamentals", {})
        tickers = model.get("tickers", [])

        if not fundamentals:
            st.warning("No fundamentals available.")
            return

        rows = []
        for t in tickers:
            f = fundamentals.get(t, {})

            rows.append({
                "Ticker": t,
                "PE": f.get("pe"),
                "PS": f.get("ps"),
                "PB": f.get("pb"),
                "Dividend Yield": f.get("dividend_yield"),
                "Analyst Rating": f.get("recommendation"),
                "Target Price": f.get("target_mean_price"),
                "Market Cap": f.get("market_cap"),
                "Beta": f.get("beta"),
            })

        df = pd.DataFrame(rows)

        st.dataframe(df, use_container_width=True)

        st.markdown("### Interpretation Guide")
        st.write("""
        - **PE < 15** → Often undervalued  
        - **PS < 2** → Reasonable valuation  
        - **PB < 3** → Healthy  
        - **Analyst Rating**:  
            - *strong_buy* → Very bullish  
            - *buy* → Bullish  
            - *hold* → Neutral  
            - *sell* → Bearish  
        """)
