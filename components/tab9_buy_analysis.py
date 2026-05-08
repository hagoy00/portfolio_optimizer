import streamlit as st
import pandas as pd
import numpy as np

def render_buy_analysis_tab(tab, prices, model):

    with tab:

        st.subheader("Buy / Hold / Sell Analysis")

        fundamentals = model.get("fundamentals", {})
        tickers = model.get("tickers", [])
        returns = model.get("returns", None)
        momentum = model.get("momentum", None)

        if fundamentals is None or len(fundamentals) == 0:
            st.warning("No fundamentals available.")
            return

        if returns is None or returns.empty:
            st.warning("No returns data available.")
            return

        # ---------------------------------------------------------
        # Compute 21‑day momentum score
        # ---------------------------------------------------------
        mom_scores = {}
        if isinstance(momentum, pd.Series):
            for t in tickers:
                val = momentum.get(t, None)
                mom_scores[t] = float(val) if val is not None else None
        else:
            for t in tickers:
                mom_scores[t] = None

        # ---------------------------------------------------------
        # Compute volatility score (lower volatility = better)
        # ---------------------------------------------------------
        vol_scores = {}
        for t in tickers:
            try:
                vol = returns[t].std() * np.sqrt(252)
                vol_scores[t] = float(vol)
            except:
                vol_scores[t] = None

        # ---------------------------------------------------------
        # Compute fundamentals score
        # ---------------------------------------------------------
        results = []

        for t in tickers:
            f = fundamentals.get(t, {})

            pe = f.get("pe")
            pb = f.get("pb")
            ps = f.get("ps")
            forward_pe = f.get("forward_pe")
            eps = f.get("eps")
            margins = f.get("profit_margins")
            rating = f.get("recommendation")
            target = f.get("target_mean_price")

            # Score components
            score = 0
            reasons = []

            # Valuation
            if pe is not None and pe > 0 and pe < 20:
                score += 1
                reasons.append("Attractive PE")
            if pb is not None and pb < 3:
                score += 1
                reasons.append("Reasonable PB")
            if ps is not None and ps < 5:
                score += 1
                reasons.append("Healthy PS")

            # Growth / Profitability
            if eps is not None and eps > 0:
                score += 1
                reasons.append("Positive EPS")
            if margins is not None and margins > 0.10:
                score += 1
                reasons.append("Strong margins")

            # Analyst sentiment
            if rating in ["buy", "strong_buy"]:
                score += 1
                reasons.append("Analysts bullish")

            # Target price premium
            try:
                current_price = prices[t].iloc[-1]
                if target is not None and current_price is not None:
                    if target > current_price * 1.10:
                        score += 1
                        reasons.append("Upside to target price")
            except:
                pass

            # Momentum
            mom = mom_scores.get(t)
            if mom is not None and mom > 0:
                score += 1
                reasons.append("Positive momentum")

            # Volatility
            vol = vol_scores.get(t)
            if vol is not None and vol < 0.30:
                score += 1
                reasons.append("Low volatility")

            # Final rating
            if score >= 6:
                rating_final = "BUY"
            elif score >= 3:
                rating_final = "HOLD"
            else:
                rating_final = "SELL"

            results.append({
                "Ticker": t,
                "Score": score,
                "Rating": rating_final,
                "Reasons": ", ".join(reasons),
                "PE": pe,
                "PB": pb,
                "PS": ps,
                "EPS": eps,
                "Margins": margins,
                "Momentum": mom,
                "Volatility": vol,
                "Analyst Rating": rating,
                "Target Price": target,
            })

        df = pd.DataFrame(results)

        st.subheader("Buy / Hold / Sell Summary")
        st.dataframe(df, use_container_width=True)

        # Highlight BUY opportunities
        buys = df[df["Rating"] == "BUY"]
        if not buys.empty:
            st.success("**BUY Opportunities:** " + ", ".join(buys["Ticker"].tolist()))
        else:
            st.info("No strong BUY signals at this time.")
