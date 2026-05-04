import streamlit as st

def render_ai_commentary_tab(tab, prices, model):

#def render_ai_commentary_tab(tab, model, sector_weights):
    tab.markdown("## AI Commentary")
sector_weights = model.get("sector_weights")

    if model is None:
        tab.info("Run optimization to generate commentary.")
        return

    try:
        perf = model.get("performance", {})
        dd = model.get("drawdown", None)
        w = model.get("weights", None)

        # ---------------------------------------------------
        # GUARD CLAUSE — missing data
        # ---------------------------------------------------
        if not perf or w is None:
            tab.warning("Model exists but performance or weights are missing.")
            return

        # Extract metrics
        ret = perf.get("return", 0)
        vol = perf.get("volatility", 0)
        sharpe = perf.get("sharpe", 0)
        max_dd = dd.min().min() if dd is not None else None

        # Sector commentary
        if sector_weights:
            top_sector = max(sector_weights, key=sector_weights.get)
            top_sector_weight = sector_weights[top_sector]
        else:
            top_sector = "Unknown"
            top_sector_weight = 0

        # ---------------------------------------------------
        # METRIC PANEL
        # ---------------------------------------------------
        tab.markdown("### Portfolio Metrics Summary")

        col1, col2, col3 = tab.columns(3)
        col1.metric("Expected Return", f"{ret:.2%}")
        col2.metric("Volatility", f"{vol:.2%}")
        col3.metric("Sharpe Ratio", f"{sharpe:.2f}")

        col4, col5 = tab.columns(2)
        col4.metric("Max Drawdown", f"{max_dd:.2%}" if max_dd else "N/A")
        col5.metric("Top Sector Weight", f"{top_sector_weight:.2%}")

        tab.markdown("---")

        # ---------------------------------------------------
        # COMMENTARY BLOCKS
        # ---------------------------------------------------
        with tab.expander("Portfolio Overview", expanded=True):
            tab.markdown(
                f"""
The portfolio targets an expected return of **{ret:.2%}** with an annualized volatility of 
**{vol:.2%}**, resulting in a **Sharpe ratio of {sharpe:.2f}**.  
This places the allocation in a balanced risk‑reward posture consistent with diversified equity portfolios.
"""
            )

        with tab.expander("Risk & Drawdown Analysis", expanded=False):
            tab.markdown(
                f"""
Historical drawdown analysis indicates a maximum drawdown of **{max_dd:.2%}**, 
suggesting controlled downside risk and stable recovery behavior.
"""
            )

        with tab.expander("Allocation Insights", expanded=False):
            tab.markdown(
                f"""
The optimizer allocated the highest weight to **{top_sector}**, representing 
**{top_sector_weight:.2%}** of the portfolio.  
This tilt reflects favorable risk‑adjusted characteristics in that sector.
"""
            )

        with tab.expander("Risk Parity Comparison", expanded=False):
            tab.markdown(
                """
Risk‑parity weights differ from the Markowitz solution, indicating:

- Uneven covariance structure  
- Strong return‑to‑risk imbalance  
- Higher concentration in return‑dominant assets  
"""
            )

        with tab.expander("Summary", expanded=True):
            tab.markdown(
                """
Overall, the portfolio is positioned for efficient growth, balancing return potential with 
controlled volatility. Sector tilts reflect areas of relative strength, while risk metrics 
remain well‑anchored.
"""
            )

    except Exception as e:
        tab.error(f"Error generating AI commentary: {e}")
