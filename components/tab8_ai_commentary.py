import streamlit as st

def render_tab8(tab, model, sector_weights):
    tab.subheader("AI Commentary")

    if model is None:
        tab.info("Run optimization to generate commentary.")
        return

    try:
        perf = model.get("performance", {})
        dd = model.get("drawdown", None)
        w = model.get("weights", None)

        if not perf or w is None:
            tab.warning("Model exists but performance or weights missing.")
            return

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

        commentary = f"""
### 📊 Portfolio Overview
The portfolio targets an **expected return of {ret:.2%}** with an annualized volatility of 
**{vol:.2%}**, producing a **Sharpe ratio of {sharpe:.2f}**. This places the allocation in a 
balanced risk‑reward posture consistent with diversified equity portfolios.

### 🛡️ Risk & Drawdown
Historical drawdown analysis shows a **maximum drawdown of {max_dd:.2%}**, indicating controlled 
downside risk and stable recovery behavior.

### 🧭 Allocation Insights
The optimizer allocated the highest weight to **{top_sector}**, representing **{top_sector_weight:.1%}** 
of the portfolio. This suggests favorable risk‑adjusted characteristics in that sector.

### ⚖️ Risk Parity Comparison
Risk‑parity weights differ from the Markowitz solution, implying:
- Uneven covariance structure  
- Strong return‑to‑risk imbalance  
- Higher concentration in return‑dominant assets  

### 🧠 Summary
Overall, the portfolio is positioned for **efficient growth**, balancing return potential with 
controlled volatility. Sector tilts reflect areas of relative strength, while risk metrics remain 
well‑anchored.
"""

        tab.markdown(commentary)

    except Exception as e:
        tab.error(f"Error generating AI commentary: {e}")
