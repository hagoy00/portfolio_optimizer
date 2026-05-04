import streamlit as st
import pandas as pd
import numpy as np

st.write("DEBUG — entering Tab 6")

def render_montecarlo_tab(tab, prices, model):
    tab.markdown("## Monte Carlo Simulation")

    if model is None:
        tab.info("Run optimization first.")
        return

    mc = model.get("monte_carlo", None)

    if mc is None or not isinstance(mc, pd.DataFrame) or mc.empty:
        tab.warning("Model exists but Monte Carlo data is missing.")
        return

    # ---------------------------------------------------
    # MAIN SIMULATION CHART
    # ---------------------------------------------------
    tab.markdown("### Simulation Paths")
    tab.line_chart(mc)

    # ---------------------------------------------------
    # FINAL VALUE DISTRIBUTION
    # ---------------------------------------------------
    final_values = mc.iloc[-1]

    tab.markdown("### Distribution of Final Portfolio Values")
    tab.bar_chart(final_values)

    # ---------------------------------------------------
    # SUMMARY STATISTICS
    # ---------------------------------------------------
    mean_val = final_values.mean()
    median_val = final_values.median()
    worst_val = final_values.min()
    best_val = final_values.max()

    col1, col2, col3, col4 = tab.columns(4)
    col1.metric("Mean Final Value", f"{mean_val:.2f}")
    col2.metric("Median Final Value", f"{median_val:.2f}")
    col3.metric("Best Case", f"{best_val:.2f}")
    col4.metric("Worst Case", f"{worst_val:.2f}")

    # ---------------------------------------------------
    # PERCENTILES
    # ---------------------------------------------------
    p5 = np.percentile(final_values, 5)
    p50 = np.percentile(final_values, 50)
    p95 = np.percentile(final_values, 95)

    tab.markdown("### Percentile Outcomes")
    colA, colB, colC = tab.columns(3)
    colA.metric("5th Percentile (Bad Case)", f"{p5:.2f}")
    colB.metric("50th Percentile (Median)", f"{p50:.2f}")
    colC.metric("95th Percentile (Great Case)", f"{p95:.2f}")

    # ---------------------------------------------------
    # RISK METRICS (VaR & CVaR)
    # ---------------------------------------------------
    # Value-at-Risk (95% confidence)
    var_95 = 1 - p5

    # Conditional VaR (Expected Shortfall)
    cvar_95 = 1 - final_values[final_values <= p5].mean()

    tab.markdown("### Risk Metrics")
    colX, colY = tab.columns(2)
    colX.metric("95% VaR (1-year)", f"{var_95:.2%}")
    colY.metric("95% CVaR (Expected Shortfall)", f"{cvar_95:.2%}")

    # ---------------------------------------------------
    # PROBABILITIES
    # ---------------------------------------------------
    prob_loss = (final_values < 1).mean()
    prob_gain = (final_values > 1).mean()
    prob_double = (final_values >= 2).mean()

    tab.markdown("### Probabilities")
    colL, colM, colN = tab.columns(3)
    colL.metric("Probability of Loss", f"{prob_loss:.1%}")
    colM.metric("Probability of Gain", f"{prob_gain:.1%}")
    colN.metric("Probability of Doubling", f"{prob_double:.1%}")

    # ---------------------------------------------------
    # EXPLANATION SECTION
    # ---------------------------------------------------
    tab.markdown("### How to Interpret This Simulation")

    tab.write("""
Monte Carlo simulation shows **many possible futures** for your portfolio by reshuffling historical returns.

**How to read it:**

- The spaghetti chart shows **200 possible paths** your portfolio might take.
- The spread of the lines shows **uncertainty** — wider = riskier.
- The 5th percentile is a **bad-case scenario**.
- The 95th percentile is a **great-case scenario**.
- VaR tells you **how much you could lose in a bad year**.
- CVaR tells you **how bad things get when they go wrong**.
- Probability of loss shows **how often the portfolio ends below today’s value**.
- Probability of doubling shows **how often the portfolio ends above 2×**.

This gives you a full picture of **risk, reward, and tail events**.
""")
