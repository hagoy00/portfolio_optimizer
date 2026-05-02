import streamlit as st
import pandas as pd
import yfinance as yf
import datetime

import optimizer_core as optimizer_core
import tab1_summary as tab1_summary
import tab2_frontier as tab2_frontier
import tab3_weights as tab3_weights
import tab4_sector as tab4_sector
import tab5_drawdown as tab5_drawdown
import tab6_montecarlo as tab6_montecarlo
import tab7_rebalancing as tab7_rebalancing
import tab8_ai_commentary as tab8_ai_commentary
import tab9_buy_analysis as tab9_buy_analysis

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Portfolio Optimizer Dashboard",
    layout="wide",
)

# ---------------------------------------------------
# REMOVE STREAMLIT DEFAULT HEADER/FOOTER
# ---------------------------------------------------
hide_streamlit_style = """
    <style>
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ---------------------------------------------------
# STICKY HEADER CSS
# ---------------------------------------------------
st.markdown("""
<style>
.sticky-header {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    padding: 18px 30px 12px 30px;
    background-color: #1C1F26;
    z-index: 9999;
    border-bottom: 1px solid #444;
}
.gradient-line {
    height: 3px;
    background: linear-gradient(to right, #2E86C1, #6BB9F0);
    margin-top: 6px;
}
.main-content {
    margin-top: 120px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# STICKY HEADER HTML
# ---------------------------------------------------
st.markdown(f"""
<div class="sticky-header">
    <h1 style="margin-bottom: 0; color: #2E86C1;">Portfolio Optimizer Dashboard</h1>
    <p style="color: #AAAAAA; font-size: 14px; margin-top: 4px;">
        Last updated: {datetime.datetime.now().strftime("%B %d, %Y • %I:%M %p")}
    </p>
    <div class="gradient-line"></div>
</div>

<div class="main-content">
""", unsafe_allow_html=True)

# ---------------------------------------------------
# SIDEBAR — GLOBAL CONTROLS
# ---------------------------------------------------
st.sidebar.header("Portfolio Settings")

tickers_input = st.sidebar.text_input(
    "Tickers (comma-separated)",
    value="AAPL, MSFT, AMZN",
    placeholder="Enter tickers like: AAPL, MSFT, AMZN",
    help="Enter the tickers you want to include in the portfolio."
)

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

risk_free_rate = st.sidebar.number_input(
    "Risk-free rate (annual, %)",
    min_value=0.0,
    max_value=10.0,
    value=4.0,
    step=0.25
)

rebalance_freq = st.sidebar.selectbox(
    "Rebalancing frequency",
    ["Monthly", "Quarterly", "Yearly"]
)

# You can store shared state here if tabs need it
st.session_state["tickers_input"] = tickers_input
st.session_state["start_date"] = start_date
st.session_state["end_date"] = end_date
st.session_state["risk_free_rate"] = risk_free_rate
st.session_state["rebalance_freq"] = rebalance_freq

# ---------------------------------------------------
# TABS
# ---------------------------------------------------
tab_titles = [
    "Summary",
    "Efficient Frontier",
    "Weights",
    "Sector",
    "Drawdown",
    "Monte Carlo",
    "Rebalancing",
    "AI Commentary",
    "Buy Analysis",
]

(
    tab1,
    tab2,
    tab3,
    tab4,
    tab5,
    tab6,
    tab7,
    tab8,
    tab9,
) = st.tabs(tab_titles)

# ---------------------------------------------------
# TAB 1 — SUMMARY
# ---------------------------------------------------
with tab1:
    # Assumes tab1_summary.py exposes a function like: render()
    tab1_summary.render()

# ---------------------------------------------------
# TAB 2 — EFFICIENT FRONTIER
# ---------------------------------------------------
with tab2:
    tab2_frontier.render()

# ---------------------------------------------------
# TAB 3 — WEIGHTS
# ---------------------------------------------------
with tab3:
    tab3_weights.render()

# ---------------------------------------------------
# TAB 4 — SECTOR
# ---------------------------------------------------
with tab4:
    tab4_sector.render()

# ---------------------------------------------------
# TAB 5 — DRAWDOWN
# ---------------------------------------------------
with tab5:
    tab5_drawdown.render()

# ---------------------------------------------------
# TAB 6 — MONTE CARLO
# ---------------------------------------------------
with tab6:
    tab6_montecarlo.render()

# ---------------------------------------------------
# TAB 7 — REBALANCING
# ---------------------------------------------------
with tab7:
    tab7_rebalancing.render()

# ---------------------------------------------------
# TAB 8 — AI COMMENTARY
# ---------------------------------------------------
with tab8:
    tab8_ai_commentary.render()

# ---------------------------------------------------
# TAB 9 — BUY ANALYSIS
# ---------------------------------------------------
with tab9:
    tab9_buy_analysis.render()

# ---------------------------------------------------
# CLOSE MAIN CONTENT WRAPPER
# ---------------------------------------------------
st.markdown("</div>", unsafe_allow_html=True)
