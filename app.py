import streamlit as st
import pandas as pd
import yfinance as yf
import datetime

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Portfolio Optimizer Dashboard",
    layout="wide",
)

# ---------------------------------------------------
# STICKY HEADER CSS
# ---------------------------------------------------
st.markdown("""
<style>
/* Sticky header container */
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

/* Gradient line under header */
.gradient-line {
    height: 3px;
    background: linear-gradient(to right, #2E86C1, #6BB9F0);
    margin-top: 6px;
}

/* Push page content down so header doesn't overlap */
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
# SIDEBAR — Ticker Input
# ---------------------------------------------------
st.sidebar.header("Portfolio Settings")

tickers_input = st.sidebar.text_input(
    "Tickers (comma-separated)",
    value="",
    placeholder="Enter tickers like: AAPL, MSFT, AMZN",
    help="Enter the tickers you want to include in the portfolio."
)

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# ---------------------------------------------------
# MAIN APP CONTENT (placeholder)
# ---------------------------------------------------
st.write("Your dashboard content goes here…")
st.write("Tabs, charts, optimizers, etc.")

# ---------------------------------------------------
# CLOSE MAIN CONTENT WRAPPER
# ---------------------------------------------------
st.markdown("</div>", unsafe_allow_html=True)
