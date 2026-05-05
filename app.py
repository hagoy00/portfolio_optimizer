import streamlit as st
import pandas as pd
import numpy as np
from datetime import date

# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Portfolio Optimizer Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# FIXED STICKY TITLE BAR
# ---------------------------------------------------------
st.markdown("""
<style>
/* Hide Streamlit default header */
header[data-testid="stHeader"] {
    display: none;
}

/* Custom fixed title bar */
.app-title {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 9999;
    background-color: #0E1117;
    padding: 16px 32px;
    border-bottom: 1px solid #1F2937;
}

/* Push content down */
div[data-testid="stAppViewContainer"] {
    padding-top: 70px;
}

/* Title text */
.app-title h1 {
    color: #4DA8FF !important;
    font-size: 28px;
    font-weight: 700;
    margin: 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="app-title"><h1>Portfolio Optimizer Dashboard</h1></div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# IMPORTS FOR TABS + DATA
# ---------------------------------------------------------
from utils.data_loader import load_price_data, load_returns_data

from components.tab1_summary import render_summary_tab
from components.tab2_frontier import render_frontier_tab
from components.tab3_weights import render_weights_tab
from components.tab4_sector import render_sector_tab
from components.tab5_drawdown import render_drawdown_tab
from components.tab6_montecarlo import render_montecarlo_tab
from components.tab7_rebalancing import render_rebalancing_tab
from components.tab8_ai_commentary import render_ai_commentary_tab
from components.tab9_buy_analysis import render_buy_analysis_tab

# ---------------------------------------------------------
# SIDEBAR INPUTS
# ---------------------------------------------------------
st.sidebar.header("Input Parameters")

tickers_input = st.sidebar.text_input(
    "Please enter your tickers (comma separated)",
    value=""
)

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

start_date = st.sidebar.date_input("Start Date", value=date(2021, 1, 1))
end_date = st.sidebar.date_input("End Date", value=date.today())

investment_amount = st.sidebar.number_input(
    "Investment Amount ($)",
    min_value=1000,
    value=100000,
    step=1000
)

run_button = st.sidebar.button("Run Analysis")

# ---------------------------------------------------------
# MAIN LOGIC
# ---------------------------------------------------------
if run_button:

    if not tickers:
        st.error("Please enter at least one ticker.")
        st.stop()

try:
    # Load data
    tickers_str = ", ".join(tickers)
    prices = load_price_data(tickers_str, start_date, end_date)
    returns = load_returns_data(tickers_str, start_date, end_date)

    # DEBUG — ADD THESE 3 LINES
    st.write("DEBUG — prices:", prices)
    st.write("DEBUG — returns:", returns)
    st.write("DEBUG — tickers:", tickers)


    # Filter valid tickers
    if isinstance(prices.columns, pd.MultiIndex):
        prices = prices.loc[:, prices.columns.get_level_values("Ticker").isin(tickers)]

    if returns is not None and not returns.empty:
        returns = returns[[t for t in tickers if t in returns.columns]]

    # MUST BE HERE — BEFORE fundamentals
    tickers_final = [t for t in tickers if t in returns.columns]

    if not tickers_final:
        st.error("No valid tickers found in the data.")
        st.stop()

    # Load fundamentals AFTER tickers_final exists
    from utils.fundamentals_loader import load_fundamentals
    fundamentals = load_fundamentals(tickers_final)

    # Portfolio returns
    w_series = pd.Series({t: 1 / len(tickers_final) for t in tickers_final})
    portfolio_returns = (returns[tickers_final] * w_series).sum(axis=1)

    if portfolio_returns.empty or portfolio_returns.isna().all():
        st.error("No valid return data for these tickers. Try different dates.")
        st.stop()

    # Drawdown
    cumulative = (1 + portfolio_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown_df = ((cumulative - running_max) / running_max).to_frame("Drawdown")

    # Monte Carlo
    mu = portfolio_returns.mean()
    sigma = portfolio_returns.std()
    num_paths = 200
    num_days = 252

    sims = np.zeros((num_days, num_paths))
    for p in range(num_paths):
        daily = np.random.normal(mu, sigma, num_days)
        sims[:, p] = np.cumprod(1 + daily)

    mc_df = pd.DataFrame(sims, columns=[f"Path_{i}" for i in range(num_paths)])

    # Performance
    annual_return = portfolio_returns.mean() * 252
    annual_vol = portfolio_returns.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0

    performance = {
        "expected_return": annual_return,
        "volatility": annual_vol,
        "sharpe": sharpe,
    }

    # Model dictionary
    model = {
        "prices": prices,
        "returns": returns,
        "tickers": tickers_final,
        "weights": w_series.to_dict(),
        "investment_amount": investment_amount,
        "drawdown": drawdown_df,
        "monte_carlo": mc_df,
        "performance": performance,
        "fundamentals": fundamentals,
    }

    st.session_state["model"] = model
    st.session_state["prices"] = prices

    st.success("Data loaded successfully.")

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()


    # ---------------------------------------------------------
    # TABS
    # ---------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        " Summary",
        " Efficient Frontier",
        " Optimal Weights",
        " Sector Exposure",
        " Drawdowns",
        " Monte Carlo",
        " Rebalancing",
        " AI Commentary",
        " Buy Analysis"
    ])

    render_summary_tab(tab1, prices, model)
    render_frontier_tab(tab2, prices, model)
    render_weights_tab(tab3, prices, model)
    render_sector_tab(tab4, prices, model)
    render_drawdown_tab(tab5, prices, model)
    render_montecarlo_tab(tab6, prices, model)
    render_rebalancing_tab(tab7, prices, model)
    render_ai_commentary_tab(tab8, prices, model)
    render_buy_analysis_tab(tab9, prices, model)
