import streamlit as st
from datetime import date

# Loaders
from utils.data_loader import load_price_data, load_returns_data

# Import tab modules
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
# Streamlit Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Portfolio Optimizer Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 Portfolio Optimizer Dashboard")


# ---------------------------------------------------------
# Sidebar Inputs
# ---------------------------------------------------------
st.sidebar.header("Input Parameters")

tickers_input = st.sidebar.text_input(
    "Tickers (comma separated)",
    value="AAPL, MSFT, NVDA, AMZN"
)

start_date = st.sidebar.date_input(
    "Start Date",
    value=date(2020, 1, 1)
)

end_date = st.sidebar.date_input(
    "End Date",
    value=date.today()
)

investment_amount = st.sidebar.number_input(
    "Investment Amount ($)",
    min_value=1000,
    value=100000,
    step=1000
)

run_button = st.sidebar.button("Run Analysis")


# ---------------------------------------------------------
# Main Logic
# ---------------------------------------------------------
if run_button:

    try:
        # ---------------------------------------------------------
        # Load data
        # ---------------------------------------------------------
        prices = load_price_data(tickers_input, start_date, end_date)
        returns = load_returns_data(tickers_input, start_date, end_date)

        # ---------------------------------------------------------
        # Build model dictionary
        # ---------------------------------------------------------
        tickers = list(prices.columns)
        cov_matrix = returns.cov()

        # TEMPORARY equal weights (until optimizer added)
        weights = {t: 1/len(tickers) for t in tickers}

        model = {
            "prices": prices,
            "returns": returns,
            "tickers": tickers,
            "cov_matrix": cov_matrix,
            "weights": weights,
            "investment_amount": investment_amount,
        }

        st.success("Data loaded successfully.")

        # ---------------------------------------------------------
        # Create Tabs
        # ---------------------------------------------------------
        (
            tab1, tab2, tab3, tab4, tab5,
            tab6, tab7, tab8, tab9
        ) = st.tabs([
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

        # ---------------------------------------------------------
        # Render Tabs
        # ---------------------------------------------------------
        render_summary_tab(tab1, prices, model)
        render_frontier_tab(tab2, prices, model)
        render_weights_tab(tab3, prices, model)
        render_sector_tab(tab4, prices, model)
        render_drawdown_tab(tab5, prices, model)
        render_montecarlo_tab(tab6, prices, model)
        render_rebalancing_tab(tab7, prices, model)
        render_ai_commentary_tab(tab8, prices, model)
        render_buy_analysis_tab(tab9, prices, model)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
