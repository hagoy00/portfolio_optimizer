import streamlit as st
from datetime import date

# Loaders
#from utils.data_loader import load_price_data, load_returns_data

# Import all tab modules
from components.tab1_summary import render_tab as tab1_render
from components.tab2_frontier import render_tab as tab2_render
from components.tab3_weights import render_tab as tab3_render
from components.tab4_sector import render_tab as tab4_render
from components.tab5_drawdown import render_tab as tab5_render
from components.tab6_montecarlo import render_tab as tab6_render
from components.tab7_rebalancing import render_tab as tab7_render
from components.tab8_ai_commentary import render_tab as tab8_render
from components.tab9_buy_analysis import render_tab as tab9_render


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

run_button = st.sidebar.button("Run Analysis")


# ---------------------------------------------------------
# Main Logic
# ---------------------------------------------------------
if run_button:

    try:
        # Load data
        prices = load_price_data(tickers_input, start_date, end_date)
        returns = load_returns_data(tickers_input, start_date, end_date)

        st.success("Data loaded successfully.")

        # ---------------------------------------------------------
        # Create 9 tabs
        # ---------------------------------------------------------
        (
            tab1, tab2, tab3, tab4, tab5,
            tab6, tab7, tab8, tab9
        ) = st.tabs([
            "📊 Summary",
            "📈 Efficient Frontier",
            "⚖️ Optimal Weights",
            "🏭 Sector Exposure",
            "📉 Drawdowns",
            "🎲 Monte Carlo",
            "🔄 Rebalancing",
            "🤖 AI Commentary",
            "🛒 Buy Analysis"
        ])

        # ---------------------------------------------------------
        # Render each tab
        # ---------------------------------------------------------
        with tab1:
            tab1_render(prices, returns)

        with tab2:
            tab2_render(prices, returns)

        with tab3:
            tab3_render(prices, returns)

        with tab4:
            tab4_render(prices, returns)

        with tab5:
            tab5_render(prices, returns)

        with tab6:
            tab6_render(prices, returns)

        with tab7:
            tab7_render(prices, returns)

        with tab8:
            tab8_render(prices, returns)

        with tab9:
            tab9_render(prices, returns)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
