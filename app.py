import streamlit as st
import pandas as pd
import numpy as np
# ... all your other imports ...

# ⬇️ PASTE THE STICKY HEADER CSS RIGHT HERE
st.markdown("""
<style>
#portfolio-header {
    position: sticky;
    top: 0;
    background-color: #0E1117;
    padding: 18px 0px 18px 0px;
    z-index: 9999;
    border-bottom: 1px solid #1F2937;
}
#portfolio-header h1 {
    color: #4DA8FF !important;
    font-size: 32px;
    font-weight: 700;
    margin: 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div id="portfolio-header"><h1>Portfolio Optimizer Dashboard</h1></div>', unsafe_allow_html=True)

# ⬇️ Your sidebar, tabs, and page layout come AFTER this
st.sidebar.title("Navigation")


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

# ---------------------------------------------------------
# Global UI styling (FINAL sticky + blue)
# ---------------------------------------------------------
st.markdown("""
<style>

/* --- FORCE BLUE TITLE (Overrides Streamlit Theme) --- */
div[data-testid="stHeader"] h1,
div[data-testid="stHeader"] h2,
div[data-testid="stHeader"] h3 {
    color: #1E90FF !important;
}

/* Also force all headings everywhere */
h1, h2, h3, h4, h5, h6 {
    color: #1E90FF !important;
}

/* --- TRUE STICKY HEADER (No movement, no flicker) --- */
div[data-testid="stHeader"] {
    position: sticky !important;
    top: 0;
    background-color: white !important;
    z-index: 9999 !important;
    border-bottom: 1px solid #e0e0e0;
    padding-top: 8px;
    padding-bottom: 8px;
}

/* Prevent Streamlit from collapsing the header container */
header[data-testid="stHeader"] {
    height: auto !important;
}

/* Sticky sidebar */
section[data-testid="stSidebar"] {
    position: fixed !important;
    top: 0;
    left: 0;
    height: 100%;
    z-index: 100;
}

/* Push main content right */
div[data-testid="stAppViewContainer"] {
    margin-left: 18rem !important;
}

/* --- BLUE SIDEBAR TEXT --- */
section[data-testid="stSidebar"] * {
    color: #1E90FF !important;
}

/* --- BLUE TAB LABELS --- */
.stTabs [data-baseweb="tab"] {
    color: #1E90FF !important;
}

/* --- BLUE METRIC VALUES --- */
[data-testid="stMetricValue"] {
    color: #1E90FF !important;
}

/* ---------------------------------------------------------
   FINAL FIX — Make title sticky inside the REAL scroll container
   --------------------------------------------------------- */
div[data-testid="stAppViewBlockContainer"] h1 {
    position: sticky !important;
    top: 0 !important;
    background-color: white !important;
    padding: 14px 0 !important;
    margin: 0 !important;
    z-index: 999999 !important;
    border-bottom: 1px solid #e0e0e0 !important;
}

</style>
""", unsafe_allow_html=True)

st.title("Portfolio Optimizer Dashboard")


# ---------------------------------------------------------
# Sidebar Inputs
# ---------------------------------------------------------
st.sidebar.header("Input Parameters")

tickers_input = st.sidebar.text_input(
    "Tickers (comma separated)",
    value="AAPL, MSFT, NVDA, AMZN"
)

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

start_date = st.sidebar.date_input(
    "Start Date",
    value=date(2021, 1, 1)
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

    if not tickers:
        st.error("Please enter at least one ticker.")
    else:
        try:
            # ---------------------------------------------------------
            # Load data
            # ---------------------------------------------------------
            tickers_str = ", ".join(tickers)
            prices = load_price_data(tickers_str, start_date, end_date)
            returns = load_returns_data(tickers_str, start_date, end_date)

            # Safety filter
            if isinstance(prices.columns, pd.MultiIndex) and "Ticker" in prices.columns.names:
                prices = prices.loc[:, prices.columns.get_level_values("Ticker").isin(tickers)]
            if returns is not None and not returns.empty:
                returns = returns[[t for t in tickers if t in returns.columns]]

            tickers_final = [t for t in tickers if t in returns.columns]

            if not tickers_final:
                st.error("No valid tickers found in the data.")
            else:
                # ---------------------------------------------------------
                # Core model components
                # ---------------------------------------------------------
                cov_matrix = returns.cov()

                weights = {t: 1 / len(tickers_final) for t in tickers_final}
                w_series = pd.Series(weights)

                # ---------------------------------------------------------
                # Sector Weights (Tab 4)
                # ---------------------------------------------------------
                sector_map = {
                    "AAPL": "Technology",
                    "MSFT": "Technology",
                    "NVDA": "Technology",
                    "AMZN": "Consumer Discretionary",
                    "GOOG": "Communication Services",
                    "TSLA": "Consumer Discretionary",
                    "WFC": "Financials",
                }

                mapped = {t: sector_map.get(t, "Other") for t in tickers_final}
                sector_weights = w_series.groupby(mapped).sum()

                # ---------------------------------------------------------
                # Portfolio Returns
                # ---------------------------------------------------------
                portfolio_returns = (returns[tickers_final] * w_series).sum(axis=1)

                # ---------------------------------------------------------
                # Drawdown (Tab 5)
                # ---------------------------------------------------------
                cumulative = (1 + portfolio_returns).cumprod()
                running_max = cumulative.cummax()
                drawdown = (cumulative - running_max) / running_max
                drawdown_df = drawdown.to_frame("Drawdown")

                # ---------------------------------------------------------
                # Monte Carlo Simulation (Tab 6)
                # ---------------------------------------------------------
                num_paths = 200
                num_days = 252

                mu = portfolio_returns.mean()
                sigma = portfolio_returns.std()

                simulations = np.zeros((num_days, num_paths))
                for p in range(num_paths):
                    daily_returns = np.random.normal(mu, sigma, num_days)
                    simulations[:, p] = np.cumprod(1 + daily_returns)

                mc_df = pd.DataFrame(
                    simulations,
                    columns=[f"Path_{i}" for i in range(num_paths)]
                )

                # ---------------------------------------------------------
                # Performance Metrics (Tab 8)
                # ---------------------------------------------------------
                annual_return = portfolio_returns.mean() * 252
                annual_vol = portfolio_returns.std() * np.sqrt(252)
                sharpe = annual_return / annual_vol if annual_vol > 0 else 0

                performance = {
                    "expected_return": annual_return,
                    "volatility": annual_vol,
                    "sharpe": sharpe,
                }

                # ---------------------------------------------------------
                # Build model dictionary
                # ---------------------------------------------------------
                model = {
                    "prices": prices,
                    "returns": returns,
                    "tickers": tickers_final,
                    "cov_matrix": cov_matrix,
                    "weights": weights,
                    "investment_amount": investment_amount,
                    "sector_weights": sector_weights,
                    "drawdown": drawdown_df,
                    "monte_carlo": mc_df,
                    "performance": performance,
                }

                st.session_state["model"] = model
                st.session_state["prices"] = prices

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
                # Render Tabs (Architecture B)
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
