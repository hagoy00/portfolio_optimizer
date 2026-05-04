import streamlit as st
from datetime import date
import pandas as pd
import numpy as np

# Loaders
from utils.data_loader import load_price_data, load_returns_data
st.write("DEBUG — column names:", prices.columns.names)
st.write("DEBUG — reached before tabs")

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

st.title(" Portfolio Optimizer Dashboard")


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

    try:
        
        # ---------------------------------------------------------
        # Load data
        # ---------------------------------------------------------
        prices = load_price_data(tickers_input, start_date, end_date)
        returns = load_returns_data(tickers_input, start_date, end_date)

        # ⭐ DEBUG: Check MultiIndex level names
        st.write("DEBUG — column names:", prices.columns.names)

        # ---------------------------------------------------------
        # Core model components
        # ---------------------------------------------------------
        tickers = list(prices.columns)
        cov_matrix = returns.cov()

        # TEMP: equal weights (replace with optimizer later)
        weights = {t: 1 / len(tickers) for t in tickers}
        w_series = pd.Series(weights)

        # ---------------------------------------------------------
        # Sector Weights (Tab 4)
        # ---------------------------------------------------------
        sector_map = {
            "AAPL": "Technology",
            "MSFT": "Technology",
            "NVDA": "Technology",
            "AMZN": "Consumer Discretionary",
        }
        mapped = {t: sector_map.get(t, "Other") for t in tickers}
        sector_weights = w_series.groupby(mapped).sum()

        # ---------------------------------------------------------
        # Portfolio Returns (used by multiple tabs)
        # ---------------------------------------------------------
        portfolio_returns = (returns * w_series).sum(axis=1)

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
            "return": annual_return,
            "volatility": annual_vol,
            "sharpe": sharpe,
        }

        # ---------------------------------------------------------
        # Build model dictionary
        # ---------------------------------------------------------
        model = {
            "prices": prices,
            "returns": returns,
            "tickers": tickers,
            "cov_matrix": cov_matrix,
            "weights": weights,
            "investment_amount": investment_amount,
            "sector_weights": sector_weights,
            "drawdown": drawdown_df,
            "monte_carlo": mc_df,
            "performance": performance,
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
