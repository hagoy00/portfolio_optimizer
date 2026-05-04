import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize


def render_frontier_tab(tab, prices, model):
    tab.markdown("## Efficient Frontier")

    if model is None:
        tab.info("Run analysis first.")
        return

    returns = model.get("returns")
    cov = model.get("cov_matrix")
    tickers = model.get("tickers")

    if returns is None or cov is None:
        tab.error("Missing returns or covariance matrix.")
        return

    # ---------------------------------------------------
    # Prepare data
    # ---------------------------------------------------
    mean_returns = returns.mean() * 252
    cov_matrix = cov * 252
    n = len(tickers)

    # ---------------------------------------------------
    # Optimization helpers
    # ---------------------------------------------------
    def portfolio_volatility(weights):
        return np.sqrt(weights.T @ cov_matrix @ weights)

    def portfolio_return(weights):
        return np.dot(weights, mean_returns)

    def min_vol_for_target(target):
        """Minimize volatility for a given target return."""
        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w: portfolio_return(w) - target},
        )
        bounds = tuple((0, 1) for _ in range(n))
        w0 = np.ones(n) / n

        result = minimize(
            portfolio_volatility,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if not result.success:
            return None, None

        return result.x, portfolio_volatility(result.x)

    # ---------------------------------------------------
    # Compute Efficient Frontier
    # ---------------------------------------------------
    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), 50)

    frontier_vols = []
    frontier_rets = []

    for target in target_returns:
        w_opt, vol = min_vol_for_target(target)
        if vol is not None:
            frontier_rets.append(target)
            frontier_vols.append(vol)

    if len(frontier_rets) == 0:
        tab.warning("Could not compute efficient frontier.")
        return

    df = pd.DataFrame({
        "Return": frontier_rets,
        "Volatility": frontier_vols
    })

    # ---------------------------------------------------
    # Display Chart
    # ---------------------------------------------------
    tab.markdown("### Efficient Frontier Curve")
    tab.line_chart(df.set_index("Volatility"))

    # ---------------------------------------------------
    # Display Table (optional)
    # ---------------------------------------------------
    with tab.expander("Show Frontier Data"):
        tab.dataframe(df.style.format({"Return": "{:.2%}", "Volatility": "{:.2%}"}))
