import streamlit as st
import pandas as pd
import numpy as np

def render_frontier_tab(tab, prices, model):
    tab.markdown("## Efficient Frontier")

    if model is None:
        tab.info("Run optimization first.")
        return

    returns = model.get("returns")
    cov = model.get("cov_matrix")
    tickers = model.get("tickers")

    if returns is None or cov is None:
        tab.error("Missing returns or covariance matrix.")
        return

    # ---------------------------------------------------
    # Compute Efficient Frontier
    # ---------------------------------------------------
    num_points = 50
    frontier_returns = []
    frontier_vols = []

    mean_returns = returns.mean()

    for target_return in np.linspace(mean_returns.min(), mean_returns.max(), num_points):
        best_vol = None

        for _ in range(2000):
            w = np.random.random(len(tickers))
            w /= w.sum()

            port_ret = np.sum(w * mean_returns) * 252
            if abs(port_ret - target_return * 252) < 0.002:
                port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w))) * np.sqrt(252)

                if best_vol is None or port_vol < best_vol:
                    best_vol = port_vol

        if best_vol is not None:
            frontier_returns.append(target_return * 252)
            frontier_vols.append(best_vol)

    if not frontier_returns:
        tab.warning("Could not compute efficient frontier.")
        return

    df = pd.DataFrame({
        "Return": frontier_returns,
        "Volatility": frontier_vols
    })

    tab.markdown("### Efficient Frontier Curve")
    tab.line_chart(df.set_index("Volatility"))
