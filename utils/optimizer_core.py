import numpy as np
import pandas as pd

print(">>> USING optimizer_core.py FROM:", __file__)

# ---------------------------------------------------
# HELPER: Compute portfolio performance
# ---------------------------------------------------
def portfolio_performance(weights, mean_returns, cov_matrix):
    weights = np.array(weights)
    port_return = np.sum(mean_returns * weights) * 252
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return port_return, port_vol


# ---------------------------------------------------
# HELPER: Compute Sharpe Ratio
# ---------------------------------------------------
def sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
    ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)
    return (ret - risk_free_rate) / vol if vol > 0 else 0


# ---------------------------------------------------
# HELPER: Sector Weights
# ---------------------------------------------------
def compute_sector_weights(weights, tickers):
    sector
