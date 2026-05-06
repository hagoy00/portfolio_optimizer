import numpy as np
import pandas as pd

# ---------------------------------------------------------
# Utility: Portfolio Performance
# ---------------------------------------------------------
def portfolio_performance(weights, mean_returns, cov_matrix):
    """
    Computes expected return, volatility, and Sharpe ratio.
    """
    ret = np.dot(weights, mean_returns) * 252
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe = ret / vol if vol > 0 else 0
    return ret, vol, sharpe


# ---------------------------------------------------------
# Optimization Helpers
# ---------------------------------------------------------
def min_volatility(mean_returns, cov_matrix):
    """
    Computes the minimum-volatility portfolio using a simple grid search.
    """
    n = len(mean_returns)
    best_vol = 1e9
    best_w = None

    # Simple grid search (fast for small ticker sets)
    for _ in range(5000):
        w = np.random.random(n)
        w /= w.sum()
        _, vol, _ = portfolio_performance(w, mean_returns, cov_matrix)
        if vol < best_vol:
            best_vol = vol
            best_w = w

    return best_w


def max_sharpe_ratio(mean_returns, cov_matrix):
    """
    Computes the maximum Sharpe portfolio using random search.
    """
    n = len(mean_returns)
    best_sharpe = -1e9
    best_w = None

    for _ in range(5000):
        w = np.random.random(n)
        w /= w.sum()
        _, _, sharpe = portfolio_performance(w, mean_returns, cov_matrix)
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_w = w

    return best_w


# ---------------------------------------------------------
# MAIN OPTIMIZER FUNCTION (REQUIRED BY app.py)
# ---------------------------------------------------------
def run_optimizer(returns, cov_matrix):
    """
    Main optimizer used by the dashboard.
    Computes:
    - Equal weight portfolio
    - Minimum volatility portfolio
    - Maximum Sharpe portfolio
    """

    if returns is None or returns.empty:
        return {"error": "Returns data is empty"}

    if cov_matrix is None or cov_matrix.empty:
        return {"error": "Covariance matrix is empty"}

    tickers = list(returns.columns)
    n = len(tickers)

    mean_returns = returns.mean()

    # Equal weight
    w_equal = np.array([1 / n] * n)
    ret_eq, vol_eq, sharpe_eq = portfolio_performance(w_equal, mean_returns, cov_matrix)

    # Min volatility
    w_min = min_volatility(mean_returns, cov_matrix)
    ret_min, vol_min, sharpe_min = portfolio_performance(w_min, mean_returns, cov_matrix)

    # Max Sharpe
    w_max = max_sharpe_ratio(mean_returns, cov_matrix)
    ret_max, vol_max, sharpe_max = portfolio_performance(w_max, mean_returns, cov_matrix)

    # Return clean dictionary for Streamlit
    return {
        "tickers": tickers,

        "equal_weight": {
            "weights": w_equal,
            "expected_return": ret_eq,
            "volatility": vol_eq,
            "sharpe": sharpe_eq,
        },

        "min_volatility": {
            "weights": w_min,
            "expected_return": ret_min,
            "volatility": vol_min,
            "sharpe": sharpe_min,
        },

        "max_sharpe": {
            "weights": w_max,
            "expected_return": ret_max,
            "volatility": vol_max,
            "sharpe": sharpe_max,
        }
    }
