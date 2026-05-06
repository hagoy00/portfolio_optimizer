import numpy as np
import pandas as pd

# ---------------------------------------------------------
# Utility: Portfolio Performance
# ---------------------------------------------------------
def portfolio_performance(weights, mean_returns, cov_matrix):
    """
    Computes expected return, volatility, and Sharpe ratio.
    """
    weights = np.array(weights)

    ret = np.dot(weights, mean_returns) * 252
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe = ret / vol if vol > 0 else 0

    return ret, vol, sharpe


# ---------------------------------------------------------
# Safe Random Search Optimizer
# ---------------------------------------------------------
def safe_random_search(mean_returns, cov_matrix, objective="sharpe", n_iter=5000):
    n = len(mean_returns)
    best_score = -1e18 if objective == "sharpe" else 1e18
    best_w = None

    for _ in range(n_iter):
        w = np.random.random(n)
        w /= w.sum()

        try:
            ret, vol, sharpe = portfolio_performance(w, mean_returns, cov_matrix)

            if objective == "sharpe":
                if sharpe > best_score:
                    best_score = sharpe
                    best_w = w

            elif objective == "vol":
                if vol < best_score:
                    best_score = vol
                    best_w = w

        except Exception:
            continue

    # Fallback if search fails
    if best_w is None:
        best_w = np.array([1 / n] * n)

    return best_w


# ---------------------------------------------------------
# MAIN OPTIMIZER FUNCTION
# ---------------------------------------------------------
def run_optimizer(returns, cov_matrix):
    """
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

    # Clean mean returns
    mean_returns = returns.mean().fillna(0)

    # Equal weight
    w_equal = np.array([1 / n] * n)
    ret_eq, vol_eq, sharpe_eq = portfolio_performance(w_equal, mean_returns, cov_matrix)

    # Min volatility
    w_min = safe_random_search(mean_returns, cov_matrix, objective="vol")
    ret_min, vol_min, sharpe_min = portfolio_performance(w_min, mean_returns, cov_matrix)

    # Max Sharpe
    w_max = safe_random_search(mean_returns, cov_matrix, objective="sharpe")
    ret_max, vol_max, sharpe_max = portfolio_performance(w_max, mean_returns, cov_matrix)

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
