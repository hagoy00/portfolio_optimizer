import numpy as np
import pandas as pd


# =========================================================
# SAFE PORTFOLIO PERFORMANCE
# =========================================================
def portfolio_performance(weights, mean_returns, cov_matrix):
    """
    Computes annualized return, volatility, and Sharpe ratio.
    Fully crash-proof.
    """
    weights = np.array(weights, dtype=float)

    try:
        ret = float(np.dot(weights, mean_returns) * 252)
    except Exception:
        ret = np.nan

    try:
        vol = float(np.sqrt(weights.T @ (cov_matrix.values * 252) @ weights))
    except Exception:
        vol = np.nan

    sharpe = ret / vol if vol and vol > 0 else np.nan

    return ret, vol, sharpe


# =========================================================
# SAFE RANDOM SEARCH OPTIMIZER
# =========================================================
def safe_random_search(mean_returns, cov_matrix, objective="sharpe", n_iter=5000):
    """
    Random search optimizer with:
    - crash-proof performance evaluation
    - fallback to equal-weight
    - stable objective handling
    """

    n = len(mean_returns)
    best_score = -1e18 if objective == "sharpe" else 1e18
    best_w = None

    for _ in range(n_iter):
        w = np.random.random(n)
        w /= w.sum()

        try:
            ret, vol, sharpe = portfolio_performance(w, mean_returns, cov_matrix)

            if objective == "sharpe":
                if sharpe is not None and sharpe > best_score:
                    best_score = sharpe
                    best_w = w

            elif objective == "vol":
                if vol is not None and vol < best_score:
                    best_score = vol
                    best_w = w

        except Exception:
            continue

    if best_w is None:
        best_w = np.array([1 / n] * n)

    return best_w


# =========================================================
# MAIN OPTIMIZER
# =========================================================
def run_optimizer(returns, cov_matrix):
    """
    Main optimizer entry point.
    Computes:
    - Equal weight portfolio
    - Minimum volatility portfolio
    - Maximum Sharpe portfolio
    Fully crash-proof.
    """

    # Guard clauses
    if returns is None or returns.empty:
        return {"error": "Returns data is empty"}

    if cov_matrix is None or cov_matrix.empty:
        return {"error": "Covariance matrix is empty"}

    tickers = list(returns.columns)
    n = len(tickers)

    # Clean mean returns
    mean_returns = returns.mean().fillna(0)

    # =====================================================
    # EQUAL WEIGHT
    # =====================================================
    w_equal = np.array([1 / n] * n)
    ret_eq, vol_eq, sharpe_eq = portfolio_performance(w_equal, mean_returns, cov_matrix)

    # =====================================================
    # MINIMUM VOLATILITY
    # =====================================================
    w_min = safe_random_search(mean_returns, cov_matrix, objective="vol")
    ret_min, vol_min, sharpe_min = portfolio_performance(w_min, mean_returns, cov_matrix)

    # =====================================================
    # MAXIMUM SHARPE
    # =====================================================
    w_max = safe_random_search(mean_returns, cov_matrix, objective="sharpe")
    ret_max, vol_max, sharpe_max = portfolio_performance(w_max, mean_returns, cov_matrix)

    # =====================================================
    # OUTPUT
    # =====================================================
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
