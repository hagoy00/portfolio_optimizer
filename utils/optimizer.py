import numpy as np
import pandas as pd

# ---------------------------------------------------
# Compute Returns
# ---------------------------------------------------
def compute_returns(prices):
    return prices.pct_change().dropna()

# ---------------------------------------------------
# Compute Drawdown
# ---------------------------------------------------
def compute_drawdown(series):
    cum = (1 + series).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd

# ---------------------------------------------------
# Risk Parity Weights
# ---------------------------------------------------
def risk_parity_weights(cov):
    inv_vol = 1 / np.sqrt(np.diag(cov))
    w = inv_vol / inv_vol.sum()
    return w

# ---------------------------------------------------
# Max Sharpe Weights
# ---------------------------------------------------
def max_sharpe_weights(returns, cov):
    mean_ret = returns.mean()
    inv_cov = np.linalg.pinv(cov)
    w = inv_cov @ mean_ret
    w = np.maximum(w, 0)
    w = w / w.sum()
    return w

# ---------------------------------------------------
# Monte Carlo Simulation
# ---------------------------------------------------
def monte_carlo_simulation(prices, weights, n_sims=200, horizon=252):
    returns = compute_returns(prices)
    mu = returns.mean().values
    cov = returns.cov().values

    sims = []
    for _ in range(n_sims):
        path = [1]
        for _ in range(horizon):
            daily = np.random.multivariate_normal(mu, cov)
            path.append(path[-1] * (1 + np.dot(weights, daily)))
        sims.append(path)

    return pd.DataFrame(sims).T

# ---------------------------------------------------
# Main Optimizer
# ---------------------------------------------------
def run_optimizer(prices):
    try:
        returns = compute_returns(prices)
        cov = returns.cov()

        # Max Sharpe
        w = max_sharpe_weights(returns, cov)

        # Risk Parity
        rp = risk_parity_weights(cov)

        # Performance
        port_ret = np.dot(w, returns.mean()) * 252
        port_vol = np.sqrt(w @ cov.values @ w) * np.sqrt(252)
        sharpe = port_ret / port_vol if port_vol > 0 else 0

        # Drawdown
        dd = compute_drawdown(returns @ w)

        # Monte Carlo
        mc = monte_carlo_simulation(prices, w)

        # Sector weights placeholder
        sector_weights = {t: 1/len(w) for t in prices.columns}

        return {
            "weights": pd.Series(w, index=prices.columns),
            "risk_parity": pd.Series(rp, index=prices.columns),
            "performance": {
                "return": port_ret,
                "volatility": port_vol,
                "sharpe": sharpe
            },
            "drawdown": dd,
            "montecarlo": mc,
            "sector_weights": sector_weights
        }

    except Exception as e:
        print("OPTIMIZER ERROR:", e)
        return None
