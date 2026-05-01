import numpy as np
import pandas as pd

print(">>> optimizer_core.py PATH:", __file__)
print(">>> LOADED optimizer_core.py FROM:", __file__)

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
    print(">>> RUN OPTIMIZER CALLED")

    try:
        if prices is None or prices.empty:
            raise ValueError("Price data is empty")

        returns = compute_returns(prices)
        if returns.empty:
            raise ValueError("Returns could not be computed")

        cov = returns.cov()
        if cov.isna().any().any():
            raise ValueError("Covariance matrix contains NaN")

        w = max_sharpe_weights(returns, cov)
        if np.sum(w) == 0:
            raise ValueError("Optimizer produced zero weights")

        rp = risk_parity_weights(cov)

        port_ret = np.dot(w, returns.mean()) * 252
        port_vol = np.sqrt(w @ cov.values @ w) * np.sqrt(252)
        sharpe = port_ret / port_vol if port_vol > 0 else 0

        dd = compute_drawdown(returns @ w)

        mc = monte_carlo_simulation(prices, w)

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
        print(">>> OPTIMIZER ERROR:", e)
        return None

# ---------------------------------------------------
# Rebalancing Backtest
# ---------------------------------------------------
def rebalancing_backtest(prices, weights, freq="ME"):
    try:
        if not isinstance(weights, pd.Series):
            weights = pd.Series(weights, index=prices.columns)

        rets = prices.pct_change().dropna()

        rb_dates = rets.resample(freq).last().index

        port_val = pd.Series(index=rets.index, dtype=float)
        value = 1.0

        current_weights = weights.copy()

        for date in rets.index:
            if date in rb_dates:
                current_weights = weights.copy()

            daily_ret = (rets.loc[date] * current_weights).sum()
            value *= (1 + daily_ret)
            port_val.loc[date] = value

        return port_val.dropna()

    except Exception as e:
        print("REBALANCING ERROR:", e)
        return None
