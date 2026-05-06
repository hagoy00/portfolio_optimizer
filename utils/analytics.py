import numpy as np
import pandas as pd
import yfinance as yf


# ---------------------------------------------------------
# Build portfolio model (returns, covariance, weights, metrics)
# ---------------------------------------------------------
def build_portfolio_model(prices):
    """
    Builds a portfolio model containing:
    - daily returns
    - covariance matrix
    - optimized weights (equal-weight for now)
    - performance metrics (expected return, volatility, Sharpe) 
    """

    # Daily returns
    returns = prices.xs("Close", level=1, axis=1).pct_change().dropna()

    # Covariance matrix
    cov_matrix = returns.cov()

    # Equal-weight portfolio (placeholder for optimizer)
    n = len(returns.columns)
    weights = np.array([1 / n] * n)

    # Annualized metrics
    expected_return = np.sum(returns.mean() * weights) * 252
    volatility = np.sqrt(weights.T @ cov_matrix.values @ weights) * 252**0.5
    sharpe = expected_return / volatility if volatility > 0 else 0

    return {
        "returns": returns,
        "cov_matrix": cov_matrix,
        "weights": weights,
        "performance": {
            "expected_return": expected_return,
            "volatility": volatility,
            "sharpe": sharpe,
        },
    }


# ---------------------------------------------------------
# Sector weights builder
# ---------------------------------------------------------
def build_sector_weights(weights, tickers):
    """
    Builds sector exposure using Yahoo Finance metadata.
    """

    sectors = {}
    for i, t in enumerate(tickers):
        try:
            info = yf.Ticker(t).info
            sector = info.get("sector", "Unknown")
        except Exception:
            sector = "Unknown"

        sectors[sector] = sectors.get(sector, 0) + weights[i]

    return sectors

# ---------------------------------------------------------
# Compute drawdown series
# ---------------------------------------------------------
def compute_drawdown(series):
    """
    Compute drawdown from a portfolio value series.
    Input: pandas Series (portfolio value over time)
    Output: DataFrame with columns:
        - value
        - peak
        - drawdown
    """
    # Ensure Series
    series = series.astype(float)

    # Compute running peak
    peak = series.cummax()

    # Compute drawdown
    drawdown = (series - peak) / peak

    return pd.DataFrame({
        "value": series,
        "peak": peak,
        "drawdown": drawdown
    })


# ---------------------------------------------------------
# Compute Beta vs SPY
# ---------------------------------------------------------
def compute_beta_vs_spy(prices, ticker):
    """
    Computes beta of a single ticker vs SPY using daily returns.
    Returns:
        beta (float)
    """

    # Extract close prices
    close = prices.xs("Close", level=1, axis=1)

    if ticker not in close.columns:
        return None

    # Download SPY benchmark
    spy = yf.download("SPY", start=close.index.min(), end=close.index.max(), progress=False)
    spy_ret = spy["Adj Close"].pct_change().dropna()

    # Ticker returns
    ret = close[ticker].pct_change().dropna()

    # Align dates
    df = pd.concat([ret, spy_ret], axis=1).dropna()
    df.columns = ["asset", "spy"]

    # Covariance / variance formula
    cov = df.cov().iloc[0, 1]
    var = df["spy"].var()

    beta = cov / var if var != 0 else 0

    return beta

# ---------------------------------------------------------
# MONTE CARLO SIMULATION ENGINE (REQUIRED BY app.py)
# ---------------------------------------------------------

import pandas as pd
import numpy as np

def run_monte_carlo_simulation(returns, sims=500, horizon=252):
    """
    Simple Monte Carlo simulation using:
    - Historical mean
    - Historical volatility
    - Geometric Brownian Motion (GBM)
    """

    # Portfolio returns (equal weight)
    n = len(returns.columns)
    weights = np.array([1/n] * n)
    port_ret = returns.dot(weights)

    mu = port_ret.mean()
    sigma = port_ret.std()

    # Simulations
    simulations = []

    for _ in range(sims):
        prices = [1.0]
        for _ in range(horizon):
            drift = mu - 0.5 * sigma**2
            shock = sigma * np.random.normal()
            prices.append(prices[-1] * np.exp(drift + shock))
        simulations.append(prices)

    df = pd.DataFrame(simulations).T
    df.index.name = "Day"
    return df
