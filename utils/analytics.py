import numpy as np
import pandas as pd
import yfinance as yf


# =========================================================
# SAFE RETURN CALCULATION
# =========================================================
def compute_returns(prices):
    """
    Extracts daily returns from a MultiIndex price DataFrame.
    Ensures:
    - 'Close' level exists
    - No empty returns
    - No crashes on missing tickers
    """
    if prices is None or prices.empty:
        return pd.DataFrame()

    try:
        close = prices.xs("Close", level=1, axis=1)
    except Exception:
        return pd.DataFrame()

    returns = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")

    return returns


# =========================================================
# PORTFOLIO MODEL BUILDER
# =========================================================
def build_portfolio_model(prices):
    """
    Builds a portfolio model containing:
    - daily returns
    - covariance matrix
    - equal-weight portfolio (placeholder)
    - performance metrics (expected return, volatility, Sharpe)
    """

    returns = compute_returns(prices)

    if returns.empty:
        return {
            "returns": pd.DataFrame(),
            "cov_matrix": pd.DataFrame(),
            "weights": np.array([]),
            "performance": {
                "expected_return": np.nan,
                "volatility": np.nan,
                "sharpe": np.nan,
            },
        }

    cov_matrix = returns.cov()

    n = len(returns.columns)
    weights = np.array([1 / n] * n)

    # Annualized metrics
    mu = returns.mean()
    expected_return = float(np.sum(mu * weights) * 252)

    try:
        volatility = float(np.sqrt(weights.T @ cov_matrix.values @ weights) * np.sqrt(252))
    except Exception:
        volatility = np.nan

    sharpe = expected_return / volatility if volatility and volatility > 0 else np.nan

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


# =========================================================
# SECTOR WEIGHTS
# =========================================================
def build_sector_weights(weights, tickers):
    """
    Builds sector exposure using Yahoo Finance metadata.
    Fully crash-proof.
    """

    if weights is None or len(weights) == 0:
        return {}

    sectors = {}

    for i, t in enumerate(tickers):
        try:
            info = yf.Ticker(t).info
            sector = info.get("sector", "Unknown")
        except Exception:
            sector = "Unknown"

        sectors[sector] = sectors.get(sector, 0) + float(weights[i])

    return sectors


# =========================================================
# DRAWDOWN
# =========================================================
def compute_drawdown(series):
    """
    Compute drawdown from a portfolio value series.
    Returns a DataFrame with:
    - value
    - peak
    - drawdown
    """

    if series is None or len(series) == 0:
        return pd.DataFrame({"value": [], "peak": [], "drawdown": []})

    series = pd.Series(series).astype(float)
    peak = series.cummax()
    drawdown = (series - peak) / peak

    return pd.DataFrame({
        "value": series,
        "peak": peak,
        "drawdown": drawdown
    })


# =========================================================
# BETA VS SPY
# =========================================================
def compute_beta_vs_spy(prices, ticker):
    """
    Computes beta of a single ticker vs SPY using daily returns.
    Fully crash-proof.
    """

    if prices is None or prices.empty:
        return np.nan

    try:
        close = prices.xs("Close", level=1, axis=1)
    except Exception:
        return np.nan

    if ticker not in close.columns:
        return np.nan

    # Asset returns
    ret = close[ticker].pct_change().dropna()

    if ret.empty:
        return np.nan

    # SPY returns
    try:
        spy = yf.download("SPY", start=close.index.min(), end=close.index.max(), progress=False)
        spy_ret = spy["Adj Close"].pct_change().dropna()
    except Exception:
        return np.nan

    df = pd.concat([ret, spy_ret], axis=1).dropna()
    if df.empty:
        return np.nan

    df.columns = ["asset", "spy"]

    cov = df.cov().iloc[0, 1]
    var = df["spy"].var()

    if var == 0 or np.isnan(var):
        return np.nan

    return float(cov / var)


# =========================================================
# MONTE CARLO SIMULATION
# =========================================================
def run_monte_carlo_simulation(returns, sims=500, horizon=252):
    """
    Runs a Monte Carlo simulation using historical returns.
    Crash-proof, stable, and handles singular covariance matrices.
    """

    if returns is None or returns.empty:
        return pd.DataFrame()

    mu = returns.mean()
    cov = returns.cov()

    # Safe Cholesky
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals[eigvals < 0] = 0
        chol = eigvecs @ np.diag(np.sqrt(eigvals))

    n_assets = len(returns.columns)
    sim_paths = []

    for _ in range(sims):
        rand = np.random.normal(size=(horizon, n_assets))
        shocks = rand @ chol.T
        daily_returns = mu.values + shocks
        path = (1 + daily_returns).cumprod(axis=0)
        portfolio_path = path.mean(axis=1)
        sim_paths.append(portfolio_path)

    df = pd.DataFrame(sim_paths).T
    df.columns = [f"Sim_{i+1}" for i in range(sims)]

    return df
