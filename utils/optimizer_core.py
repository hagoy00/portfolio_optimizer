import numpy as np
import pandas as pd

# ---------------------------------------------------
# HELPER: Compute portfolio performance
# ---------------------------------------------------
def portfolio_performance(weights, mean_returns, cov_matrix):
    """
    Returns expected portfolio return and volatility.
    """
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
    """
    Simple sector mapping for FAANG-style tickers.
    Extend this as needed.
    """
    sector_map = {
        "AAPL": "Technology",
        "MSFT": "Technology",
        "AMZN": "Consumer Discretionary",
        "GOOG": "Communication Services",
        "META": "Communication Services",
        "TSLA": "Consumer Discretionary"
    }

    df = pd.DataFrame({"Ticker": tickers, "Weight": weights})
    df["Sector"] = df["Ticker"].map(sector_map).fillna("Other")

    return df.groupby("Sector")["Weight"].sum()


# ---------------------------------------------------
# MAIN OPTIMIZER
# ---------------------------------------------------
def run_optimizer(prices, investment_amount=None):
    """
    Core optimizer:
    - Computes returns
    - Computes covariance
    - Generates random portfolios
    - Selects max Sharpe portfolio
    - Computes sector weights
    - Stores investment amount for downstream tabs
    """

    # ---------------------------------------------------
    # VALIDATE INPUT
    # ---------------------------------------------------
    if prices is None or prices.empty:
        return None

    try:
        adj = prices.xs("Adj Close", level=1, axis=1)
    except Exception:
        return None

    returns = adj.pct_change().dropna()

    if returns.empty:
        return None

    tickers = list(adj.columns)
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    # ---------------------------------------------------
    # RANDOM PORTFOLIO SEARCH
    # ---------------------------------------------------
    num_portfolios = 5000
    results = np.zeros((3, num_portfolios))
    weight_records = []

    for i in range(num_portfolios):
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)

        ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)
        sharpe = (ret / vol) if vol > 0 else 0

        results[0, i] = ret
        results[1, i] = vol
        results[2, i] = sharpe
        weight_records.append(weights)

    # ---------------------------------------------------
    # SELECT MAX SHARPE PORTFOLIO
    # ---------------------------------------------------
    max_sharpe_idx = np.argmax(results[2])
    best_weights = weight_records[max_sharpe_idx]

    exp_return = results[0, max_sharpe_idx]
    volatility = results[1, max_sharpe_idx]
    sharpe = results[2, max_sharpe_idx]

    # ---------------------------------------------------
    # SECTOR WEIGHTS
    # ---------------------------------------------------
    sector_weights = compute_sector_weights(best_weights, tickers)

    # ---------------------------------------------------
    # BUILD MODEL DICTIONARY
    # ---------------------------------------------------
    model = {
        "tickers": tickers,
        "weights": best_weights,
        "expected_return": exp_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "sector_weights": sector_weights,
        "returns": returns,
        "cov_matrix": cov_matrix,
        "investment_amount": investment_amount
    }

    return model


# ---------------------------------------------------
# REBALANCING BACKTEST
# ---------------------------------------------------
def rebalancing_backtest(prices, target_weights, rebalance_freq="M"):
    """
    Simple rebalancing backtest:
    - Rebalances monthly or quarterly
    - Computes portfolio value over time
    """

    try:
        adj = prices.xs("Adj Close", level=1, axis=1)
    except Exception:
        return None

    returns = adj.pct_change().dropna()
    if returns.empty:
        return None

    target_weights = np.array(target_weights)
    target_weights /= target_weights.sum()

    portfolio_value = 1.0
    values = []

    for date, row in returns.iterrows():
        portfolio_value *= (1 + np.dot(row.values, target_weights))
        values.append(portfolio_value)

    df = pd.DataFrame({"Portfolio Value": values}, index=returns.index)
    return df
