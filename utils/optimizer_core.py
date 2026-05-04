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
    if prices is None or prices.empty:
        return None

    try:
        adj = prices.xs("Adj Close", level="Field", axis=1)
    except Exception:
        return None

    returns = adj.pct_change().dropna()
    if returns.empty:
        return None

    tickers = list(adj.columns)
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

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

    max_sharpe_idx = np.argmax(results[2])
    best_weights = weight_records[max_sharpe_idx]

    exp_return = results[0, max_sharpe_idx]
    volatility = results[1, max_sharpe_idx]
    sharpe = results[2, max_sharpe_idx]

    sector_weights = compute_sector_weights(best_weights, tickers)

    port_daily_ret = (returns * best_weights).sum(axis=1)
    equity_curve = (1 + port_daily_ret).cumprod()
    drawdown = equity_curve / equity_curve.cummax() - 1.0

    num_paths = 200
    horizon = 252
    mc_paths = []

    for _ in range(num_paths):
        sampled = port_daily_ret.sample(horizon, replace=True).reset_index(drop=True)
        path_curve = (1 + sampled).cumprod()
        mc_paths.append(path_curve)

    mc_df = pd.DataFrame(mc_paths).T

    model = {
        "tickers": tickers,
        "weights": pd.Series(best_weights, index=tickers),
        "expected_return": exp_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "sector_weights": sector_weights,
        "returns": returns,
        "cov_matrix": cov_matrix,
        "investment_amount": investment_amount,
        "equity_curve": equity_curve,
        "drawdown": drawdown,
        "monte_carlo": mc_df,
    }

    return model


# ---------------------------------------------------
# REBALANCING BACKTEST (FIXED)
# ---------------------------------------------------
def rebalancing_backtest(prices, target_weights, freq=None, rebalance_freq=None, **kwargs):
    """
    Simple rebalancing backtest:
    - Accepts freq OR rebalance_freq (or neither)
    - Uses Adj Close from MultiIndex prices
    - Aligns weights to tickers
    """

    if freq is None and rebalance_freq is None:
        freq = "M"
    elif freq is None:
        freq = rebalance_freq

    try:
        adj = prices.xs("Adj Close", level="Field", axis=1)
    except Exception:
        return None

    returns = adj.pct_change().dropna()
    if returns.empty:
        return None

    # Normalize target_weights into a Series aligned to tickers
tickers = adj.columns

if isinstance(target_weights, pd.Series):
    w_series = target_weights.reindex(tickers).fillna(0.0)

elif isinstance(target_weights, dict):
    w_series = pd.Series(target_weights).reindex(tickers).fillna(0.0)

else:
    w_array = np.array(target_weights)
    if len(w_array) != len(tickers):
        raise ValueError("Weight array length does not match tickers")
    w_series = pd.Series(w_array, index=tickers)

# Normalize
if w_series.sum() == 0:
    raise ValueError("Weights sum to zero")

w_series = w_series / w_series.sum()
w = w_series.values

    rebal_dates = returns.resample(freq).first().index

    portfolio_value = 1.0
    values = []
    index_dates = []
    current_weights = w.copy()

    for date in returns.index:
        if date in rebal_dates:
            current_weights = w.copy()

        r = returns.loc[date].values
        portfolio_value *= (1 + np.dot(r, current_weights))

        values.append(portfolio_value)
        index_dates.append(date)

    df = pd.DataFrame({"Portfolio Value": values}, index=index_dates)
    return df
