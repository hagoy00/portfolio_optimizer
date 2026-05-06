import numpy as np
import pandas as pd


def rebalancing_backtest(prices, target_weights, freq="M", rf_rate=0.0):
    """
    Institutional-grade rebalancing backtest.

    Returns:
        {
            "equity_curve": DataFrame[Portfolio Value],
            "metrics": dict,
            "trades": DataFrame (optional, simple turnover info)
        }
    """
# ---------------------------------------------------------
# SIMPLE INSTITUTIONAL OPTIMIZER (REQUIRED BY app.py)
# ---------------------------------------------------------
# ---------------------------------------------------------
# SIMPLE INSTITUTIONAL OPTIMIZER (REQUIRED BY app.py)
# ---------------------------------------------------------

def run_optimizer(returns, cov):
    """
    Minimal working optimizer:
    Computes equal weights, max-sharpe, and min-vol portfolios.
    This ensures the Optimizer tab and Run Analysis button work.
    """

    tickers = list(returns.columns)
    n = len(tickers)

    # Equal weights
    equal_weights = np.array([1/n] * n)

    # Expected returns (annualized)
    mu = returns.mean() * 252

    # Volatility (annualized)
    vol = np.sqrt(np.diag(cov) * 252)

    # Sharpe ratio (risk-free = 0)
    sharpe = mu / vol.replace(0, np.nan)

    # Max Sharpe portfolio (1-stock version)
    max_sharpe_idx = sharpe.idxmax()
    max_sharpe_port = np.zeros(n)
    max_sharpe_port[tickers.index(max_sharpe_idx)] = 1.0

    # Min Vol portfolio (1-stock version)
    min_vol_idx = vol.argmin()
    min_vol_port = np.zeros(n)
    min_vol_port[min_vol_idx] = 1.0

    return {
        "tickers": tickers,
        "equal_weights": equal_weights,
        "max_sharpe_port": max_sharpe_port,
        "min_vol_port": min_vol_port,
        "expected_returns": mu.to_dict(),
        "volatility": vol.tolist(),
        "sharpe": sharpe.tolist(),
    }
    # ---------- 1. Normalize frequency ----------
    freq_map = {
        "W": "W-FRI",
        "Weekly": "W-FRI",
        "M": "M",
        "ME": "M",
        "Monthly": "M",
        "Q": "Q-DEC",
        "QE": "Q-DEC",
        "Quarterly": "Q-DEC",
        "Y": "A-DEC",
        "YE": "A-DEC",
        "Annual": "A-DEC",
    }
    safe_freq = freq_map.get(freq, "M")

    # ---------- 2. Extract Adj Close ----------
    try:
        adj = prices.xs("Adj Close", level="Field", axis=1)
    except Exception:
        return None

    returns = adj.pct_change().dropna()
    if returns.empty:
        return None

    # ---------- 3. Align & normalize weights ----------
    tickers = adj.columns
    w = target_weights.reindex(tickers).fillna(0.0)
    if w.sum() == 0:
        return None
    w = w / w.sum()

    # ---------- 4. Rebalancing dates ----------
    try:
        rb_dates = returns.resample(safe_freq).last().index
    except Exception:
        return None

    # ---------- 5. Backtest loop ----------
    port_val = []
    weights_history = []
    value = 1.0
    current_weights = w.copy()
    last_weights = w.copy()
    turnover_list = []
    rebalance_count = 0

    for dt in returns.index:
        # Rebalance on scheduled dates
        if dt in rb_dates:
            rebalance_count += 1
            # turnover = 0.5 * sum(|w_new - w_old|)
            turnover = 0.5 * np.abs(current_weights - last_weights).sum()
            turnover_list.append((dt, turnover))
            last_weights = current_weights.copy()
            current_weights = w.copy()

        daily_ret = float((returns.loc[dt] * current_weights).sum())
        value *= (1.0 + daily_ret)

        port_val.append((dt, value))
        weights_history.append((dt, current_weights.copy()))

    equity = pd.DataFrame(port_val, columns=["Date", "Portfolio Value"]).set_index("Date")

    # ---------- 6. Metrics ----------
    daily_ret_series = equity["Portfolio Value"].pct_change().dropna()
    if daily_ret_series.empty:
        return None

    # CAGR
    days = (equity.index[-1] - equity.index[0]).days
    years = days / 365.25 if days > 0 else 0
    if years > 0:
        cagr = (equity["Portfolio Value"].iloc[-1] / equity["Portfolio Value"].iloc[0]) ** (1 / years) - 1
    else:
        cagr = np.nan

    # Volatility (annualized)
    vol = daily_ret_series.std() * np.sqrt(252)

    # Sharpe (using rf_rate as annual)
    if vol > 0 and not np.isnan(vol):
        excess_ret = cagr - rf_rate
        sharpe = excess_ret / vol
    else:
        sharpe = np.nan

    # Max drawdown
    roll_max = equity["Portfolio Value"].cummax()
    dd = equity["Portfolio Value"] / roll_max - 1.0
    max_dd = dd.min()

    # Turnover summary
    if turnover_list:
        turnover_df = pd.DataFrame(turnover_list, columns=["Date", "Turnover"]).set_index("Date")
        avg_turnover = turnover_df["Turnover"].mean()
    else:
        turnover_df = pd.DataFrame(columns=["Turnover"])
        avg_turnover = np.nan

    metrics = {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "Rebalance Count": rebalance_count,
        "Average Turnover": avg_turnover,
    }

    return {
        "equity_curve": equity,
        "metrics": metrics,
        "turnover": turnover_df,
    }
