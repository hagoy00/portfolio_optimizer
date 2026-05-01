import numpy as np
import pandas as pd

# ---------------------------------------------------
# BASIC RETURN & RISK FUNCTIONS
# ---------------------------------------------------

def compute_returns(prices):
    return prices.pct_change().dropna()

def compute_annualized_return(returns):
    return returns.mean() * 252

def compute_annualized_volatility(returns):
    return returns.std() * np.sqrt(252)

def compute_covariance(returns):
    return returns.cov() * 252


# ---------------------------------------------------
# RANDOM PORTFOLIOS FOR FRONTIER
# ---------------------------------------------------

def random_portfolios(returns, n=2000):
    tickers = returns.columns
    cov = compute_covariance(returns)
    mu = compute_annualized_return(returns)

    results = []
    weights_list = []

    for _ in range(n):
        w = np.random.random(len(tickers))
        w /= w.sum()

        port_return = np.dot(w, mu)
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        sharpe = port_return / port_vol if port_vol > 0 else 0

        results.append([port_return, port_vol, sharpe])
        weights_list.append(w)

    df = pd.DataFrame(results, columns=["return", "volatility", "sharpe"])
    return df, weights_list, tickers


# ---------------------------------------------------
# OPTIMAL PORTFOLIOS
# ---------------------------------------------------

def max_sharpe_portfolio(returns):
    df, weights_list, tickers = random_portfolios(returns)
    idx = df["sharpe"].idxmax()
    return df.iloc[idx], dict(zip(tickers, weights_list[idx]))

def min_vol_portfolio(returns):
    df, weights_list, tickers = random_portfolios(returns)
    idx = df["volatility"].idxmin()
    return df.iloc[idx], dict(zip(tickers, weights_list[idx]))

def compute_frontier(returns, n=2000):
    df, _, _ = random_portfolios(returns, n=n)
    return df


# ---------------------------------------------------
# RISK PARITY
# ---------------------------------------------------

def risk_parity_weights(returns):
    cov = compute_covariance(returns)
    inv_vol = 1 / np.sqrt(np.diag(cov))
    w = inv_vol / inv_vol.sum()
    return dict(zip(returns.columns, w))


# ---------------------------------------------------
# PORTFOLIO PERFORMANCE
# ---------------------------------------------------

def portfolio_performance(weights, returns):
    w = np.array(list(weights.values()))
    mu = compute_annualized_return(returns)
    cov = compute_covariance(returns)

    port_return = np.dot(w, mu)
    port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
    sharpe = port_return / port_vol if port_vol > 0 else 0

    return {
        "return": port_return,
        "volatility": port_vol,
        "sharpe": sharpe
    }


# ---------------------------------------------------
# DRAWDOWN
# ---------------------------------------------------

def compute_drawdown(prices):
    cum = (1 + prices.pct_change()).cumprod()
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max
    return dd


# ---------------------------------------------------
# MONTE CARLO
# ---------------------------------------------------

def monte_carlo_simulation(prices, weights=None, n_sims=50, horizon=252):
    returns = prices.pct_change().dropna()

    if weights is None:
        w = np.ones(len(returns.columns)) / len(returns.columns)
    else:
        w = np.array(list(weights.values()))

    port_ret = returns.dot(w)

    mu = port_ret.mean()
    sigma = port_ret.std()

    sims = []
    for _ in range(n_sims):
        sim = np.random.normal(mu, sigma, horizon)
        sims.append((1 + sim).cumprod())

    return pd.DataFrame(sims).T


# ---------------------------------------------------
# FIXED & STABLE REBALANCING BACKTEST (ME, W, Q)
# ---------------------------------------------------

def rebalancing_backtest(prices, weights, freq="ME"):
    """
    Supports:
    - ME = Month-End
    - W  = Weekly
    - Q  = Quarterly (converted to QE)
    """

    # Normalize frequency
    if freq == "M":
        freq = "ME"
    if freq == "Q":
        freq = "QE"  # Pandas 2.2+ requirement

    returns = prices.pct_change().dropna()
    w = np.array(list(weights.values()))

    portfolio = []
    index_list = []

    # Group returns by frequency
    try:
        grouped = returns.groupby(pd.Grouper(freq=freq))
    except Exception as e:
        print("Grouping error:", e)
        return pd.Series(dtype=float)

    for period_end, group in grouped:
        if group.empty:
            continue

        period_ret = group.dot(w)
        cumulative = (1 + period_ret).cumprod()

        portfolio.extend(cumulative.values)
        index_list.extend(cumulative.index)

    if not portfolio:
        return pd.Series(dtype=float)

    return pd.Series(portfolio, index=index_list)


# ---------------------------------------------------
# SECTOR EXPOSURE
# ---------------------------------------------------

def compute_sector_exposure(weights, sector_map):
    exposure = {}
    for ticker, w in weights.items():
        sector = sector_map.get(ticker, "Unknown")
        exposure[sector] = exposure.get(sector, 0) + w
    return exposure


# ---------------------------------------------------
# BUY SCORE (USED IN BUY ANALYSIS)
# ---------------------------------------------------

def buy_score(prices):
    returns = prices.pct_change().dropna()
    trend = prices.iloc[-50:].pct_change().mean()
    vol = returns.std()
    score = (trend - vol).rank(pct=True)
    return score.to_dict()
