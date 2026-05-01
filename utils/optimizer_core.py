print(">>> LOADED optimizer_core.py FROM:", __file__)
def run_optimizer(prices):
    print(">>> RUN OPTIMIZER CALLED")   # Debug print

    try:
        # -----------------------------
        # BASIC VALIDATION
        # -----------------------------
        if prices is None or prices.empty:
            raise ValueError("Price data is empty")

        returns = compute_returns(prices)
        if returns.empty:
            raise ValueError("Returns could not be computed")

        cov = returns.cov()
        if cov.isna().any().any():
            raise ValueError("Covariance matrix contains NaN")

        # -----------------------------
        # MAX SHARPE WEIGHTS
        # -----------------------------
        w = max_sharpe_weights(returns, cov)
        if np.sum(w) == 0:
            raise ValueError("Optimizer produced zero weights")

        # -----------------------------
        # RISK PARITY
        # -----------------------------
        rp = risk_parity_weights(cov)

        # -----------------------------
        # PERFORMANCE METRICS
        # -----------------------------
        port_ret = np.dot(w, returns.mean()) * 252
        port_vol = np.sqrt(w @ cov.values @ w) * np.sqrt(252)
        sharpe = port_ret / port_vol if port_vol > 0 else 0

        # -----------------------------
        # DRAWDOWN
        # -----------------------------
        dd = compute_drawdown(returns @ w)

        # -----------------------------
        # MONTE CARLO
        # -----------------------------
        mc = monte_carlo_simulation(prices, w)

        # -----------------------------
        # SECTOR WEIGHTS (placeholder)
        # -----------------------------
        sector_weights = {t: 1/len(w) for t in prices.columns}

        # -----------------------------
        # RETURN MODEL
        # -----------------------------
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
        print(">>> OPTIMIZER ERROR:", e)   # Debug print
        return None
