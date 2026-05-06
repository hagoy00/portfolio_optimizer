import pandas as pd

def run_buy_analysis(tickers, fundamentals, prices):
    results = []

    for ticker in tickers:
        f = fundamentals.get(ticker, {})
        pe = f.get("PE", 0) or 0
        pb = f.get("PB", 0) or 0
        dy = f.get("DividendYield", 0) or 0

        # --- safe momentum & risk calculation ---
        try:
            px = prices[ticker].dropna()

            # If px is a DataFrame, reduce to Close column
            if isinstance(px, pd.DataFrame):
                if "Close" in px.columns:
                    px = px["Close"]
                else:
                    px = px.iloc[:, 0]

            returns = px.pct_change().dropna()

            momentum = float(returns.tail(60).mean() * 252) if len(returns) >= 60 else 0
            risk = float(returns.std() * (252 ** 0.5)) if len(returns) > 0 else 0

        except Exception:
            momentum, risk = 0, 0

        # --- scoring model ---
        score = 0

        # momentum
        if momentum > 0:
            score += 1

        # risk (lower better)
        if risk > 0 and risk < 0.30:
            score += 1

        # valuation
        if 0 < pe < 30:
            score += 1
        if 0 < pb < 5:
            score += 1

        # dividend
        if dy > 0.01:
            score += 1

        # rating
        if score >= 4:
            rating = "Buy"
        elif score >= 2:
            rating = "Hold"
        else:
            rating = "Sell"

        results.append({
            "Ticker": ticker,
            "Momentum": momentum,
            "Risk": risk,
            "PE": pe,
            "PB": pb,
            "DividendYield": dy,
            "Score": score,
            "Rating": rating,
        })

    return pd.DataFrame(results)
