import pandas as pd

def run_buy_analysis(tickers, fundamentals, prices):
    results = []

    for ticker in tickers:
        f = fundamentals.get(ticker, {})
        pe = f.get("PE", 0) or 0
        pb = f.get("PB", 0) or 0
        dy = f.get("DividendYield", 0) or 0

        try:
            px = prices[ticker].dropna()

            if isinstance(px, pd.DataFrame):
                if "Close" in px.columns:
                    px = px["Close"]
                else:
                    px = px.iloc[:, 0]

            returns = px.pct_change().dropna()

            if len(returns) >= 60:
                momentum = float(returns.tail(60).mean() * 252)
            elif len(returns) > 0:
                momentum = float(returns.mean() * 252)
            else:
                momentum = 0.0

            risk = float(returns.std() * (252 ** 0.5)) if len(returns) > 0 else 0.0

        except Exception:
            momentum, risk = 0.0, 0.0

        score = 0

        if momentum > 0:
            score += 1

        if 0 < risk < 0.40:
            score += 1

        if 0 < pe < 40:
            score += 1

        if 0 < pb < 8:
            score += 1

        if dy > 0.005:
            score += 1

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
