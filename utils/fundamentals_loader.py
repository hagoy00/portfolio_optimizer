
import yfinance as yf

def load_fundamentals(tickers, full_prices=None):
    """
    Load fundamentals for each ticker.
    full_prices is optional but required by Buy Analysis and Commentary.
    """

    fundamentals = {}

    for t in tickers:
        try:
            yf_t = yf.Ticker(t)
            info = yf_t.info

            fundamentals[t] = {
                "PE": info.get("trailingPE"),
                "PB": info.get("priceToBook"),
                "EPS": info.get("trailingEps"),
                "ROE": info.get("returnOnEquity"),
                "DividendYield": info.get("dividendYield"),
                "DebtToEquity": info.get("debtToEquity"),
                "Beta": info.get("beta"),
                "Sector": info.get("sector"),
                "MarketCap": info.get("marketCap"),
                "full_prices": full_prices[t] if full_prices is not None and t in full_prices else None
            }

        except Exception:
            # Fail gracefully — never break the app
            fundamentals[t] = {
                "PE": None,
                "PB": None,
                "EPS": None,
                "ROE": None,
                "DividendYield": None,
                "DebtToEquity": None,
                "Beta": None,
                "Sector": None,
                "MarketCap": None,
                "full_prices": full_prices[t] if full_prices is not None and t in full_prices else None
            }

    return fundamentals
