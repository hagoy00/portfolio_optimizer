import yfinance as yf

def safe_get(value):
    try:
        if value in [None, "None", "nan", "NaN"]:
            return None
        return float(value)
    except:
        return None

def load_fundamentals(tickers):
    fundamentals = {}

    for t in tickers:
        try:
            stock = yf.Ticker(t)
            info = stock.info

            fundamentals[t] = {
                # Core valuation
                "PE": safe_get(info.get("trailingPE")),
                "PB": safe_get(info.get("priceToBook")),
                "DividendYield": safe_get(info.get("dividendYield")),
                
                # Profitability
                "EPS": safe_get(info.get("trailingEps")),
                "ROE": safe_get(info.get("returnOnEquity")),
                "DebtToEquity": safe_get(info.get("debtToEquity")),

                # A2 Enhancements
                "Beta": safe_get(info.get("beta")),
                "MarketCap": safe_get(info.get("marketCap")),
                "Sector": info.get("sector") or "Unknown",
            }

        except Exception as e:
            fundamentals[t] = {
                "PE": None,
                "PB": None,
                "DividendYield": None,
                "EPS": None,
                "ROE": None,
                "DebtToEquity": None,
                "Beta": None,
                "MarketCap": None,
                "Sector": "Unknown",
            }

    return fundamentals
