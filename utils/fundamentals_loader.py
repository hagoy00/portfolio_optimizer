import yfinance as yf
import numpy as np

# ---------------------------------------------------------
# Safe extraction helper
# ---------------------------------------------------------
def safe_get(value):
    if value in [None, "None", "nan", "NaN"]:
        return None
    try:
        return float(value)
    except:
        return value

# ---------------------------------------------------------
# Clean Fundamentals Loader
# ---------------------------------------------------------
def load_fundamentals(tickers):
    """
    Clean, stable fundamentals loader.
    Matches the structure expected by:
    - Tab 5 (Fundamentals)
    - Tab 7 (AI Commentary)
    - Tab 8 (Buy Analysis)
    """

    fundamentals = {}

    for t in tickers:
        try:
            info = yf.Ticker(t).info

            fundamentals[t] = {
                "PE": safe_get(info.get("trailingPE")),
                "PB": safe_get(info.get("priceToBook")),
                "EPS": safe_get(info.get("trailingEps")),
                "ROE": safe_get(info.get("returnOnEquity")),
                "DividendYield": safe_get(info.get("dividendYield")),
                "DebtToEquity": safe_get(info.get("debtToEquity")),
                "Beta": safe_get(info.get("beta")),
                "Sector": info.get("sector", "Unknown"),
                "MarketCap": safe_get(info.get("marketCap")),
            }

        except Exception:
            fundamentals[t] = {
                "PE": None,
                "PB": None,
                "EPS": None,
                "ROE": None,
                "DividendYield": None,
                "DebtToEquity": None,
                "Beta": None,
                "Sector": "Unknown",
                "MarketCap": None,
            }

    return fundamentals
