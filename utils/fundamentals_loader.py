import yfinance as yf
import numpy as np

def safe_get(value, default=None):
    return value if value not in [None, "None", "nan", "NaN"] else default

def get_fundamentals(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        return {
            "PE": safe_get(info.get("trailingPE")),
            "PB": safe_get(info.get("priceToBook")),
            "DividendYield": safe_get(info.get("dividendYield")),
            "EPS": safe_get(info.get("trailingEps")),
            "ROE": safe_get(info.get("returnOnEquity")),
            "DebtToEquity": safe_get(info.get("debtToEquity")),
        }

    except Exception:
        return {
            "PE": None,
            "PB": None,
            "DividendYield": None,
            "EPS": None,
            "ROE": None,
            "DebtToEquity": None,
        }

def load_fundamentals(tickers):
    fundamentals = {}
    for t in tickers:
        fundamentals[t] = get_fundamentals(t)
    return fundamentals
