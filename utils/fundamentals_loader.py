import yfinance as yf
import numpy as np
import pandas as pd


# =========================================================
# SAFE NUMERIC EXTRACTION
# =========================================================
def safe_get(value):
    """
    Converts Yahoo values into clean floats.
    Returns None for invalid or missing values.
    """
    if value in [None, "None", "nan", "NaN", "", "-", "N/A"]:
        return None
    try:
        return float(value)
    except Exception:
        return None


# =========================================================
# CLEAN FUNDAMENTALS LOADER
# =========================================================
def load_fundamentals(tickers):
    """
    Clean, stable fundamentals loader.
    Fully crash-proof.
    Returns a dict:
        fundamentals[ticker] = {
            "PE": float or None,
            "PB": float or None,
            "EPS": float or None,
            "ROE": float or None,
            "DividendYield": float or None,
            "DebtToEquity": float or None,
            "Beta": float or None,
            "Sector": str,
            "MarketCap": float or None,
        }
    """

    fundamentals = {}

    for t in tickers:
        try:
            yf_t = yf.Ticker(t)
            info = yf_t.info

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
            # Full fallback — never break the app
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
