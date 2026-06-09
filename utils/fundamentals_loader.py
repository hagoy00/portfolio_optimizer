import yfinance as yf
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor


# ---------------------------------------------------------
# SAFE FLOAT CONVERSION
# ---------------------------------------------------------
def safe_float(x):
    try:
        if x in (None, "", "None", "NaN", "nan"):
            return np.nan
        x = float(x)
        return np.nan if np.isnan(x) or np.isinf(x) else x
    except:
        return np.nan


# ---------------------------------------------------------
# LOAD FUNDAMENTALS FOR A SINGLE TICKER (YAHOO FINANCE)
# ---------------------------------------------------------
def load_single_fundamental_yahoo(ticker):
    ticker = ticker.upper().strip()

    for _ in range(3):  # retry up to 3 times
        try:
            stock = yf.Ticker(ticker)
            info = stock.get_info()

            if info and "sector" in info:
                return ticker, {
                    "PE": safe_float(info.get("trailingPE")),
                    "PB": safe_float(info.get("priceToBook")),
                    "EPS": safe_float(info.get("trailingEps")),
                    "ROE": safe_float(info.get("returnOnEquity")),
                    "DividendYield": safe_float(info.get("dividendYield")),
                    "DebtToEquity": safe_float(info.get("debtToEquity")),
                    "Beta": safe_float(info.get("beta")),
                    "MarketCap": safe_float(info.get("marketCap")),
                    "Sector": info.get("sector", "Unknown")
                }
        except:
            pass

    # fallback if Yahoo fails
    return ticker, {
        "PE": np.nan,
        "PB": np.nan,
        "EPS": np.nan,
        "ROE": np.nan,
        "DividendYield": np.nan,
        "DebtToEquity": np.nan,
        "Beta": np.nan,
        "MarketCap": np.nan,
        "Sector": "Unknown"
    }


# ---------------------------------------------------------
# MULTI‑TICKER FUNDAMENTALS LOADER
# ---------------------------------------------------------
def load_fundamentals_multi(tickers):
    fundamentals = {}
    tickers = [t.upper().strip() for t in tickers]

    with ThreadPoolExecutor(max_workers=3):  # Yahoo safe limit
        for t, data in map(load_single_fundamental_yahoo, tickers):
            fundamentals[t] = data

    df = pd.DataFrame.from_dict(fundamentals, orient="index")

    # Clean sector
    df["Sector"] = df["Sector"].fillna("Unknown").replace("", "Unknown")

    return df
