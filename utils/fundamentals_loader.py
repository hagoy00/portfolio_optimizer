import yfinance as yf
import pandas as pd
import numpy as np


# ---------------------------------------------------------
# SAFE FLOAT
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
def load_fundamentals(ticker):
    ticker = ticker.upper().strip()

    for _ in range(3):  # retry
        try:
            stock = yf.Ticker(ticker)
            info = stock.get_info()

            if info and "sector" in info:
                return pd.DataFrame([{
                    "PE": safe_float(info.get("trailingPE")),
                    "PB": safe_float(info.get("priceToBook")),
                    "EPS": safe_float(info.get("trailingEps")),
                    "ROE": safe_float(info.get("returnOnEquity")),
                    "DividendYield": safe_float(info.get("dividendYield")),
                    "DebtToEquity": safe_float(info.get("debtToEquity")),
                    "Beta": safe_float(info.get("beta")),
                    "MarketCap": safe_float(info.get("marketCap")),
                    "Sector": info.get("sector", "Unknown")
                }], index=[ticker])

        except:
            pass

    # fallback
    return pd.DataFrame([{
        "PE": np.nan,
        "PB": np.nan,
        "EPS": np.nan,
        "ROE": np.nan,
        "DividendYield": np.nan,
        "DebtToEquity": np.nan,
        "Beta": np.nan,
        "MarketCap": np.nan,
        "Sector": "Unknown"
    }], index=[ticker])


# ---------------------------------------------------------
# MULTI‑TICKER FUNDAMENTALS LOADER
# ---------------------------------------------------------
def load_fundamentals_multi(tickers):
    frames = []

    for t in tickers:
        try:
            df = load_fundamentals(t)
            frames.append(df)
        except:
            frames.append(pd.DataFrame([{
                "PE": np.nan,
                "PB": np.nan,
                "EPS": np.nan,
                "ROE": np.nan,
                "DividendYield": np.nan,
                "DebtToEquity": np.nan,
                "Beta": np.nan,
                "MarketCap": np.nan,
                "Sector": "Unknown"
            }], index=[t]))

    if len(frames) == 0:
        return pd.DataFrame()

    df_all = pd.concat(frames)
    df_all["Sector"] = df_all["Sector"].fillna("Unknown").replace("", "Unknown")

    return df_all
