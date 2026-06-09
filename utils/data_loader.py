import yfinance as yf
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor


# =========================================================
# CLEAN TICKER INPUT
# =========================================================
def clean_tickers(tickers):
    if tickers is None:
        return []

    if isinstance(tickers, str):
        tickers = [tickers]

    tickers = [t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()]
    return list(dict.fromkeys(tickers))  # remove duplicates


# =========================================================
# SAFE FLOAT
# =========================================================
def safe_float(x):
    try:
        if x in (None, "", "None", "NaN", "nan"):
            return np.nan
        x = float(x)
        return np.nan if np.isnan(x) or np.isinf(x) else x
    except:
        return np.nan


# =========================================================
# PRICE LOADER (ADJ CLOSE)
# =========================================================
def load_price_data(tickers, start_date, end_date):
    tickers = clean_tickers(tickers)
    if len(tickers) == 0:
        return pd.DataFrame()

    try:
        raw = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=False
        )
    except Exception:
        return pd.DataFrame()

    if raw is None or raw.empty:
        return pd.DataFrame()

    # MultiIndex (multi‑ticker)
    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" in raw.columns.get_level_values(0):
            adj = raw["Adj Close"]
        elif "Adj Close" in raw.columns.get_level_values(1):
            adj = raw.xs("Adj Close", level=1, axis=1)
        elif "Close" in raw.columns.get_level_values(1):
            adj = raw.xs("Close", level=1, axis=1)
        else:
            return pd.DataFrame()

    # Single ticker
    else:
        if "Adj Close" in raw.columns:
            adj = raw["Adj Close"]
        elif "Close" in raw.columns:
            adj = raw["Close"]
        else:
            return pd.DataFrame()

    if isinstance(adj, pd.Series):
        adj = adj.to_frame()

    adj = adj[[t for t in tickers if t in adj.columns]]
    adj = adj.dropna(axis=1, how="all")
    adj = adj.dropna(how="all")

    return adj


# =========================================================
# FULL OHLCV PANEL
# =========================================================
def load_full_price_panel(tickers, start_date, end_date):
    tickers = clean_tickers(tickers)
    if len(tickers) == 0:
        return pd.DataFrame()

    try:
        raw = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=False
        )
    except Exception:
        return pd.DataFrame()

    if raw is None or raw.empty:
        return pd.DataFrame()

    if not isinstance(raw.columns, pd.MultiIndex):
        raw = pd.concat({tickers[0]: raw}, axis=1)

    return raw


# =========================================================
# VALIDATE TICKERS
# =========================================================
def validate_tickers(tickers):
    tickers = clean_tickers(tickers)
    valid = []

    for t in tickers:
        try:
            df = yf.download(t, period="3mo", progress=False)
            if df is not None and not df.empty:
                valid.append(t)
        except Exception:
            continue

    return valid


# =========================================================
# FUNDAMENTALS — SINGLE TICKER (YAHOO FINANCE)
# =========================================================
def load_single_fundamental_yahoo(ticker):
    ticker = ticker.upper().strip()

    for _ in range(3):  # retry
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

    # fallback
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


# =========================================================
# FUNDAMENTALS — MULTI‑TICKER
# =========================================================
def load_fundamentals_auto(tickers):
    tickers = clean_tickers(tickers)
    fundamentals = {}

    with ThreadPoolExecutor(max_workers=3):  # Yahoo safe limit
        for t, data in map(load_single_fundamental_yahoo, tickers):
            fundamentals[t] = data

    df = pd.DataFrame.from_dict(fundamentals, orient="index")

    df["Sector"] = df["Sector"].fillna("Unknown").replace("", "Unknown")

    return df
