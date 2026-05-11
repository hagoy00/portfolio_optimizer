import yfinance as yf
import pandas as pd
import numpy as np


# =========================================================
# CLEAN TICKER INPUT
# =========================================================
def clean_tickers(tickers):
    """
    Ensures tickers is a clean list of strings.
    Removes duplicates and empty entries.
    """
    if tickers is None:
        return []

    if isinstance(tickers, str):
        tickers = [tickers]

    tickers = [t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()]

    return list(dict.fromkeys(tickers))  # remove duplicates


# =========================================================
# SAFE YAHOO PRICE LOADER
# =========================================================
def load_price_data(tickers, start_date, end_date):
    """
    Clean, stable price loader.
    Returns a DataFrame of Adjusted Close prices only.
    Handles:
    - single ticker
    - multiple tickers
    - MultiIndex Yahoo formats
    - flat Yahoo formats
    - missing tickers
    - missing Adj Close
    """

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

    # =====================================================
    # CASE 1 — MULTI-INDEX FORMAT (most common for multi-ticker)
    # =====================================================
    if isinstance(raw.columns, pd.MultiIndex):

        # Level 0 contains fields (Adj Close, Close, etc.)
        if "Adj Close" in raw.columns.get_level_values(0):
            adj = raw["Adj Close"]

        # Level 1 contains fields
        elif "Adj Close" in raw.columns.get_level_values(1):
            adj = raw.xs("Adj Close", level=1, axis=1)

        # Fallback to Close
        elif "Close" in raw.columns.get_level_values(1):
            adj = raw.xs("Close", level=1, axis=1)

        else:
            return pd.DataFrame()

    # =====================================================
    # CASE 2 — FLAT FORMAT (single ticker)
    # =====================================================
    else:
        if "Adj Close" in raw.columns:
            adj = raw["Adj Close"]
        elif "Close" in raw.columns:
            adj = raw["Close"]
        else:
            return pd.DataFrame()

    # Ensure DataFrame
    if isinstance(adj, pd.Series):
        adj = adj.to_frame()

    # Keep only requested tickers
    adj = adj[[t for t in tickers if t in adj.columns]]

    # Drop all-NaN columns
    adj = adj.dropna(axis=1, how="all")

    # Drop all-NaN rows
    adj = adj.dropna(how="all")

    return adj


# =========================================================
# LOAD FULL PRICE PANEL (OHLCV)
# =========================================================
def load_full_price_panel(tickers, start_date, end_date):
    """
    Loads full OHLCV data for each ticker.
    Returns a MultiIndex DataFrame:
        level 0 = ticker
        level 1 = field (Open, High, Low, Close, Adj Close, Volume)
    """

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

    # MultiIndex expected
    if not isinstance(raw.columns, pd.MultiIndex):
        # Wrap single ticker into MultiIndex
        raw = pd.concat({tickers[0]: raw}, axis=1)

    return raw


# =========================================================
# VALIDATE TICKERS (EXISTS + HAS DATA)
# =========================================================
def validate_tickers(tickers):
    """
    Returns only tickers that return valid price data.
    """

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
