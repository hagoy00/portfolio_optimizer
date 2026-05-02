import pandas as pd
import yfinance as yf

# ---------------------------------------------------
# CLEAN TICKER INPUT
# ---------------------------------------------------
def clean_ticker_input(ticker_string):
    """
    Converts a comma/space separated string into a clean list of tickers.
    Example: "AAPL, MSFT TSLA" → ["AAPL", "MSFT", "TSLA"]
    """
    if not ticker_string:
        return []

    tickers = (
        ticker_string.replace(",", " ")
        .upper()
        .split()
    )

    return list(dict.fromkeys(tickers))  # remove duplicates, preserve order


# ---------------------------------------------------
# RAW PRICE DOWNLOAD
# ---------------------------------------------------
def load_full_prices_from_raw(tickers, start, end):
    """
    Downloads OHLCV data for multiple tickers and returns a MultiIndex DataFrame:
    (ticker, field)
    """

    if not tickers:
        return None

    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True
    )

    if data is None or data.empty:
        return None

    # If only one ticker → wrap into MultiIndex
    if isinstance(data.columns, pd.Index):
        data = pd.concat({tickers[0]: data}, axis=1)

    cleaned = {}

    for t in tickers:
        if t not in data.columns.levels[0]:
            continue

        df = data[t].copy()

        # Ensure Close exists
        if "Close" not in df.columns and "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]

        # Ensure Adj Close exists
        if "Adj Close" not in df.columns and "Close" in df.columns:
            df["Adj Close"] = df["Close"]

        if df["Close"].dropna().empty:
            continue

        cleaned[t] = df

    if not cleaned:
        return None

    final = pd.concat(cleaned, axis=1)
    final = final.dropna(how="all")

    return final


# ---------------------------------------------------
# EXTRACT ADJ CLOSE
# ---------------------------------------------------
def extract_adj_close(full_prices):
    """
    Extracts only the Adj Close column from the MultiIndex OHLCV data.
    Returns a DataFrame: index = dates, columns = tickers.
    """
    if full_prices is None:
        return None

    try:
        adj = full_prices.xs("Adj Close", level=1, axis=1)
        return adj.dropna(how="all")
    except Exception:
        return None
