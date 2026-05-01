import pandas as pd
import yfinance as yf

def load_price_data(tickers, start_date, end_date):
    """
    Robust price loader that:
    - Downloads OHLCV for all tickers
    - Keeps partial data
    - Ensures Close & Adj Close exist
    - Returns a clean MultiIndex DataFrame
    """

    if not tickers:
        return None

    try:
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False,
            group_by="ticker"
        )

        if data is None or data.empty:
            return None

        # Wrap single ticker into MultiIndex
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

            # Skip tickers with no usable data
            if "Close" not in df.columns or df["Close"].dropna().empty:
                continue

            cleaned[t] = df

        if not cleaned:
            return None

        # Rebuild MultiIndex DataFrame
        final = pd.concat(cleaned, axis=1)

        # Drop rows where all tickers are NaN
        final = final.dropna(how="all")

        return final

    except Exception:
        return None
