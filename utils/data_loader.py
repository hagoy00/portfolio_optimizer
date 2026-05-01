import pandas as pd
import yfinance as yf

def load_price_data(tickers, start_date, end_date):
    """
    Downloads OHLCV data for multiple tickers using yfinance.
    Always returns a clean MultiIndex DataFrame:
        columns = (Ticker, Field)
    Ensures both 'Close' and 'Adj Close' exist.
    Removes tickers with incomplete data.
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

        # If only one ticker → wrap into MultiIndex
        if isinstance(data.columns, pd.Index):
            data = pd.concat({tickers[0]: data}, axis=1)

        # Keep only tickers that actually downloaded
        valid = [t for t in tickers if t in data.columns.levels[0]]
        if not valid:
            return None

        data = data[valid]

        cleaned = []

        for t in valid:
            fields = data[t].columns

            # Skip tickers with no usable price data
            if ("Close" not in fields) and ("Adj Close" not in fields):
                continue

            # If Close missing → use Adj Close
            if "Close" not in fields and "Adj Close" in fields:
                data[(t, "Close")] = data[(t, "Adj Close")]

            # If Adj Close missing → use Close
            if "Adj Close" not in fields and "Close" in fields:
                data[(t, "Adj Close")] = data[(t, "Close")]

            cleaned.append(t)

        # Keep only cleaned tickers
        if not cleaned:
            return None

        data = data[cleaned]

        # Drop rows where all tickers have NaN
        data = data.dropna(how="all")

        return data

    except Exception:
        return None
