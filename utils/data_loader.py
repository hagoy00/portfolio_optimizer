import pandas as pd
import yfinance as yf

# ---------------------------------------------------
# CLEAN PRICE DOWNLOAD FUNCTION
# ---------------------------------------------------
def load_price_data(tickers, start_date, end_date):
    """
    Downloads OHLCV data for multiple tickers using yfinance.
    Returns a clean multi-index DataFrame:
        columns = (Ticker, Field)
        fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
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

        # ---------------------------------------------------
        # STANDARDIZE FORMAT
        # ---------------------------------------------------
        # If only one ticker, yfinance returns a single-level column index
        if isinstance(data.columns, pd.MultiIndex) is False:
            # Convert to multi-index: (Ticker, Field)
            data = pd.concat({tickers[0]: data}, axis=1)

        # Sort columns for consistency
        data = data.sort_index(axis=1)

        # Drop rows where all tickers have NaN
        data = data.dropna(how="all")

        return data

    except Exception:
        return None
