import yfinance as yf
import pandas as pd

# ---------------------------------------------------------
# Modern, Safe Yahoo Loader
# ---------------------------------------------------------
def load_price_data(tickers, start_date, end_date):
    """
    Clean, stable price loader.
    Returns a DataFrame of Adjusted Close prices only.
    Works for:
    - single ticker
    - multiple tickers
    - MultiIndex Yahoo formats
    - flat Yahoo formats
    """

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

        if raw is None or raw.empty:
            return pd.DataFrame()

        # ---------------------------------------------------------
        # MULTI-INDEX FORMAT (most common for multiple tickers)
        # ---------------------------------------------------------
        if isinstance(raw.columns, pd.MultiIndex):

            # Case 1: Level 0 contains fields (Adj Close, Close, etc.)
            if "Adj Close" in raw.columns.get_level_values(0):
                adj = raw["Adj Close"]

            # Case 2: Level 1 contains fields
            elif "Adj Close" in raw.columns.get_level_values(1):
                adj = raw.xs("Adj Close", level=1, axis=1)

            # Fallback to Close
            elif "Close" in raw.columns.get_level_values(1):
                adj = raw.xs("Close", level=1, axis=1)

            else:
                return pd.DataFrame()

        # ---------------------------------------------------------
        # FLAT FORMAT (single ticker)
        # ---------------------------------------------------------
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

        return adj

    except Exception as e:
        # Never crash the app
        return pd.DataFrame()
