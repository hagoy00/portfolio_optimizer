import pandas as pd
import yfinance as yf

def load_price_data(tickers, start_date, end_date):
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
        # CASE 1 — MultiIndex (normal yfinance)
        # ---------------------------------------------------
        if isinstance(data.columns, pd.MultiIndex):
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
        # CASE 2 — SingleIndex (your environment)
        # ---------------------------------------------------
        else:
            # Wrap into MultiIndex manually
            df = data.copy()

            # Ensure Adj Close exists
            if "Adj Close" not in df.columns and "Close" in df.columns:
                df["Adj Close"] = df["Close"]

            final = pd.concat({tickers[0]: df}, axis=1)
            return final

    except Exception:
        return None
