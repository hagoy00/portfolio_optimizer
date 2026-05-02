import pandas as pd
import yfinance as yf

def clean_ticker_input(ticker_string):
    if not ticker_string:
        return []
    tickers = (
        ticker_string.replace(",", " ")
        .upper()
        .split()
    )
    return list(dict.fromkeys(tickers))


def load_full_prices_from_raw(tickers, start, end):
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

        # Normalize column names
        df.columns = (
            df.columns
            .str.lower()
            .str.replace(" ", "")
            .str.replace("_", "")
        )

        # Map all possible variants to standard names
        rename_map = {
            "close": "close",
            "adjclose": "adjclose",
            "adj_close": "adjclose",
            "close*": "close",
            "closingprice": "close",
        }
        df = df.rename(columns=rename_map)

        # Ensure close exists
        if "close" not in df.columns and "adjclose" in df.columns:
            df["close"] = df["adjclose"]

        # Ensure adjclose exists
        if "adjclose" not in df.columns and "close" in df.columns:
            df["adjclose"] = df["close"]

        if df["close"].dropna().empty:
            continue

        cleaned[t] = df

    if not cleaned:
        return None

    # Rebuild MultiIndex with normalized names
    final = pd.concat(cleaned, axis=1)
    final = final.dropna(how="all")

    return final


def extract_adj_close(full_prices):
    if full_prices is None:
        return None

    # inner level is now lowercase: "adjclose"
    try:
        adj = full_prices.xs("adjclose", level=1, axis=1)
        return adj.dropna(how="all")
    except Exception:
        return None
