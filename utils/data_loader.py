import pandas as pd
import yfinance as yf
import re

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

        # Flatten MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join([str(c) for c in col]).lower() for col in df.columns]
        else:
            df.columns = df.columns.str.lower()

        # ---------------------------------------------------
        # DYNAMIC COLUMN DETECTION
        # ---------------------------------------------------
        close_col = None
        adjclose_col = None

        for col in df.columns:
            if re.search(r"adj.*close", col):
                adjclose_col = col
            elif re.search(r"close", col):
                close_col = col

        # If only adjclose exists → create close
        if close_col is None and adjclose_col is not None:
            df["close"] = df[adjclose_col]
            close_col = "close"

        # If only close exists → create adjclose
        if adjclose_col is None and close_col is not None:
            df["adjclose"] = df[close_col]
            adjclose_col = "adjclose"

        # If neither exists → skip ticker
        if close_col is None:
            continue

        cleaned[t] = df

    if not cleaned:
        return None

    final = pd.concat(cleaned, axis=1)
    final = final.dropna(how="all")

    return final


def extract_adj_close(full_prices):
    if full_prices is None:
        return None

    # Find adjclose dynamically
    try:
        adj_cols = [c for c in full_prices.columns.levels[1] if "adjclose" in c]
        if not adj_cols:
            return None
        adj = full_prices.xs(adj_cols[0], level=1, axis=1)
        return adj.dropna(how="all")
    except Exception:
        return None
