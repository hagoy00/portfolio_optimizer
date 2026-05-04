import pandas as pd
import yfinance as yf

# ---------------------------------------------------------
# Clean ticker input
# ---------------------------------------------------------
def clean_ticker_input(raw):
    raw = raw.replace(" ", "")
    parts = raw.split(",")
    parts = [p for p in parts if p]

    cleaned = []
    for p in parts:
        if "." in p and p.count(".") > 1:
            cleaned.extend(p.split("."))
        else:
            cleaned.append(p)

    return cleaned


# ---------------------------------------------------------
# Convert raw Yahoo Finance output into MultiIndex
# ---------------------------------------------------------
def load_full_prices_from_raw(raw, tickers):

    if raw is None or raw.empty:
        return None

    if isinstance(raw.columns, pd.MultiIndex):
        data = raw.copy()
    else:
        new_cols = []
        for col in raw.columns:
            parts = col.split("_")
            if len(parts) == 2:
                ticker, field = parts
                new_cols.append((ticker, field))
            else:
                new_cols.append((tickers[0], col))

        data = raw.copy()
        data.columns = pd.MultiIndex.from_tuples(new_cols)

    valid = [t for t in tickers if t in data.columns.levels[0]]
    if not valid:
        return None

    data = data[valid]
# Ensure Close and Adj Close exist

    for t in valid:
        fields = data[t].columns

        if "Close" not in fields and "Adj Close" in fields:
            data[(t, "Close")] = data[(t, "Adj Close")]

        if "Adj Close" not in fields and "Close" in fields:
            data[(t, "Adj Close")] = data[(t, "Close")]
    data.columns = data.columns.set_names(["Ticker", "Field"])

    return data.dropna(how="all")


# ---------------------------------------------------------
# Extract Adjusted Close (fallback to Close)
# ---------------------------------------------------------
def extract_adj_close(full_prices):

    level1 = full_prices.columns.get_level_values(1)

    if "Adj Close" in level1:
        return full_prices.xs("Adj Close", level=1, axis=1).dropna(how="all")

    if "Close" in level1:
        return full_prices.xs("Close", level=1, axis=1).dropna(how="all")

    raise KeyError("Neither 'Adj Close' nor 'Close' found in price data.")


# ---------------------------------------------------------
# Public API: load price data
# ---------------------------------------------------------
def load_price_data(tickers_input, start, end):
    tickers = clean_ticker_input(tickers_input)

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True
    )

    full_prices = load_full_prices_from_raw(raw, tickers)
    return full_prices


# ---------------------------------------------------------
# Public API: load returns
# ---------------------------------------------------------
def load_returns_data(tickers_input, start, end):
    prices = load_price_data(tickers_input, start, end)
    adj = extract_adj_close(prices)
    returns = adj.pct_change().dropna(how="all")
    return returns
