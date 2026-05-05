import pandas as pd
import yfinance as yf

# ---------------------------------------------------------
# Clean ticker input
# ---------------------------------------------------------
def clean_ticker_input(raw):
    if isinstance(raw, list):
        return [t.strip().upper() for t in raw if t]

    raw = raw.replace(" ", "")
    parts = raw.split(",")
    parts = [p.strip().upper() for p in parts if p]

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
        return pd.DataFrame()

    # If Yahoo already returns MultiIndex
    if isinstance(raw.columns, pd.MultiIndex):
        data = raw.copy()
    else:
        # Convert single-level columns into MultiIndex
        new_cols = []
        for col in raw.columns:
            parts = col.split("_")
            if len(parts) == 2:
                ticker, field = parts
                new_cols.append((ticker.upper(), field))
            else:
                new_cols.append((tickers[0], col))

        data = raw.copy()
        data.columns = pd.MultiIndex.from_tuples(new_cols)

    # Filter only tickers that exist in the data
    level0 = [str(x).upper() for x in data.columns.get_level_values(0)]
    valid = [t for t in tickers if t.upper() in level0]

    if not valid:
        return pd.DataFrame()

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

    if full_prices is None or full_prices.empty:
        return pd.DataFrame()

    level1 = full_prices.columns.get_level_values(1)

    if "Adj Close" in level1:
        return full_prices.xs("Adj Close", level=1, axis=1).dropna(how="all")

    if "Close" in level1:
        return full_prices.xs("Close", level=1, axis=1).dropna(how="all")

    return pd.DataFrame()


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

    if raw is None or raw.empty:
        return pd.DataFrame()

    full_prices = load_full_prices_from_raw(raw, tickers)

    if full_prices is None or full_prices.empty:
        return pd.DataFrame()

    return full_prices

# ---------------------------------------------------------
# Public API: load returns
# ---------------------------------------------------------
def load_returns_data(tickers_input, start, end):
    prices = load_price_data(tickers_input, start, end)

    if prices is None or prices.empty:
        return pd.DataFrame()

    adj = extract_adj_close(prices)

    if adj is None or adj.empty:
        return pd.DataFrame()

    returns = adj.pct_change().dropna(how="all")
    return returns
