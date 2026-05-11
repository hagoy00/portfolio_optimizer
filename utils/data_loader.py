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
# Normalize Yahoo Finance output (new API format)
# ---------------------------------------------------------
def normalize_yahoo_output(raw, tickers):

    if raw is None or raw.empty:
        return pd.DataFrame()

    # If Yahoo returns Field → Ticker (new format)
    if isinstance(raw.columns, pd.MultiIndex):
        if raw.columns.names == ["Attributes", "Ticker"]:
            raw = raw.swaplevel(0, 1, axis=1).sort_index(axis=1)

    # Now raw should be Ticker → Field
    if not isinstance(raw.columns, pd.MultiIndex):
        return pd.DataFrame()

    # Filter only valid tickers
    level0 = [str(x).upper() for x in raw.columns.get_level_values(0)]
    valid = [t for t in tickers if t.upper() in level0]

    if not valid:
        return pd.DataFrame()

    data = raw[valid]

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
def load_price_data(tickers, start_date, end_date):
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        group_by="ticker",
        auto_adjust=False,
        threads=False
    )

    # STEP 1 — Keep only Close prices
    if isinstance(data.columns, pd.MultiIndex):
        data = data['Close']

    return data

    if raw is None or raw.empty:
        return pd.DataFrame()

    full_prices = normalize_yahoo_output(raw, tickers)

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
