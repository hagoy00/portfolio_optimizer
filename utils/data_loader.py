import pandas as pd

def clean_ticker_input(raw):
    """
    Cleans user input like:
    'AAPL, TSLA , MSFT'
    'BRK.B, VOO'
    'AAPL TSLA MSFT'
    """
    raw = raw.replace(" ", "")
    parts = raw.split(",")
    parts = [p for p in parts if p]

    cleaned = []
    for p in parts:
        # Fix cases like "AAPL.MSFT"
        if "." in p and p.count(".") > 1:
            cleaned.extend(p.split("."))
        else:
            cleaned.append(p)

    return cleaned


def load_full_prices_from_raw(raw, tickers):
    """
    Converts raw yf.download output (flat or grouped) into a proper MultiIndex:
        (ticker, field)
    Ensures Close and Adj Close exist.
    """

    if raw is None or raw.empty:
        return None

    # Case 1: Yahoo returned grouped MultiIndex (normal behavior)
    if isinstance(raw.columns, pd.MultiIndex):
        data = raw.copy()

    else:
        # Case 2: Yahoo returned FLAT columns (your 2026 case)
        # Example: AAPL_Open, AAPL_Close, TSLA_Open, TSLA_Close
        new_cols = []
        for col in raw.columns:
            parts = col.split("_")
            if len(parts) == 2:
                ticker, field = parts
                new_cols.append((ticker, field))
            else:
                # fallback: put everything under the first ticker
                new_cols.append((tickers[0], col))

        data = raw.copy()
        data.columns = pd.MultiIndex.from_tuples(new_cols)

    # Keep only tickers that actually downloaded
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

    return data.dropna(how="all")



def extract_adj_close(full_prices):
    """
    Returns a flat DataFrame of adjusted close prices.
    If 'Adj Close' is missing (as in some 2026 Yahoo data),
    fall back to 'Close'.
    """
    level1 = full_prices.columns.get_level_values(1)

    if "Adj Close" in level1:
        return full_prices.xs("Adj Close", level=1, axis=1).dropna(how="all")

    # Fallback for 2026+ Yahoo data
    if "Close" in level1:
        return full_prices.xs("Close", level=1, axis=1).dropna(how="all")

    raise KeyError("Neither 'Adj Close' nor 'Close' found in price data.")
