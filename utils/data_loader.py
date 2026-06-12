import pandas as pd
import yfinance as yf

# ---------------------------------------------------------
# Clean tickers
# ---------------------------------------------------------
def clean_tickers(tickers):
    if not tickers:
        return []
    cleaned = []
    for t in tickers:
        if t and isinstance(t, str):
            t2 = t.strip().upper()
            if t2 != "" and t2 not in cleaned:
                cleaned.append(t2)
    return cleaned

# ---------------------------------------------------------
# Bulletproof Price Loader
# ---------------------------------------------------------
def load_price_data(tickers, start_date, end_date):
    print(">>> DEBUG: USING BULLETPROOF PRICE LOADER <<<")

    tickers = clean_tickers(tickers)
    if len(tickers) == 0:
        return pd.DataFrame()

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
    except Exception:
        return pd.DataFrame()

    if raw is None or raw.empty:
        return pd.DataFrame()

    # Normalize column names
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = pd.MultiIndex.from_tuples(
            [(str(a).lower(), str(b).lower()) for a, b in raw.columns]
        )
    else:
        raw.columns = [str(c).lower() for c in raw.columns]

    # ---------------------------------------------------------
    # MULTI‑TICKER HANDLING
    # ---------------------------------------------------------
    if isinstance(raw.columns, pd.MultiIndex):

        lvl0 = raw.columns.get_level_values(0)
        lvl1 = raw.columns.get_level_values(1)

        # Case 1: Level 1 contains price fields
        if "adj close" in lvl1:
            adj = raw.xs("adj close", level=1, axis=1)

        elif "close" in lvl1:
            adj = raw.xs("close", level=1, axis=1)

        # Case 2: Level 0 contains price fields (Yahoo alternate format)
        elif "adj close" in lvl0:
            adj = raw.xs("adj close", level=0, axis=1)

        elif "close" in lvl0:
            adj = raw.xs("close", level=0, axis=1)

        else:
            return pd.DataFrame()

    # ---------------------------------------------------------
    # SINGLE‑TICKER HANDLING
    # ---------------------------------------------------------
    else:
        if "adj close" in raw.columns:
            adj = raw["adj close"]
        elif "close" in raw.columns:
            adj = raw["close"]
        else:
            return pd.DataFrame()

        if isinstance(adj, pd.Series):
            adj = adj.to_frame()
            adj.columns = tickers

    # Normalize ticker names
    adj.columns = [c.lower() for c in adj.columns]
    tickers_lower = [t.lower() for t in tickers]

    # Keep only requested tickers
    adj = adj[[t for t in tickers_lower if t in adj.columns]]

    # Drop empty columns/rows
    adj = adj.dropna(axis=1, how="all")
    adj = adj.dropna(how="all")

    return adj
