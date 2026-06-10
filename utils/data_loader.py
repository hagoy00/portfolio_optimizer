def load_price_data(tickers, start_date, end_date):
    print(">>> NEW PRICE LOADER ACTIVE <<<")
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

    # Normalize column names to lowercase
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = pd.MultiIndex.from_tuples(
            [(str(a).lower(), str(b).lower()) for a, b in raw.columns]
        )
    else:
        raw.columns = [str(c).lower() for c in raw.columns]

    # MULTI‑TICKER
    if isinstance(raw.columns, pd.MultiIndex):
        if "adj close" in raw.columns.get_level_values(1):
            adj = raw.xs("adj close", level=1, axis=1)
        elif "close" in raw.columns.get_level_values(1):
            adj = raw.xs("close", level=1, axis=1)
        else:
            return pd.DataFrame()

    # SINGLE‑TICKER
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

    adj = adj.dropna(axis=1, how="all")
    adj = adj.dropna(how="all")

    return adj
