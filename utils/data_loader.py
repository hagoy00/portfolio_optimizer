import yfinance as yf
import pandas as pd


def normalize_yahoo_output(raw, tickers):
    """
    Normalize Yahoo Finance multi-index output into a clean DataFrame:
    Columns become: ['AAPL', 'MSFT', 'NVDA', ...]
    """
    # If single ticker, yfinance returns a normal DataFrame
    if not isinstance(raw.columns, pd.MultiIndex):
        return raw.to_frame().T if isinstance(raw, pd.Series) else raw

    # MultiIndex: (Ticker, Field)
    clean = {}

    for t in tickers:
        if t in raw.columns.get_level_values(0):
            try:
                clean[t] = raw[t]['Close']
            except Exception:
                continue

    return pd.DataFrame(clean)


def load_price_data(tickers, start_date, end_date):
    """
    Downloads price data for multiple tickers and returns ONLY Close prices.
    This prevents OHLC columns from polluting the app.
    """

    raw = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        group_by="ticker",
        auto_adjust=False,
        threads=False
    )

    # If nothing downloaded
    if raw is None or raw.empty:
        return pd.DataFrame()

    # Normalize Yahoo output
    full_prices = normalize_yahoo_output(raw, tickers)

    if full_prices is None or full_prices.empty:
        return pd.DataFrame()

    # ---------------------------------------------------------
    # STEP 1 — KEEP ONLY CLOSE PRICES (CRITICAL FIX)
    # ---------------------------------------------------------
    # If MultiIndex: (Ticker, Field)
    if isinstance(full_prices.columns, pd.MultiIndex):
        full_prices = full_prices.xs("Close", level=1, axis=1)

    # If still multi-level or messy, flatten
    if isinstance(full_prices.columns, pd.MultiIndex):
        full_prices.columns = [col[0] for col in full_prices.columns]

    # Final cleanup: ensure only tickers remain
    full_prices = full_prices[[t for t in tickers if t in full_prices.columns]]

    return full_prices
