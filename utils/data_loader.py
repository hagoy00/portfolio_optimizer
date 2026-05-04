import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


# ---------------------------------------------------------
# Helper: Clean and validate tickers
# ---------------------------------------------------------
def clean_tickers(tickers_input: str):
    if not tickers_input or tickers_input.strip() == "":
        raise ValueError("No tickers provided.")

    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    if len(tickers) == 0:
        raise ValueError("No valid tickers found after cleaning.")

    return tickers


# ---------------------------------------------------------
# Helper: Validate date range
# ---------------------------------------------------------
def validate_dates(start_date, end_date):
    if start_date is None or end_date is None:
        raise ValueError("Start date and end date must be provided.")

    if start_date >= end_date:
        raise ValueError("Start date must be before end date.")

    # Prevent future-date errors
    today = datetime.today().date()
    if end_date > today:
        end_date = today

    return start_date, end_date


# ---------------------------------------------------------
# Core: Download price data with robust guard clauses
# ---------------------------------------------------------
def load_price_data(tickers_input: str, start_date, end_date):
    # Clean tickers
    tickers = clean_tickers(tickers_input)

    # Validate dates
    start_date, end_date = validate_dates(start_date, end_date)

    # Download data
    try:
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
            threads=True
        )
    except Exception as e:
        raise RuntimeError(f"yfinance download failed: {str(e)}")

    # Handle empty DataFrame
    if data is None or data.empty:
        raise ValueError("No price data returned. Check tickers and date range.")

    # Handle multi-index columns (yfinance quirk)
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]

    # Drop columns with all NaN (IPO issues or delisted tickers)
    data = data.dropna(axis=1, how="all")

    if data.empty:
        raise ValueError("All tickers returned empty data. Possibly invalid or too new.")

    # Forward-fill missing values (market holidays, partial data)
    data = data.ffill().bfill()

    return data


# ---------------------------------------------------------
# Returns daily returns
# ---------------------------------------------------------
def load_returns_data(tickers_input: str, start_date, end_date):
    prices = load_price_data(tickers_input, start_date, end_date)
    returns = prices.pct_change().dropna()
    return returns


# ---------------------------------------------------------
# Returns log returns
# ---------------------------------------------------------
def load_log_returns(tickers_input: str, start_date, end_date):
    prices = load_price_data(tickers_input, start_date, end_date)
    log_returns = (prices / prices.shift(1)).apply(lambda x: pd.Series(np.log(x))).dropna()
    return log_returns
