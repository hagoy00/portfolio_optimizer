import finnhub
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st

# -----------------------------
# Ticker Normalization (Fixes BRK.B → BRK-B, BF.B → BF-B)
# -----------------------------
def normalize_ticker(t):
    return t.replace(".", "-")


# -----------------------------
# Main Fundamentals Loader (Single Ticker)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_fundamentals(ticker):
    """
    Load fundamentals for a single ticker using Finnhub,
    with fallback to yfinance for missing fields.
    Returns a single-row DataFrame indexed by the ticker.
    """

    ticker = normalize_ticker(ticker)

    try:
        # -----------------------------
        # 1. Finnhub profile (sector, market cap)
        # -----------------------------
        profile = finnhub_client.company_profile2(symbol=ticker)

        # -----------------------------
        # 2. Finnhub financial metrics (PE, PB, EPS, ROE, etc.)
        # -----------------------------
        metrics = finnhub_client.company_basic_financials(ticker, "all")
        data = metrics.get("metric", {})

        # -----------------------------
        # 3. Fallback to yfinance
        # -----------------------------
        yf_t = yf.Ticker(ticker)
        yf_fast = yf_t.fast_info

        try:
            yf_info = yf_t.get_info()
        except:
            yf_info = {}

        # PE fallback
        if data.get("peNormalizedAnnual") is None:
            data["peNormalizedAnnual"] = yf_info.get("trailingPE")

        # PB fallback
        if data.get("pbAnnual") is None:
            data["pbAnnual"] = yf_info.get("priceToBook")

        # EPS fallback
        if data.get("epsNormalizedAnnual") is None:
            data["epsNormalizedAnnual"] = yf_info.get("trailingEps")

        # ROE fallback
        if data.get("roeAnnual") is None:
            data["roeAnnual"] = yf_info.get("returnOnEquity")

        # Dividend Yield fallback
        if data.get("dividendYieldIndicatedAnnual") is None:
            data["dividendYieldIndicatedAnnual"] = yf_info.get("dividendYield")

        # Debt-to-Equity fallback
        if data.get("totalDebtToEquityAnnual") is None:
            data["totalDebtToEquityAnnual"] = yf_info.get("debtToEquity")

        # Market Cap fallback
        if profile.get("marketCapitalization") is None:
            profile["marketCapitalization"] = yf_fast.get("market_cap")

        # Sector fallback
        if profile.get("finnhubIndustry") is None:
            profile["finnhubIndustry"] = yf_info.get("sector")

        # Beta fallback
        try:
            beta_value = compute_beta(ticker)
        except:
            beta_value = None

        if beta_value is None:
            beta_value = yf_fast.get("beta")

        # -----------------------------
        # 4. Build final DataFrame
        # -----------------------------
        return pd.DataFrame([{
            "PE": data.get("peNormalizedAnnual"),
            "PB": data.get("pbAnnual"),
            "EPS": data.get("epsNormalizedAnnual"),
            "ROE": data.get("roeAnnual"),
            "DividendYield": data.get("dividendYieldIndicatedAnnual"),
            "DebtToEquity": data.get("totalDebtToEquityAnnual"),
            "MarketCap": profile.get("marketCapitalization"),
            "Sector": profile.get("finnhubIndustry") or "Unknown",
            "Beta": beta_value,
        }], index=[ticker])

    except Exception as e:
        print(f"Finnhub error for {ticker}: {e}")

        return pd.DataFrame([{
            "PE": None,
            "PB": None,
            "EPS": None,
            "ROE": None,
            "DividendYield": None,
            "DebtToEquity": None,
            "MarketCap": None,
            "Sector": "Unknown",
            "Beta": None,
        }], index=[ticker])


# ---------------------------------------------------------
# MULTI‑TICKER FUNDAMENTALS LOADER (ALWAYS RETURNS A DATAFRAME)
# ---------------------------------------------------------
def load_fundamentals_multi(tickers):
    """
    Loads fundamentals for a list of tickers.
    Returns a combined DataFrame indexed by ticker.
    Guarantees a DataFrame even if some tickers fail.
    """

    frames = []

    for t in tickers:
        try:
            df = load_fundamentals(t)

            if isinstance(df, pd.DataFrame):
                frames.append(df)
            else:
                raise ValueError("Single‑ticker loader returned non‑DataFrame")

        except Exception as e:
            print(f"Error loading fundamentals for {t}: {e}")

            frames.append(pd.DataFrame([{
                "PE": None,
                "PB": None,
                "EPS": None,
                "ROE": None,
                "DividendYield": None,
                "DebtToEquity": None,
                "MarketCap": None,
                "Sector": "Unknown",
                "Beta": None,
            }], index=[t]))

    if len(frames) == 0:
        return pd.DataFrame()

    df_all = pd.concat(frames)

    # Clean sector column
    if "Sector" in df_all.columns:
        df_all["Sector"] = df_all["Sector"].fillna("Unknown").replace("", "Unknown")

    return df_all
