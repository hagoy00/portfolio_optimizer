import finnhub
import pandas as pd
import numpy as np
import yfinance as yf


def load_fundamentals(ticker):
    """
    Load fundamentals for a single ticker using Finnhub.
    Returns a single-row DataFrame indexed by the ticker.
    """

    try:
        # Finnhub company profile (sector, market cap, etc.)
        profile = finnhub_client.company_profile2(symbol=ticker)

        # Finnhub financial metrics (PE, PB, EPS, ROE, etc.)
        metrics = finnhub_client.company_basic_financials(ticker, "all")
        data = metrics.get("metric", {})

        # Build DataFrame
        return pd.DataFrame([{
            "PE": data.get("peNormalizedAnnual"),
            "PB": data.get("pbAnnual"),
            "EPS": data.get("epsNormalizedAnnual"),
            "ROE": data.get("roeAnnual"),
            "DividendYield": data.get("dividendYieldIndicatedAnnual"),
            "DebtToEquity": data.get("totalDebtToEquityAnnual"),
            "MarketCap": profile.get("marketCapitalization"),
            "Sector": profile.get("finnhubIndustry"),
            "Beta": compute_beta(ticker),
        }], index=[ticker])

    except Exception as e:
        print(f"Finnhub error for {ticker}: {e}")

        # Return empty fundamentals if Finnhub fails
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
