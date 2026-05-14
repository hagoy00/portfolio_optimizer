import finnhub
import pandas as pd
import numpy as np
import yfinance as yf

FINNHUB_API_KEY = "YOUR_KEY_HERE"

# Finnhub client
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)


def compute_beta(ticker):
    try:
        stock = yf.Ticker(ticker).history(period="1y")["Close"].pct_change().dropna()
        spy = yf.Ticker("SPY").history(period="1y")["Close"].pct_change().dropna()
        return np.cov(stock, spy)[0][1] / np.var(spy)
    except:
        return None


def load_fundamentals(tickers):
    results = {}

    for t in tickers:
        try:
            # Finnhub company fundamentals
            profile = finnhub_client.company_profile2(symbol=t)
            metrics = finnhub_client.company_basic_financials(t, "all")

            data = metrics.get("metric", {})

            results[t] = {
                "PE": data.get("peNormalizedAnnual"),
                "PB": data.get("pbAnnual"),
                "EPS": data.get("epsNormalizedAnnual"),
                "ROE": data.get("roeAnnual"),
                "DividendYield": data.get("dividendYieldIndicatedAnnual"),
                "DebtToEquity": data.get("totalDebtToEquityAnnual"),
                "MarketCap": profile.get("marketCapitalization"),
                "Sector": profile.get("finnhubIndustry"),
                "Beta": compute_beta(t),
            }

        except Exception as e:
            print(f"Finnhub error for {t}: {e}")
            results[t] = {
                "PE": None,
                "PB": None,
                "EPS": None,
                "ROE": None,
                "DividendYield": None,
                "DebtToEquity": None,
                "MarketCap": None,
                "Sector": "Unknown",
                "Beta": None,
            }

    return pd.DataFrame(results).T
