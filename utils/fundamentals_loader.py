import yfinance as yf
import pandas as pd
import numpy as np

def safe_get(x):
    try:
        if x is None or x == "" or x == "None":
            return None
        return float(x)
    except:
        return None


def load_fundamentals(tickers):
    results = {}

    for t in tickers:
        try:
            yf_t = yf.Ticker(t)

            # -------------------------------------------------
            # NEW FUNDAMENTALS SOURCE (2024+)
            # -------------------------------------------------
            try:
                summary = yf_t.get_stock_summary()
            except:
                summary = {}

            pe = safe_get(summary.get("trailingPE"))
            pb = safe_get(summary.get("priceToBook"))
            eps = safe_get(summary.get("epsTrailingTwelveMonths"))
            roe = safe_get(summary.get("returnOnEquity"))
            dy = safe_get(summary.get("dividendYield"))
            dte = safe_get(summary.get("debtToEquity"))
            beta = safe_get(summary.get("beta"))
            marketcap = safe_get(summary.get("marketCap"))
            sector = summary.get("sector")

            # -------------------------------------------------
            # FALLBACKS
            # -------------------------------------------------

            # Beta fallback
            if beta is None:
                try:
                    hist = yf_t.history(period="1y")["Close"].pct_change().dropna()
                    spy = yf.Ticker("SPY").history(period="1y")["Close"].pct_change().dropna()
                    beta = np.cov(hist, spy)[0][1] / np.var(spy)
                except:
                    pass

            if not sector:
                sector = "Unknown"

            # -------------------------------------------------
            # STORE CLEAN FUNDAMENTALS
            # -------------------------------------------------
            results[t] = {
                "PE": pe,
                "PB": pb,
                "EPS": eps,
                "ROE": roe,
                "DividendYield": dy,
                "DebtToEquity": dte,
                "Beta": beta,
                "Sector": sector,
                "MarketCap": marketcap,
            }

        except Exception as e:
            print(f"Error loading fundamentals for {t}: {e}")
            results[t] = {
                "PE": None,
                "PB": None,
                "EPS": None,
                "ROE": None,
                "DividendYield": None,
                "DebtToEquity": None,
                "Beta": None,
                "Sector": "Unknown",
                "MarketCap": None,
            }

    return pd.DataFrame(results).T
