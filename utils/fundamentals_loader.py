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
            # 1. FAST INFO (partial, unreliable now)
            # -------------------------------------------------
            fi = yf_t.fast_info

            price = safe_get(fi.get("last_price"))
            marketcap = safe_get(fi.get("market_cap"))

            # -------------------------------------------------
            # 2. get_info() (most complete)
            # -------------------------------------------------
            try:
                info = yf_t.get_info()
            except:
                info = {}

            pe = safe_get(info.get("trailingPE"))
            pb = safe_get(info.get("priceToBook"))
            beta = safe_get(info.get("beta"))
            eps = safe_get(info.get("trailingEps"))
            dy = safe_get(info.get("dividendYield"))
            roe = safe_get(info.get("returnOnEquity"))
            dte = safe_get(info.get("debtToEquity"))
            sector = info.get("sector")

            # -------------------------------------------------
            # 3. FALLBACKS
            # -------------------------------------------------

            # Compute PE if missing
            if pe is None and price is not None and eps not in (None, 0):
                pe = price / eps

            # Compute PB if missing
            if pb is None:
                try:
                    bs = yf_t.balance_sheet
                    equity = bs.loc["Total Stockholder Equity"].iloc[0]
                    shares = marketcap / price if (marketcap and price) else None
                    if equity and shares:
                        pb = price / (equity / shares)
                except:
                    pass

            # Compute Beta if missing (using covariance)
            if beta is None:
                try:
                    hist = yf_t.history(period="1y")["Close"].pct_change().dropna()
                    spy = yf.Ticker("SPY").history(period="1y")["Close"].pct_change().dropna()
                    beta = np.cov(hist, spy)[0][1] / np.var(spy)
                except:
                    pass

            # Compute MarketCap if missing
            if marketcap is None and price is not None:
                try:
                    shares = info.get("sharesOutstanding")
                    if shares:
                        marketcap = price * shares
                except:
                    pass

            # Sector fallback
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

        except Exception:
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
