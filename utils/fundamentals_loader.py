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
            # 1. FAST INFO (partial)
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

            # Correct Yahoo Finance field names
            pe = safe_get(info.get("trailingPE"))
            pb = safe_get(info.get("priceToBook"))
            eps = safe_get(info.get("trailingEps"))
            roe = safe_get(info.get("returnOnEquity"))
            dy = safe_get(info.get("dividendYield"))
            dte = safe_get(info.get("debtToEquity"))
            beta = safe_get(info.get("beta"))
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
                    shares = info.get("sharesOutstanding")
                    if equity and shares:
                        book_value_per_share = equity / shares
                        if book_value_per_share not in (None, 0):
                            pb = price / book_value_per_share
                except:
                    pass

            # Compute Beta if missing
            if beta is None:
                try:
                    hist = yf_t.history(period="1y")["Close"].pct_change().dropna()
                    spy = yf.Ticker("SPY").history(period="1y")["Close"].pct_change().dropna()
                    beta = np.cov(hist, spy)[0][1] / np.var(spy)
                except:
                    pass

            # Compute MarketCap if missing
            if marketcap is None and price is not None:
                shares = info.get("sharesOutstanding")
                if shares:
                    marketcap = price * shares

            # Sector fallback
            if not sector or sector == "":
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
