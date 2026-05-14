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
    fundamentals = {}

    for t in tickers:
        try:
            yf_t = yf.Ticker(t)

            # -----------------------------
            # 1. fast_info (most reliable)
            # -----------------------------
            fi = yf_t.fast_info

            pe  = safe_get(fi.get("trailing_pe"))
            pb  = safe_get(fi.get("price_to_book"))
            eps = safe_get(fi.get("eps"))
            beta = safe_get(fi.get("beta"))
            marketcap = safe_get(fi.get("market_cap"))

            # -----------------------------
            # 2. get_info() (new API)
            # -----------------------------
            try:
                info = yf_t.get_info()
            except:
                info = {}

            dy = safe_get(info.get("dividendYield"))
            roe = safe_get(info.get("returnOnEquity"))
            sector = info.get("sector")

            # -----------------------------
            # 3. Fallback sector if missing
            # -----------------------------
            if sector is None or sector == "":
                sector = "Unknown"

            fundamentals[t] = {
                "PE": pe,
                "PB": pb,
                "EPS": eps,
                "ROE": roe,
                "DividendYield": dy,
                "DebtToEquity": safe_get(info.get("debtToEquity")),
                "Beta": beta,
                "Sector": sector,
                "MarketCap": marketcap,
            }

        except Exception:
            fundamentals[t] = {
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

    return pd.DataFrame(fundamentals).T

