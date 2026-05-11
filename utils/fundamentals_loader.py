import yfinance as yf
import numpy as np
import pandas as pd

# =========================================================
# SAFE NUMERIC EXTRACTION
# =========================================================
def safe_get(value):
    if value in [None, "None", "nan", "NaN", "", "-", "N/A"]:
        return None
    try:
        return float(value)
    except Exception:
        return None


# =========================================================
# CLEAN FUNDAMENTALS LOADER (MODERN, RELIABLE)
# =========================================================
def load_fundamentals(tickers):
    """
    Modern fundamentals loader using:
    - fast_info (stable numeric fields)
    - get_info() (new Yahoo API)
    - fallback to .info only if needed
    """

    fundamentals = {}

    for t in tickers:
        try:
            yf_t = yf.Ticker(t)

            # -----------------------------
            # 1. Try fast_info (very reliable)
            # -----------------------------
            fi = yf_t.fast_info

            pe = safe_get(fi.get("trailing_pe"))
            pb = safe_get(fi.get("price_to_book"))
            eps = safe_get(fi.get("eps"))
            beta = safe_get(fi.get("beta"))
            marketcap = safe_get(fi.get("market_cap"))

            # -----------------------------
            # 2. Try new get_info() API
            # -----------------------------
            try:
                new_info = yf_t.get_info()
            except Exception:
                new_info = {}

            roe = safe_get(new_info.get("returnOnEquity"))
            dy = safe_get(new_info.get("dividendYield"))
            sector = new_info.get("sector", None)

            # -----------------------------
            # 3. Fallback to old .info
            # -----------------------------
            if sector is None or sector == "":
                try:
                    old_info = yf_t.info
                    sector = old_info.get("sector", "Unknown")
                    if dy is None:
                        dy = safe_get(old_info.get("dividendYield"))
                    if roe is None:
                        roe = safe_get(old_info.get("returnOnEquity"))
                except Exception:
                    sector = "Unknown"

            # -----------------------------
            # Final assembly
            # -----------------------------
            fundamentals[t] = {
                "PE": pe,
                "PB": pb,
                "EPS": eps,
                "ROE": roe,
                "DividendYield": dy,
                "DebtToEquity": safe_get(new_info.get("debtToEquity")),
                "Beta": beta,
                "Sector": sector if sector else "Unknown",
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

    return fundamentals
