import yfinance as yf
import numpy as np

def load_fundamentals(tickers):
    fundamentals = {}

    for t in tickers:
        try:
            yf_t = yf.Ticker(t)

            # Fast info (much more reliable)
            fi = yf_t.fast_info

            pe = fi.get("pe_ratio")
            pb = fi.get("pb_ratio")
            dy = fi.get("dividend_yield")

            # Fallback to .info only if needed
            info = yf_t.info

            gross = info.get("grossMargins")
            profit = info.get("profitMargins")
            revenue = info.get("totalRevenue")

            fundamentals[t] = {
                "PE": float(pe) if pe not in [None, "None", np.nan] else None,
                "PB": float(pb) if pb not in [None, "None", np.nan] else None,
                "DividendYield": float(dy) if dy not in [None, "None", np.nan] else None,
                "gross_margins": float(gross) if gross not in [None, "None", np.nan] else None,
                "profit_margins": float(profit) if profit not in [None, "None", np.nan] else None,
                "revenue": float(revenue) if revenue not in [None, "None", np.nan] else None,
            }

        except Exception as e:
            fundamentals[t] = {
                "PE": None,
                "PB": None,
                "DividendYield": None,
                "gross_margins": None,
                "profit_margins": None,
                "revenue": None,
            }

    return fundamentals
