import yfinance as yf
import numpy as np

def load_fundamentals(tickers):
    fundamentals = {}

    for t in tickers:
        try:
            yf_t = yf.Ticker(t)

            # -----------------------------
            # FAST INFO (preferred)
            # -----------------------------
            fi = yf_t.fast_info

            pe = fi.get("pe_ratio")
            pb = fi.get("pb_ratio")
            ps = fi.get("price_to_sales")
            dy = fi.get("dividend_yield")
            beta = fi.get("beta")

            # -----------------------------
            # FULL INFO (fallback)
            # -----------------------------
            info = yf_t.info

            forward_pe = info.get("forwardPE")
            revenue = info.get("totalRevenue")
            gross = info.get("grossMargins")
            profit = info.get("profitMargins")
            eps = info.get("trailingEps")
            sector = info.get("sector")

            recommendation = info.get("recommendationKey")
            target_mean_price = info.get("targetMeanPrice")

            fundamentals[t] = {
                # ---- Valuation ----
                "pe": float(pe) if pe not in [None, "None", np.nan] else None,
                "pb": float(pb) if pb not in [None, "None", np.nan] else None,
                "ps": float(ps) if ps not in [None, "None", np.nan] else None,
                "forward_pe": float(forward_pe) if forward_pe not in [None, "None", np.nan] else None,

                # ---- Dividend + Beta ----
                "dividend_yield": float(dy) if dy not in [None, "None", np.nan] else None,
                "beta": float(beta) if beta not in [None, "None", np.nan] else None,

                # ---- Margins + Revenue ----
                "gross_margins": float(gross) if gross not in [None, "None", np.nan] else None,
                "profit_margins": float(profit) if profit not in [None, "None", np.nan] else None,
                "revenue": float(revenue) if revenue not in [None, "None", np.nan] else None,

                # ---- EPS + Sector ----
                "eps": float(eps) if eps not in [None, "None", np.nan] else None,
                "sector": sector,

                # ---- Analyst Data ----
                "recommendation": recommendation,
                "target_mean_price": float(target_mean_price) if target_mean_price not in [None, "None", np.nan] else None,
            }

        except Exception:
            fundamentals[t] = {
                "pe": None, "pb": None, "ps": None, "forward_pe": None,
                "dividend_yield": None, "beta": None,
                "gross_margins": None, "profit_margins": None, "revenue": None,
                "eps": None, "sector": None,
                "recommendation": None, "target_mean_price": None,
            }

    return fundamentals
