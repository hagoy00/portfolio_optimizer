import yfinance as yf
import pandas as pd
import numpy as np

def load_fundamentals(tickers, full_prices=None):
    fundamentals = {}

    for t in tickers:
        try:
            stock = yf.Ticker(t)

            # New Yahoo Finance API (fast_info)
            fast = stock.fast_info

            pe = fast.get("trailing_pe")
            pb = fast.get("price_to_book")
            div = fast.get("dividend_yield")

            # Fallbacks
            if pe is None:
                pe = fast.get("pe_ratio")

            fundamentals[t] = {
                "PE": pe,
                "PB": pb,
                "DividendYield": div,
            }

        except Exception:
            fundamentals[t] = {
                "PE": None,
                "PB": None,
                "DividendYield": None,
            }

    # Add full prices for momentum engine
    fundamentals["full_prices"] = full_prices

    return fundamentals
