# utils/fundamentals_loader.py

import yfinance as yf
import pandas as pd

def load_fundamentals(tickers, full_prices=None):
    """
    Load key fundamentals for each ticker using yfinance.
    Optionally attach full_prices (price history) for downstream modules.
    """

    fundamentals = {}

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            fundamentals[ticker] = {
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "pb_ratio": info.get("priceToBook"),
                "dividend_yield": info.get("dividendYield"),
                "sector": info.get("sector"),
                "beta": info.get("beta"),
                "eps": info.get("trailingEps"),
                "revenue": info.get("totalRevenue"),
                "gross_margins": info.get("grossMargins"),
                "profit_margins": info.get("profitMargins"),
            }

        except Exception as e:
            fundamentals[ticker] = {"error": str(e)}

    # Attach price data if provided
    if full_prices is not None:
        fundamentals["full_prices"] = full_prices

    return fundamentals
