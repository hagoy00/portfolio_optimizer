import yfinance as yf
import pandas as pd

def load_fundamentals(tickers):
    fundamentals = {}

    for t in tickers:
        try:
            stock = yf.Ticker(t)
            info = stock.info

            fundamentals[t] = {
                "pe": info.get("trailingPE"),
                "ps": info.get("priceToSalesTrailing12Months"),
                "pb": info.get("priceToBook"),
                "dividend_yield": info.get("dividendYield"),
                "recommendation": info.get("recommendationKey"),
                "target_mean_price": info.get("targetMeanPrice"),
                "market_cap": info.get("marketCap"),
                "beta": info.get("beta"),
            }

        except Exception:
            fundamentals[t] = {
                "pe": None,
                "ps": None,
                "pb": None,
                "dividend_yield": None,
                "recommendation": None,
                "target_mean_price": None,
                "market_cap": None,
                "beta": None,
            }

    return fundamentals

