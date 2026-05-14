import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

def safe_float(x):
    try:
        return float(x.replace(",", "").replace("%", "")) if isinstance(x, str) else float(x)
    except:
        return None

def scrape_yahoo_fundamentals(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        r = requests.get(url, headers=headers)
        soup = BeautifulSoup(r.text, "html.parser")

        data = {}

        # Extract key-value pairs from Yahoo Finance summary table
        for row in soup.select("td"):
            key = row.text.strip()
            val = row.find_next("td").text.strip() if row.find_next("td") else None
            data[key] = val

        return {
            "PE": safe_float(data.get("PE Ratio (TTM)")),
            "PB": safe_float(data.get("Price/Book (mrq)")),
            "EPS": safe_float(data.get("EPS (TTM)")),
            "ROE": safe_float(data.get("Return on Equity (ttm)")),
            "DividendYield": safe_float(data.get("Forward Dividend & Yield", "").split("(")[-1].replace(")", "")),
            "DebtToEquity": safe_float(data.get("Total Debt/Equity (mrq)")),
            "MarketCap": data.get("Market Cap"),
            "Sector": data.get("Sector"),
        }

    except Exception as e:
        print(f"Scrape error for {ticker}: {e}")
        return {
            "PE": None,
            "PB": None,
            "EPS": None,
            "ROE": None,
            "DividendYield": None,
            "DebtToEquity": None,
            "MarketCap": None,
            "Sector": "Unknown",
        }

def load_fundamentals(tickers):
    results = {}
    for t in tickers:
        results[t] = scrape_yahoo_fundamentals(t)
    return pd.DataFrame(results).T
