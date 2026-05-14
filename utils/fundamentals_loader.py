import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import yfinance as yf


def safe_float(x):
    try:
        if x is None:
            return None
        x = x.replace(",", "").replace("%", "").strip()
        return float(x)
    except:
        return None


def compute_beta(ticker):
    try:
        stock = yf.Ticker(ticker).history(period="1y")["Close"].pct_change().dropna()
        spy = yf.Ticker("SPY").history(period="1y")["Close"].pct_change().dropna()
        return np.cov(stock, spy)[0][1] / np.var(spy)
    except:
        return None


def scrape_yahoo_fundamentals(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        data = {}

        # Summary table
        rows = soup.select("table tbody tr")
        for row in rows:
            cols = row.find_all("td")
            if len(cols) == 2:
                key = cols[0].text.strip()
                val = cols[1].text.strip()
                data[key] = val

        # Sector (from profile page)
        profile_url = f"https://finance.yahoo.com/quote/{ticker}/profile"
        r2 = requests.get(profile_url, headers=headers, timeout=10)
        soup2 = BeautifulSoup(r2.text, "html.parser")

        sector = None
        sector_tag = soup2.find("span", string="Sector")
        if sector_tag:
            sector = sector_tag.find_next("span").text.strip()

        # Compute Beta
        beta = compute_beta(ticker)

        return {
            "PE": safe_float(data.get("PE Ratio (TTM)")),
            "PB": safe_float(data.get("Price/Book (mrq)")),
            "EPS": safe_float(data.get("EPS (TTM)")),
            "ROE": safe_float(data.get("Return on Equity (ttm)")),
            "DividendYield": safe_float(
                data.get("Forward Dividend & Yield", "").split("(")[-1].replace(")", "")
            ),
            "DebtToEquity": safe_float(data.get("Total Debt/Equity (mrq)")),
            "MarketCap": data.get("Market Cap"),
            "Sector": sector if sector else "Unknown",
            "Beta": beta,
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
            "Beta": None,
        }


def load_fundamentals(tickers):
    results = {}
    for t in tickers:
        results[t] = scrape_yahoo_fundamentals(t)
    return pd.DataFrame(results).T
