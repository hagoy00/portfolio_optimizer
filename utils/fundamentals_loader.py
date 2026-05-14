import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import yfinance as yf

HEADERS = {"User-Agent": "Mozilla/5.0"}

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


def scrape_key_statistics(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}/key-statistics"
    r = requests.get(url, headers=HEADERS, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")

    stats = {}

    for row in soup.select("table tbody tr"):
        cols = row.find_all("td")
        if len(cols) == 2:
            key = cols[0].text.strip()
            val = cols[1].text.strip()
            stats[key] = val

    return stats


def scrape_profile_sector(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}/profile"
    r = requests.get(url, headers=HEADERS, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")

    tag = soup.find("span", string="Sector")
    if tag:
        return tag.find_next("span").text.strip()
    return "Unknown"


def load_fundamentals(tickers):
    results = {}

    for t in tickers:
        try:
            stats = scrape_key_statistics(t)
            sector = scrape_profile_sector(t)
            beta = compute_beta(t)

            results[t] = {
                "PE": safe_float(stats.get("Trailing P/E")),
                "PB": safe_float(stats.get("Price/Book (mrq)")),
                "EPS": safe_float(stats.get("Diluted EPS (ttm)")),
                "ROE": safe_float(stats.get("Return on Equity (ttm)")),
                "DividendYield": safe_float(stats.get("Forward Annual Dividend Yield")),
                "DebtToEquity": safe_float(stats.get("Total Debt/Equity (mrq)")),
                "MarketCap": stats.get("Market Cap (intraday)"),
                "Sector": sector,
                "Beta": beta,
            }

        except Exception as e:
            print(f"Yahoo scrape error for {t}: {e}")
            results[t] = {
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

    return pd.DataFrame(results).T
