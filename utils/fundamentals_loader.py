import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

def safe_float(x):
    try:
        if x is None:
            return None
        x = x.replace(",", "").replace("%", "").strip()
        return float(x)
    except:
        return None

def scrape_yahoo_fundamentals(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        data = {}

        # Yahoo summary table rows
        rows = soup.select
