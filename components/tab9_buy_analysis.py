import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# ---------------------------------------------------
# Helper: Color coding for BUY/HOLD/SELL
# ---------------------------------------------------
def color_signal(val):
    if val == "BUY":
        return "color: green; font-weight: bold;"
    elif val == "HOLD":
        return "color: orange; font-weight: bold;"
    return "color: red; font-weight: bold;"


# ---------------------------------------------------
# Technical Indicators
# ---------------------------------------------------
def compute_RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_MACD(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


def compute_bollinger(series, window=20):
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    return ma, upper, lower


# ---------------------------------------------------
# Valuation + Analyst Data
# ---------------------------------------------------
def fetch_valuation_and_analyst(ticker):
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        rec = info.get("recommendationKey", None)

        return {
            "PE": info.get("trailingPE", np.nan),
            "PB": info.get("priceToBook", np.nan),
            "PS": info.get("priceToSalesTrailing12Months", np.nan),
            "Analyst Recommendation": rec.upper() if isinstance(rec, str) else None,
        }
    except Exception:
        return {
            "PE": np.nan,
            "PB": np.nan,
            "PS": np.nan,
            "Analyst Recommendation": None,
        }


# ---------------------------------------------------
# MAIN TAB RENDERER
# ---------------------------------------------------
def render_buy_analysis_tab(tab, prices, model):

    tab.markdown("## Buy Analysis")

    if prices is None or prices.empty:
        tab.info("Load data first to analyze buy signals.")
        return

    try:
        close = prices.xs("Close", level=1, axis=1)
        results = []

        # ---------------------------------------------------
        # PER-TICKER ANALYSIS
        # ---------------------------------------------------
        for ticker in close.columns:
            series = close[ticker].dropna()
            if len(series) < 220:
                continue

            price = series.iloc[-1]

            # Technicals
            rsi = compute_RSI(series).iloc[-1]
            macd, signal = compute_MACD(series)
            macd_last = macd.iloc[-1]
            signal_last = signal.iloc[-1]
            ma20, upper, lower = compute_bollinger(series)
            trend_200 = (price / series.rolling(200).mean().iloc[-1]) - 1
            momentum_20 = (price / series.iloc[-20]) - 1

            # Valuation + Analyst
            val = fetch_valuation_and_analyst(ticker)
            pe, pb, ps = val["PE"], val["PB"], val["PS"]
            analyst = val["Analyst Recommendation"]

            # Composite Score
            score = 0
            if rsi < 30: score += 1
            if macd_last > signal_last: score += 1
            if price > ma20.iloc[-1]: score += 1
            if trend_200 > 0: score += 1
            if momentum_20 > 0: score += 1
            if not np.isnan(pe) and pe < 25: score += 1
            if not np.isnan(pb) and pb < 5: score += 1
            if not np.isnan(ps) and ps < 10: score += 1
            if analyst in ("STRONG_BUY", "BUY"): score += 1
           
