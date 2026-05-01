import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

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

def render_tab9(tab, full_prices):
    tab.subheader("Is This Stock a Good Buy?")

    if full_prices is None or full_prices.empty:
        tab.info("Load data first to analyze buy signals.")
        return

    try:
        close = full_prices.xs("Close", level=1, axis=1)
        results = []

        for ticker in close.columns:
            series = close[ticker].dropna()
            if len(series) < 220:
                continue

            # Technicals
            rsi = compute_RSI(series).iloc[-1]
            macd, signal = compute_MACD(series)
            macd_last = macd.iloc[-1]
            signal_last = signal.iloc[-1]
            ma20, upper, lower = compute_bollinger(series)
            price = series.iloc[-1]

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
            if analyst in ("SELL", "STRONG_SELL", "UNDERPERFORM"): score -= 1

            if score >= 6:
                signal_text = "BUY"
            elif score >= 3:
                signal_text = "HOLD"
            else:
                signal_text = "SELL"

            results.append({
                "Ticker": ticker,
                "Price": price,
                "RSI": rsi,
                "MACD": macd_last,
                "Signal Line": signal_last,
                "200‑Day Trend": trend_200,
                "20‑Day Momentum": momentum_20,
                "PE": pe,
                "PB": pb,
                "PS": ps,
                "Analyst Recommendation": analyst,
                "Score": score,
                "Signal": signal_text
            })

        if not results:
            tab.warning("Not enough historical data to compute buy analysis.")
            return

        df = pd.DataFrame(results).set_index("Ticker")

        tab.markdown("### 📈 Buy / Hold / Sell Signals")
        tab.dataframe(
            df.style.format({
                "Price": "{:.2f}",
                "RSI": "{:.1f}",
                "MACD": "{:.4f}",
                "Signal Line": "{:.4f}",
                "200‑Day Trend": "{:.2%}",
                "20‑Day Momentum": "{:.2%}",
                "PE": "{:.1f}",
                "PB": "{:.1f}",
                "PS": "{:.1f}",
            })
        )

        tab.markdown("### 🧠 AI Buy Commentary")
        for ticker, row in df.iterrows():
            sig = row["Signal"]
            analyst = row["Analyst Recommendation"]

            text = f"**{ticker}: {sig}** — "

            if sig == "BUY":
                text += "Strong technical and valuation profile. "
            elif sig == "HOLD":
                text += "Mixed technicals or valuation; balanced risk/reward. "
            else:
                text += "Weak technicals or stretched valuation; caution warranted. "

            if isinstance(analyst, str):
                text += f"Analyst stance: **{analyst.title().replace('_', ' ')}**. "

            if row["200‑Day Trend"] > 0:
                text += "Price is above long‑term trend."
            else:
                text += "Price is below long‑term trend."

            tab.markdown(f"- {text}")

    except Exception as e:
        tab.error(f"Error rendering buy analysis: {e}")
