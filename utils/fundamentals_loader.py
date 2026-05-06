import yfinance as yf

def _safe_get(info, key, default=0):
    value = info.get(key)
    if value in (None, "None", "N/A"):
        return default
    return value

def load_fundamentals(tickers, full_prices=None):
    fundamentals = {}

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            fundamentals[ticker] = {
                "PE": _safe_get(info, "trailingPE"),
                "PB": _safe_get(info, "priceToBook"),
                "DividendYield": _safe_get(info, "dividendYield"),
                "gross_margins": _safe_get(info, "grossMargins"),
                "profit_margins": _safe_get(info, "profitMargins"),
                "revenue": _safe_get(info, "totalRevenue"),
            }

        except Exception as e:
            fundamentals[ticker] = {"error": str(e)}

    if full_prices is not None:
        fundamentals["full_prices"] = full_prices

    return fundamentals
