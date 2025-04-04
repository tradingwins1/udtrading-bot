import yfinance as yf

def get_latest_price(symbol):
    """
    Returns the latest price for a given asset symbol using Yahoo Finance.
    """
    try:
        ticker = yf.Ticker(symbol)
        price = ticker.history(period="1d", interval="1m")['Close'].iloc[-1]
        return round(price, 2)
    except Exception as e:
        print(f"❌ Error fetching price for {symbol}: {e}")
        return None
