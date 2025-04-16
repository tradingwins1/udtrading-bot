# finnhub_client.py
import requests
import time
from config import get_config

# Cache to store the last fetched price and timestamp
_price_cache = {}
_last_call = {}

def get_live_price(symbol):
    config = get_config()
    api_key = config["finnhub_api_key"]
    
    # Check if the price is in the cache and the cache is still valid (within 5 seconds)
    current_time = time.time()
    if symbol in _price_cache and symbol in _last_call:
        if current_time - _last_call[symbol] < 5:  # Cache for 5 seconds
            return _price_cache[symbol]

    # Fetch the price from Finnhub API
    url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        price = response.json().get("c")  # current price
        # Update the cache
        _price_cache[symbol] = price
        _last_call[symbol] = current_time
        return price
    else:
        raise Exception(f"Failed to fetch price: {response.text}")