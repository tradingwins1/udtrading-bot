import requests
from config import load_config

def get_live_price(symbol):
    config = load_config()
    api_key = config["finnhub_api_key"]
    url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("c")  # current price
    else:
        raise Exception(f"Failed to fetch price: {response.text}")
