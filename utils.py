import yfinance as yf
from binance.client import Client
import pandas as pd
import datetime
import requests
import time
from config import load_config

config = load_config()

# -----------------------------
# 🟡 Binance OHLC Fetch
# -----------------------------
def get_binance_ohlc(symbol, interval='5m', lookback='1 day'):
    print(f"📡 Fetching {symbol} OHLC from Binance...")
    client = Client()
    klines = client.get_historical_klines(symbol, interval, lookback)

    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])

    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('date', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

# -----------------------------
# 🟢 Yahoo Finance OHLC Fetch
# -----------------------------
def get_yahoo_ohlc(symbol, interval='5m', lookback='7d'):
    print(f"📡 Fetching {symbol} OHLC from Yahoo Finance...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(interval=interval, period=lookback)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    df.dropna(inplace=True)
    return df

# -----------------------------
# 🛑 Deprecated Finnhub function for fallback only
# -----------------------------
def get_finnhub_ohlc(symbol, timeframe, from_unix, to_unix, api_key):
    print(f"📡 Fetching {symbol} OHLC data ({timeframe}) from Finnhub...")
    url = f"https://finnhub.io/api/v1/stock/candle?symbol={symbol}&resolution={timeframe}&from={from_unix}&to={to_unix}&token={api_key}"
    response = requests.get(url)
    data = response.json()

    if 's' not in data or data['s'] != 'ok':
        print("❌ Finnhub response error:", data)
        raise Exception("Failed to fetch OHLC data from Finnhub")

    df = pd.DataFrame({
        'date': pd.to_datetime(data['t'], unit='s'),
        'open': data['o'],
        'high': data['h'],
        'low': data['l'],
        'close': data['c'],
        'volume': data['v']
    })
    return df
