import os
import yfinance as yf
import pandas as pd
from ib_insync import IB, Stock, Forex, Future, Contract, util
from datetime import datetime, timedelta

# Environment variables
USE_BINANCE = os.getenv("USE_BINANCE", "false").lower() == "true"
IBKR_ENABLED = os.getenv("USE_IBKR", "true").lower() == "true"

# IBKR connection
ib = IB()
def connect_ibkr():
    if not ib.isConnected():
        ib.connect("127.0.0.1", 7497, clientId=1)
        print("✅ Live connection to IBKR established for data fetch")

def disconnect_ibkr():
    if ib.isConnected():
        ib.disconnect()

# === Resolve Smart Futures Contract ===
def resolve_futures_contract(symbol):
    print(f"📊 Resolving active future contract for {symbol}...")
    
    try:
        contracts = ib.reqContractDetails(Future(symbol=symbol, exchange="GLOBEX", currency="USD"))
        if not contracts:
            contracts = ib.reqContractDetails(Future(symbol=symbol, exchange="NYMEX", currency="USD"))
    except Exception as e:
        print(f"❌ Contract request failed: {e}")
        return None

    valid_contracts = []
    for detail in contracts:
        expiry = detail.contract.lastTradeDateOrContractMonth
        try:
            expiry_dt = datetime.strptime(expiry, "%Y%m")
            if expiry_dt > datetime.utcnow():
                valid_contracts.append((expiry_dt, detail.contract))
        except:
            continue

    if not valid_contracts:
        print(f"❌ No non-expired contracts found for {symbol}")
        return None

    valid_contracts.sort(key=lambda x: x[0])
    selected = valid_contracts[0][1]
    print(f"✅ Using contract {selected.localSymbol} | Expiry: {selected.lastTradeDateOrContractMonth} | ConId: {selected.conId}")
    return selected

# === IBKR Historical Data ===
def fetch_ibkr_data(symbol, interval="1 hour", lookback=100, asset_type="futures"):
    connect_ibkr()

    if asset_type == "futures":
        contract = resolve_futures_contract(symbol)
    elif asset_type == "forex":
        contract = Forex(symbol)
    elif asset_type == "stock":
        contract = Stock(symbol, "SMART", "USD")
    else:
        print(f"⚠️ Unknown asset type: {asset_type}")
        return None

    if not contract:
        print(f"❌ Could not resolve {asset_type} contract for {symbol}")
        return None

    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr="2 D",
        barSizeSetting=interval,
        whatToShow="TRADES",
        useRTH=False,
        formatDate=1
    )

    if not bars:
        print(f"❌ IBKR fetch failed: No data returned for {symbol}")
        return None

    df = util.df(bars)
    df.set_index("date", inplace=True)
    df.index.name = "Datetime"
    return df

# === Yahoo Backup ===
def fetch_yahoo_data(symbol, interval="5m", lookback=30):
    print(f"📉 Fetching Yahoo data for {symbol} as fallback...")
    try:
        df = yf.download(symbol, period=f"{lookback}d", interval=interval)
        df.dropna(inplace=True)
        df.index.name = "Datetime"
        return df
    except Exception as e:
        print(f"❌ Yahoo fallback failed: {e}")
        return None

# === Combined Fetch Logic ===
def fetch_data_by_asset_type(symbol, asset_type):
    if IBKR_ENABLED:
        df = fetch_ibkr_data(symbol, asset_type=asset_type)
        if df is not None:
            return df

    if asset_type == "crypto":
        return fetch_yahoo_data(symbol, interval="15m", lookback=10)
    elif asset_type == "forex":
        return fetch_yahoo_data(symbol + "=X", interval="1h", lookback=30)
    else:
        return fetch_yahoo_data(symbol, interval="5m", lookback=30)

# === Binance fallback removed due to restrictions ===
