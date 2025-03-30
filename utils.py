import os
import pandas as pd
import yfinance as yf
from ib_insync import IB, Forex, Stock, Future, Contract, util
from datetime import datetime, timedelta
from binance.client import Client
from dotenv import load_dotenv

load_dotenv()

ib = IB()

# Initialize Binance client (no keys to avoid geo-restriction errors)
try:
    binance_client = Client()
except:
    binance_client = None

# Predefined fallback conIds for key futures
FALLBACK_CONIDS = {
    "MNQ": 628145136,
    "MGC": 642542132,
    "MES": 628146079,
    "MCL": 463997743,
    "PL": 434221445
}

def fetch_yahoo_data(symbol, interval='5m', lookback=30):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=lookback)

    try:
        print(f"📉 Fetching Yahoo data for {symbol}...")
        df = yf.download(symbol, start=start_date, end=end_date, interval=interval, auto_adjust=True)

        if df is None or df.empty:
            print(f"❌ Yahoo returned no data for {symbol}")
            return None

        # Flatten multi-index columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0].lower() for col in df.columns]
        else:
            df.columns = [str(col).lower() for col in df.columns]

        print("✅ Yahoo raw columns:", list(df.columns))

        # Reset index to move date into a column
        df.reset_index(inplace=True)

        # Ensure 'Datetime' column exists without conflict
        if 'Datetime' not in df.columns:
            if 'date' in df.columns:
                df.rename(columns={'date': 'Datetime'}, inplace=True)
            elif 'index' in df.columns:
                df.rename(columns={'index': 'Datetime'}, inplace=True)
            else:
                df['Datetime'] = pd.date_range(start=start_date, periods=len(df), freq=interval)
        elif 'datetime' in df.columns:
            df.rename(columns={'datetime': 'Datetime'}, inplace=True)

        df['Datetime'] = pd.to_datetime(df['Datetime'])

        required_cols = ['Datetime', 'open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                print(f"❌ Missing column {col} in Yahoo data for {symbol}")
                return None

        return df[required_cols]

    except Exception as e:
        print(f"❌ Yahoo data fetch failed for {symbol}: {e}")
        return None


def fetch_binance_data(symbol, interval='15m', lookback=100):
    if not binance_client:
        print("⚠️ Binance client unavailable")
        return None
    try:
        klines = binance_client.get_klines(symbol=symbol, interval=interval, limit=lookback)
        df = pd.DataFrame(klines, columns=[
            'Datetime', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'num_trades',
            'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
        ])
        df['Datetime'] = pd.to_datetime(df['Datetime'], unit='ms')
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        return df[['Datetime', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        print(f"❌ Binance fetch error for {symbol}: {e}")
        return None

def connect_ibkr():
    if not ib.isConnected():
        ib.connect("127.0.0.1", 7497, clientId=1)
        print("✅ Live connection to IBKR established for data fetch")

def disconnect_ibkr():
    if ib.isConnected():
        ib.disconnect()
        print("🔌 Disconnected from IBKR")

def resolve_ibkr_future_contract(symbol):
    print(f"📊 Resolving active future contract for {symbol}...")
    try:
        fallback_conid = FALLBACK_CONIDS.get(symbol)
        if fallback_conid:
            contract = Contract(conId=fallback_conid, exchange='GLOBEX')
            details = ib.reqContractDetails(contract)
            if details:
                return details[0].contract
            else:
                print(f"⚠️ Fallback ConId failed for {symbol}, trying dynamic lookup...")

        for exchange in ['GLOBEX', 'NYMEX']:
            contracts = ib.reqContractDetails(Future(symbol=symbol, exchange=exchange, currency="USD"))
            active_contracts = [cd.contract for cd in contracts if cd.contract.lastTradeDateOrContractMonth > datetime.now().strftime('%Y%m')]
            if active_contracts:
                return active_contracts[0]
    except Exception as e:
        print(f"❌ Error resolving contract: {e}")
    print(f"❌ No valid contracts found for {symbol}")
    return None

def fetch_ibkr_data(symbol, asset_type):
    try:
        connect_ibkr()
        if asset_type == 'forex':
            contract = Forex(symbol, exchange='IDEALPRO')
        elif asset_type == 'stock':
            contract = Stock(symbol, 'SMART', 'USD')
        elif asset_type == 'futures':
            contract = resolve_ibkr_future_contract(symbol)
            if not contract:
                print(f"❌ Could not resolve futures contract for {symbol}")
                return None
        else:
            raise ValueError("Unknown asset_type")

        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr='2 D',
            barSizeSetting='1 hour' if asset_type == 'forex' else '5 mins',
            whatToShow='TRADES',
            useRTH=False,
            formatDate=1
        )
        if not bars:
            raise ValueError("No data returned for " + symbol)

        df = util.df(bars)
        df.rename(columns={"date": "Datetime"}, inplace=True)
        return df[['Datetime', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        print(f"❌ IBKR fetch failed: {e}")
        return None

def fetch_data_by_asset_type(symbol, asset_type):
    if asset_type == 'forex':
        data = fetch_ibkr_data(symbol, 'forex')
        if data is None or data.empty:
            print(f"📉 Fetching Yahoo data for {symbol}=X as fallback...")
            data = fetch_yahoo_data(symbol + "=X", interval='1h')
    elif asset_type == 'stock':
        data = fetch_ibkr_data(symbol, 'stock')
        if data is None or data.empty:
            print(f"📉 Fetching Yahoo data for {symbol} as fallback...")
            data = fetch_yahoo_data(symbol, interval='5m')
    elif asset_type == 'futures':
        data = fetch_ibkr_data(symbol, 'futures')
    elif asset_type == 'crypto':
        data = fetch_binance_data(symbol)
    else:
        print(f"❌ Unknown asset type: {asset_type}")
        return None

    if data is None or data.empty:
        print(f"❌ No valid data returned for {symbol}")
        return None

    data = data.copy()
    data.columns = ["Datetime" if str(c).lower() == "datetime" else c for c in data.columns]
    return data
