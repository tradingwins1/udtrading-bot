
import yfinance as yf
import pandas as pd
from datetime import datetime
from ib_insync import *

# Connect to IBKR
ib = IB()
try:
    ib.connect('127.0.0.1', 7497, clientId=2)
    print("✅ Live connection to IBKR established for data fetch")
except Exception as e:
    print(f"⚠️ Failed to connect to IBKR for data: {e}")

# Known conIds if needed (you can expand this)
futures_conids = {
    'MGC': 559041217,
}

def get_nearest_futures_expiry():
    today = datetime.today()
    year = today.year
    month = today.month
    codes = [(3, 'H'), (6, 'M'), (9, 'U'), (12, 'Z')]
    for m, code in codes:
        if month <= m:
            return f"{year}{str(m).zfill(2)}"
    return f"{year+1}03"

def fetch_ibkr_data(symbol, interval='1h', lookback=100):
    expiry = get_nearest_futures_expiry()
    print(f"[IBKR] Fetching live data for {symbol} | Expiry: {expiry}")
    try:
        if symbol in futures_conids and futures_conids[symbol]:
            contract = Future(conId=futures_conids[symbol], exchange='GLOBEX')
        else:
            contract = Future(symbol=symbol, lastTradeDateOrContractMonth=expiry, exchange='GLOBEX', currency='USD')
        ib.qualifyContracts(contract)
        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr='2 D',
            barSizeSetting='1 hour',
            whatToShow='TRADES',
            useRTH=False,
            formatDate=1
        )
        df = util.df(bars)
        df.reset_index(inplace=True)
        df.rename(columns={'date': 'Datetime'}, inplace=True)
        return df.tail(lookback)
    except Exception as e:
        print(f"❌ IBKR fetch failed: {e}")
        return pd.DataFrame()

def fetch_yahoo_data(symbol, interval='1h', lookback=100):
    try:
        period = '30d' if interval == '5m' else '90d'
        df = yf.download(symbol, period=period, interval=interval)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0].lower() for col in df.columns]
        else:
            df.columns = [col.lower() for col in df.columns]
        df.reset_index(inplace=True)
        df.rename(columns={'date': 'Datetime'}, inplace=True)
        return df.tail(lookback)
    except Exception as e:
        print(f"[Yahoo] Failed to fetch {symbol}: {e}")
        return pd.DataFrame()
