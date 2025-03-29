import os
import pandas as pd
from dotenv import load_dotenv
from utils import fetch_ibkr_data
from smc_strategy import smc_strategy
from execution import execute_trades

load_dotenv()

def main():
    assets = [
        {'symbol': 'MNQ', 'type': 'futures'},
        {'symbol': 'MGC', 'type': 'futures'},
        {'symbol': 'AAPL', 'type': 'stock'},
        {'symbol': 'EURUSD=X', 'type': 'forex'},
    ]
    for asset in assets:
        symbol = asset['symbol']
        asset_type = asset['type']
        print(f"📊 Fetching data for {symbol} ({asset_type})...")
        if asset_type == 'futures':
            df = fetch_ibkr_data(symbol)
        else:
            continue  # Extend this to support forex/stocks later
        if df is None or df.empty:
            print(f"⚠️ No data fetched for {symbol}")
            continue
        print("🔍 Running strategy...")
        signals_df = smc_strategy(df)
        print("🚀 Executing trade...")
        execute_trades(signals_df, symbol, asset_type)

if __name__ == "__main__":
    main()