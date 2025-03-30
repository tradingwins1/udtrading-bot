import os
import argparse
import pandas as pd
from dotenv import load_dotenv
from utils import fetch_data_by_asset_type
from smc_strategy import smc_strategy
from execution import execute_trades

load_dotenv()

def normalize_dataframe(df):
    df.columns = [c.lower() for c in df.columns]  # Normalize all columns to lowercase

    if 'date' in df.columns:
        df.rename(columns={'date': 'datetime'}, inplace=True)
    elif 'index' in df.columns:
        df.rename(columns={'index': 'datetime'}, inplace=True)

    if 'datetime' not in df.columns:
        df['datetime'] = pd.date_range(end=datetime.now(), periods=len(df), freq='1h')

    df.rename(columns={'datetime': 'Datetime'}, inplace=True)

    # Ensure all required price columns exist
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col not in df.columns:
            df[col] = 0.0

    return df[['Datetime', 'open', 'high', 'low', 'close', 'volume']]


def run_single_asset(symbol, asset_type):
    print(f"📊 Fetching data for {symbol} ({asset_type})...")
    df = fetch_data_by_asset_type(symbol, asset_type)
    if df is None or df.empty:
        print(f"⚠️ No data fetched for {symbol}")
        return
    df = normalize_dataframe(df)
    print("🔍 Running strategy...")
    signals_df = smc_strategy(df)
    print("🚀 Executing trade...")
    execute_trades(signals_df, symbol, asset_type)

def run_default_assets():
    assets = [
        {'symbol': 'MNQ', 'type': 'futures'},
        {'symbol': 'MGC', 'type': 'futures'},
        {'symbol': 'AAPL', 'type': 'stock'},
        {'symbol': 'EURUSD', 'type': 'forex'},
        {'symbol': 'USDJPY', 'type': 'forex'}
    ]
    for asset in assets:
        run_single_asset(asset['symbol'], asset['type'])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("symbol", nargs="?", help="Symbol to run")
    parser.add_argument("--asset_type", help="Type: stock, futures, forex")
    args = parser.parse_args()

    if args.symbol and args.asset_type:
        run_single_asset(args.symbol, args.asset_type)
    else:
        run_default_assets()

if __name__ == "__main__":
    main()
