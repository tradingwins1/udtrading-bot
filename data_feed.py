
import argparse
from utils import fetch_ibkr_data, fetch_yahoo_data
from smc_strategy import smc_strategy
from execution import execute_trades

# === Asset Type Detection & Symbol Normalization ===
def detect_asset_type(symbol):
    if symbol.upper() in ['MNQ', 'MES', 'MGC', 'MCL', 'PL']:
        return 'futures'
    elif symbol.upper() in ['USDJPY', 'EURUSD', 'GBPUSD']:
        return 'forex'
    elif symbol.upper().endswith('-USD') or symbol.upper().endswith('USD'):
        return 'crypto'
    else:
        return 'stock'

def normalize_symbol(symbol, asset_type):
    if asset_type == 'forex' and not symbol.endswith('=X'):
        return symbol + '=X'
    return symbol

def fetch_data(symbol, asset_type):
    if asset_type == 'futures':
        return fetch_ibkr_data(symbol, interval='1h', lookback=100)
    elif asset_type == 'forex':
        return fetch_yahoo_data(symbol, interval='4h', lookback=100)
    elif asset_type == 'crypto':
        return fetch_yahoo_data(symbol, interval='1h', lookback=100)
    elif asset_type == 'stock':
        return fetch_yahoo_data(symbol, interval='5m', lookback=100)
    else:
        print("⚠️ Unknown asset type. Using 1h Yahoo fallback.")
        return fetch_yahoo_data(symbol, interval='1h', lookback=100)

# === Main CLI Entry ===
def main():
    parser = argparse.ArgumentParser(description="Run trading strategy dynamically")
    parser.add_argument("symbol", type=str, help="Symbol to trade (e.g., MNQ, AAPL, ETH-USD)")
    parser.add_argument("--asset_type", type=str, choices=["futures", "forex", "crypto", "stock"],
                        help="Optional: Specify asset type. If omitted, auto-detected.")
    args = parser.parse_args()

    raw_symbol = args.symbol.upper()
    asset_type = args.asset_type if args.asset_type else detect_asset_type(raw_symbol)
    symbol = normalize_symbol(raw_symbol, asset_type)

    print(f"📊 Fetching data for {symbol} ({asset_type})...")
    df = fetch_data(symbol, asset_type)

    print("🔍 Running strategy...")
    signals_df = smc_strategy(df)

    print("🚀 Executing trade...")
    execute_trades(signals_df, symbol, asset_type=asset_type)

if __name__ == "__main__":
    main()
