
from utils import fetch_ibkr_data
from smc_strategy import smc_strategy
from execution import execute_trades

# Simulated asset for dry run (change as needed)
symbol = 'MNQ'
asset_type = 'futures'

def run_dry_test():
    print(f"📊 Fetching data for {symbol}...")
    df = fetch_ibkr_data(symbol)

    print("🔍 Running strategy...")
    signals_df = smc_strategy(df)

    print("🚀 Executing simulated trade (dry run)...")
    execute_trades(signals_df, symbol, asset_type=asset_type)

# Run the test
if __name__ == "__main__":
    run_dry_test()
