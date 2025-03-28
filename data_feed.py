import pandas as pd
from config import load_config
from utils import get_yahoo_ohlc, get_binance_ohlc
from smc_strategy import smc_strategy
from risk_manager import apply_risk_management
from execution import execute_trade_with_trailing_sl
from discord_alert import send_discord_alert
from trade_logger import log_trade
from performance_summary import print_summary
from live_dashboard import generate_pnl_dashboard
print("✅ Bot started... Running data_feed.py")

config = load_config()

symbol = config["symbol"]
timeframe = config["timeframe"]
plot_dashboard = config.get("plot_live_dashboard", True)

# Determine data source based on asset type
is_crypto = symbol.upper().startswith("BINANCE:")
clean_symbol = symbol.split(":")[-1] if is_crypto else symbol.upper()

# -------------------------------
# 📈 Fetch OHLC Data
# -------------------------------
if is_crypto:
    df = get_binance_ohlc(clean_symbol, interval='5m', lookback='1 day')
else:
    df = get_yahoo_ohlc(clean_symbol, interval='5m', lookback='7d')

print("\n--- Last 5 rows of data ---")
print(df.tail())

# -------------------------------
# 📊 Detect Smart Money Setups
# -------------------------------
signals_df = smc_strategy(df)

if signals_df.empty:
    print("🚫 No valid trade setups detected.")
    exit()

# -------------------------------
# 📉 Apply Risk Management
# -------------------------------
managed_signals = apply_risk_management(signals_df, config)

# -------------------------------
# 💰 Execute Trades
# -------------------------------
executed_trades = execute_trade_with_trailing_sl(managed_signals)

# -------------------------------
# 📤 Send Alerts and Log Trades
# -------------------------------
for _, row in executed_trades.iterrows():
    log_trade(row)
    send_discord_alert(row)

# -------------------------------
# 📈 Summary and Dashboard
# -------------------------------
print_summary()
if plot_dashboard:
    generate_pnl_dashboard()
input("✅ Done. Press Enter to exit...")
