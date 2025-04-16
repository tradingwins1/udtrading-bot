# live_tracker.py
import time
import pandas as pd
from finnhub_client import get_live_price
from discord_alert import send_alert
from config import get_config

config = get_config()

open_trades = []

def track_live_trades():
    global open_trades

    if not open_trades:
        print("No active trades to track.")
        return

    print("\n🔄 Tracking live trades...")
    closed_trades = []

    for trade in open_trades:
        symbol = trade['Symbol']
        entry = trade['Entry']
        sl = trade['StopLoss']
        tp = trade['TakeProfit']
        direction = trade['Direction']

        try:
            current_price = get_live_price(symbol)
        except Exception as e:
            print(f"Error fetching price for {symbol}: {e}")
            continue

        print(f"{symbol} current: {current_price} | SL: {sl} | TP: {tp}")

        # Exit Conditions
        if direction == 'BUY' and current_price >= tp:
            comment = "TP Hit ✅ — Price respected liquidity sweep zone"
            status = 'TP'
        elif direction == 'BUY' and current_price <= sl:
            comment = "SL Hit ❌ — Entry invalidated by wick rejection"
            status = 'SL'
        elif direction == 'SELL' and current_price <= tp:
            comment = "TP Hit ✅ — Price respected imbalance"
            status = 'TP'
        elif direction == 'SELL' and current_price >= sl:
            comment = "SL Hit ❌ — Entry invalidated by bullish engulfing"
            status = 'SL'
        else:
            continue  # No exit condition met

        # Send Discord Alert
        pnl = round(abs(entry - current_price), 2)
        pnl = pnl if status == 'TP' else -pnl

        send_alert(
            symbol=symbol,
            side=direction,
            entry=entry,
            sl=sl,
            tp=tp,
            timeframe="5m",
            confidence=8,
            alert_type="scalping",
            reason=f"{status}: {comment} (PnL: {pnl})"
        )

        closed_trades.append(trade)

    # Remove closed trades from the open list
    open_trades = [t for t in open_trades if t not in closed_trades]

    # Add a sleep to reduce CPU usage
    time.sleep(15)  # Sleep for 1 second between checks to prevent high CPU usage

def add_live_trade(symbol, entry, sl, tp, direction):
    global open_trades
    open_trades.append({
        'Symbol': symbol,
        'Entry': entry,
        'StopLoss': sl,
        'TakeProfit': tp,
        'Direction': direction
    })
    print(f"✅ Live trade added for {symbol} at {entry}.")