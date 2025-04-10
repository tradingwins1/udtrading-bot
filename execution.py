# execution.py
import os
import time
from ibkr_client import ib, connect_ibkr
from discord_alert import send_alert
from dotenv import load_dotenv
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def execute_trades(signals_df, symbol, asset_type, candles_df):
    print("🚀 Executing trade...")
    if signals_df is None or signals_df.empty or signals_df['signal'].iloc[-1] is None:
        print(f"[Execution] No actionable signal for {symbol}")
        try:
            send_alert(
                symbol=symbol,
                side="N/A",
                entry=0.0,
                sl=0.0,
                tp=0.0,
                timeframe="5m" if asset_type == "stock" else "15m" if asset_type == "crypto" else "4h",
                confidence=0,
                alert_type="scalp" if asset_type in ["stock", "crypto"] else "swing",
                reason="No actionable signal"
            )
        except Exception as e:
            logging.warning(f"❌ Failed to send no-trade alert: {e}")
        return

    connect_ibkr()

    for _, row in signals_df.iterrows():
        if row['signal'] is None:
            continue

        side = row['signal'].upper()
        entry = row['close']
        sl = entry - 1.0 if side == 'BUY' else entry + 1.0  # Example SL logic
        tp = entry + 2.0 if side == 'BUY' else entry - 2.0  # Example TP logic

        # Calculate confidence score
        from scorer import score_setup
        confidence = score_setup(candles_df)

        if confidence < 6:
            print(f"[Execution] Low confidence score ({confidence}) for {symbol}, skipping trade")
            try:
                send_alert(
                    symbol=symbol,
                    side=side,
                    entry=entry,
                    sl=sl,
                    tp=tp,
                    timeframe="5m" if asset_type == "stock" else "15m" if asset_type == "crypto" else "4h",
                    confidence=confidence,
                    alert_type="scalp" if asset_type in ["stock", "crypto"] else "swing",
                    reason="No Trade is better than losing a trade"
                )
            except Exception as e:
                logging.warning(f"❌ Failed to send no-trade alert: {e}")
            continue

        print(f"[Execution] {side} signal for {symbol}")
        print(f"Entry: {entry} | SL: {sl} | TP: {tp}")

        try:
            send_alert(
                symbol=symbol,
                side=side,
                entry=entry,
                sl=sl,
                tp=tp,
                timeframe="5m" if asset_type == "stock" else "15m" if asset_type == "crypto" else "4h",
                confidence=confidence,
                alert_type="scalp" if asset_type in ["stock", "crypto"] else "swing",
                reason="Strategy signal triggered"
            )
        except Exception as e:
            logging.error(f"❌ Discord alert failed during execution: {e}")

        time.sleep(0.3)  # Wait 300ms to avoid Discord rate limiting

    ib.disconnect()
    print("🔌 Disconnected from IBKR")