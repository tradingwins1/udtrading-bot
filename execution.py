# execution.py
import os
import time
from ibkr_client import ib, connect_ibkr
from discord_alert import send_alert
from dotenv import load_dotenv
import logging
import pandas as pd

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def check_entry(trader, candles_df: pd.DataFrame, contract):
    """
    Evaluates entry conditions for a trade based on candlestick data.
    
    Args:
        trader: IBKRTrader instance for interacting with IBKR.
        candles_df: DataFrame containing candlestick data.
        contract: Contract object for the asset.
    
    Returns:
        dict: Entry decision with keys 'entry_price', 'stop_loss', 'take_profit', 'direction', and 'score'.
              Returns None if no entry condition is met.
    """
    if candles_df.empty:
        logging.info("No candlestick data available for entry evaluation.")
        return None

    # Get the latest and previous candles
    latest_candle = candles_df.iloc[-1]
    previous_candle = candles_df.iloc[-2] if len(candles_df) > 1 else None

    if previous_candle is None:
        logging.info("Not enough candlestick data for entry evaluation.")
        return None

    # Simple breakout logic: Buy if the latest close is above the previous high
    entry_price = latest_candle['close']
    direction = None
    score = 0.5  # Default confidence score

    if latest_candle['close'] > previous_candle['high']:
        direction = 'buy'
        stop_loss = previous_candle['low']
        take_profit = entry_price + 2 * (entry_price - stop_loss)  # 2:1 risk-reward ratio
        score = 0.95  # High confidence for a breakout
    # Sell if the latest close is below the previous low
    elif latest_candle['close'] < previous_candle['low']:
        direction = 'sell'
        stop_loss = previous_candle['high']
        take_profit = entry_price - 2 * (stop_loss - entry_price)  # 2:1 risk-reward ratio
        score = 0.95

    if direction:
        logging.info(f"Entry condition met: {direction} at {entry_price} with SL {stop_loss} and TP {take_profit}")
        return {
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'direction': direction,
            'score': score
        }
    logging.info("No entry condition met.")
    return None

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