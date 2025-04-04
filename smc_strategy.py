from db import log_trade
from scorer import score_setup
import pandas as pd

def smc_strategy(df):
    # Standardize datetime column
    if 'Datetime' not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df['Datetime'] = df.index
        elif 'date' in df.columns:
            df.rename(columns={'date': 'Datetime'}, inplace=True)

    # Required columns check
    required_cols = ['Datetime', 'open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    # Strategy Logic (Placeholder - replace with real SMC logic)
    df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['signal'] = df.apply(lambda row: 'buy' if row['ema_9'] > row['ema_21'] else 'sell', axis=1)

    return df[['Datetime', 'open', 'high', 'low', 'close', 'volume', 'signal']]

def execute_trade(setup_type, direction, entry_price, stop_loss, take_profit, asset="MNQ"):
    rr_ratio = round(abs(entry_price - take_profit) / abs(stop_loss - entry_price), 2)
    confidence_score = score_setup(setup_type)

    # Adjusted Risk Based on Confidence
    base_risk = 50  # $ risk per trade
    adjusted_risk = round(base_risk * (confidence_score / 10), 2)

    # SL Distance
    sl_distance = abs(entry_price - stop_loss)
    if sl_distance == 0:
        print("⚠️ Stop loss is at entry price. Invalid setup.")
        return

    # Asset-specific multiplier
    asset_multiplier = {
        # Futures
        "MNQ": 2.0, "MES": 5.0, "MCL": 10.0, "MGC": 10.0,
        # Forex
        "EURUSD": 1.0, "USDJPY": 1.0,
        # Stocks
        "AAPL": 1.0, "TSLA": 1.0, "NVDA": 1.0,
        # Crypto
        "BTCUSD": 1.0, "ETHUSD": 1.0, "SOLUSD": 1.0,
    }

    multiplier = asset_multiplier.get(asset.upper(), 1.0)
    effective_sl = sl_distance * multiplier
    position_size = round(adjusted_risk / effective_sl, 4)

    print(f"\n🔍 Setup: {setup_type}")
    print(f"📊 Confidence Score: {confidence_score}/10")
    print(f"💸 Adjusted Risk: ${adjusted_risk}")
    print(f"📉 SL Distance: {sl_distance} * ${multiplier} = ${effective_sl}")
    print(f"📦 Position Size: {position_size} ({asset})")

    if confidence_score >= 6.0:
        print("✅ High confidence — logging trade.")

        log_trade({
            "asset": asset,
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": None,  # To be updated when SL/TP is hit
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "rr_ratio": rr_ratio,
            "result": None,  # Will be updated in real-time tracking
            "setup_type": setup_type,
            "confidence_score": confidence_score,
            "notes": f"Risk: ${adjusted_risk} | SL: {sl_distance} | Size: {position_size}"
        })

        print(f"📥 Trade logged. Awaiting price action.")
    else:
        print("❌ Low confidence — skipping trade.")
