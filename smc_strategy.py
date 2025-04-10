from db import log_trade
from scorer import score_setup
import pandas as pd
from order_block import check_order_block
from liquidity_sweep import check_liquidity_sweep
from rsi_divergence import check_rsi_divergence
from bos_detector import check_bos
from trend_detector import determine_trend
from red_news_filter import is_red_folder_event_today
from utils import check_3_bar_pattern, check_pop_and_fade, calculate_atr

# Dynamic strategy parameters, adjustable by the learning model
dynamic_params = {
    "volume_threshold": 1.5,  # Default volume spike requirement
    "avoid_low_atr": False,   # Default: Don't avoid low ATR trades
    "tighten_entry": False    # Default: Don't tighten entry criteria
}

def update_strategy_params(adjustments):
    """
    Update strategy parameters based on learning model adjustments.
    adjustments: Dict with suggested adjustments from bot_trainer.py.
    """
    global dynamic_params
    if 'increase_volume_requirement' in adjustments:
        dynamic_params['volume_threshold'] = 2.0  # Increase to 2x average
        print(f"Updated volume threshold to {dynamic_params['volume_threshold']}")
    if 'avoid_low_atr' in adjustments:
        dynamic_params['avoid_low_atr'] = True
        print("Enabled avoiding low ATR trades")
    if 'tighten_entry' in adjustments:
        dynamic_params['tighten_entry'] = True
        print("Enabled tightening entry criteria")

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

    # Enhanced Strategy Logic
    trend = determine_trend(df)
    pattern = check_3_bar_pattern(df)
    pop_fade = check_pop_and_fade(df, trend)

    # Check core conditions
    if not (pattern or pop_fade):
        df['signal'] = None
        return df[['Datetime', 'open', 'high', 'low', 'close', 'volume', 'signal']]

    if not (check_bos(df) or check_liquidity_sweep(df)):
        df['signal'] = None
        return df[['Datetime', 'open', 'high', 'low', 'close', 'volume', 'signal']]

    if not check_order_block(df):
        df['signal'] = None
        return df[['Datetime', 'open', 'high', 'low', 'close', 'volume', 'signal']]

    if not check_rsi_divergence(df):
        df['signal'] = None
        return df[['Datetime', 'open', 'high', 'low', 'close', 'volume', 'signal']]

    # Apply dynamic volume threshold
    if df['volume'].iloc[-1] < dynamic_params['volume_threshold'] * df['volume'][-10:-1].mean():
        df['signal'] = None
        return df[['Datetime', 'open', 'high', 'low', 'close', 'volume', 'signal']]

    if is_red_folder_event_today():
        df['signal'] = None
        return df[['Datetime', 'open', 'high', 'low', 'close', 'volume', 'signal']]

    # Avoid low ATR trades if specified
    if dynamic_params['avoid_low_atr']:
        atr = calculate_atr(df)
        avg_atr = calculate_atr(df[-50:])
        if atr < 0.5 * avg_atr:
            df['signal'] = None
            return df[['Datetime', 'open', 'high', 'low', 'close', 'volume', 'signal']]

    # Tighten entry criteria if specified (e.g., require more confluences)
    if dynamic_params['tighten_entry']:
        # Placeholder: Require an additional confluence (e.g., stricter volume or ATR)
        if df['volume'].iloc[-1] < 2.0 * df['volume'][-10:-1].mean():
            df['signal'] = None
            return df[['Datetime', 'open', 'high', 'low', 'close', 'volume', 'signal']]

    # Determine signal
    signal = 'buy' if pattern == "bullish" else 'sell'
    if pop_fade:
        signal = 'sell'

    df['signal'] = signal
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