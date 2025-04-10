import pandas as pd
from order_block import check_order_block
from liquidity_sweep import check_liquidity_sweep
from rsi_divergence import check_rsi_divergence
from bos_detector import check_bos
from trend_detector import determine_trend
from learn import get_stats
from utils import check_3_bar_pattern, check_pop_and_fade, calculate_atr

# Dynamic weights for confluences, adjustable by the learning model
dynamic_weights = {
    "has_3_bar_pattern": 1.0,
    "has_pop_fade": 1.0,
    "volume_spike": 1.0,
    "bos": 1.0,
    "liquidity_sweep": 1.0,
    "order_block": 1.0,
    "rsi_divergence": 1.0
}

def update_weights(adjustments):
    """
    Update dynamic weights based on learning model adjustments.
    adjustments: Dict with suggested adjustments from bot_trainer.py.
    """
    global dynamic_weights
    if 'reduce_3_bar_weight' in adjustments:
        dynamic_weights['has_3_bar_pattern'] *= 0.8  # Reduce weight by 20%
        print(f"Updated 3-bar pattern weight: {dynamic_weights['has_3_bar_pattern']}")
    if 'reduce_pop_fade_weight' in adjustments:
        dynamic_weights['has_pop_fade'] *= 0.8  # Reduce weight by 20%
        print(f"Updated Pop and Fade weight: {dynamic_weights['has_pop_fade']}")
    if 'increase_volume_requirement' in adjustments:
        dynamic_weights['volume_spike'] *= 1.2  # Increase weight by 20%
        print(f"Updated volume spike weight: {dynamic_weights['volume_spike']}")

def score_setup(candles_df):
    """
    Returns: Score (1-10), higher = better.
    """
    confluences = 0
    trend = determine_trend(candles_df)

    # Check confluences with dynamic weights
    if check_bos(candles_df):
        confluences += dynamic_weights['bos']
    if check_liquidity_sweep(candles_df):
        confluences += dynamic_weights['liquidity_sweep']
    if check_order_block(candles_df):
        confluences += dynamic_weights['order_block']
    if check_rsi_divergence(candles_df) in ["bullish", "bearish"]:
        confluences += dynamic_weights['rsi_divergence']
    if check_3_bar_pattern(candles_df) in ["bullish", "bearish"]:
        confluences += dynamic_weights['has_3_bar_pattern']
    if check_pop_and_fade(candles_df, trend):
        confluences += dynamic_weights['has_pop_fade']
    if candles_df['volume'].iloc[-1] > 1.5 * candles_df['volume'][-10:-1].mean():
        confluences += dynamic_weights['volume_spike']

    # Setup Quality (1-5)
    setup_score = min(5, confluences)

    # Entry Precision (1-3)
    entry_deviation = 0.003  # Placeholder: Calculate deviation from order block
    entry_score = 3 if entry_deviation < 0.005 else (2 if entry_deviation < 0.01 else 1)

    # Market Condition (1-2)
    atr = calculate_atr(candles_df)
    avg_atr = calculate_atr(candles_df[-50:])  # Average over last 50 candles
    market_score = 2 if atr > 0.5 * avg_atr else 1

    total_score = setup_score + entry_score + market_score
    return round(total_score, 1)