
def is_valid_pop_and_fade(candle, trend, volume_spike, wick_ratio):
    '''
    Determines if the current candle setup meets Pop and Fade criteria.

    Args:
        candle (dict): Contains candle info like 'rejection' (bool)
        trend (str): 'bullish' or 'bearish'
        volume_spike (bool): True if volume exceeds threshold
        wick_ratio (float): Wick-to-body ratio

    Returns:
        bool: True if valid Pop and Fade entry, False otherwise
    '''
    return (
        trend == "bearish"
        and candle.get('rejection', False)
        and wick_ratio > 1.5
        and volume_spike
    )
