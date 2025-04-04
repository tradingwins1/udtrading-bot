import pandas as pd

def simulate_trailing_stop(trade, price_stream, trail_buffer=1.0):
    """
    Simulates trailing stop loss:
    - Adjusts SL as price moves favorably
    - Exits when price hits trailing SL
    """
    result = {
        'Date': trade['Date'],
        'Asset': trade.get('Asset', 'Unknown'),
        'Signal': trade['Signal'],
        'EntryPrice': trade['EntryPrice'],
        'InitialStopLoss': trade['StopLoss'],
        'TrailingStop': trade['StopLoss'],  # Updates dynamically
        'ExitPrice': None,
        'ExitType': 'Trailing SL',
        'PnL': 0.0
    }

    entry = trade['EntryPrice']
    trail_sl = trade['StopLoss']
    size = trade['PositionSize']
    side = trade['Signal']

    for price in price_stream:
        if side == 'BUY':
            if price - trail_sl >= trail_buffer:
                trail_sl = price - trail_buffer
            if price <= trail_sl:
                result.update({
                    'ExitPrice': price,
                    'TrailingStop': trail_sl,
                    'PnL': round((price - entry) * size, 2)
                })
                break

        elif side == 'SELL':
            if trail_sl - price >= trail_buffer:
                trail_sl = price + trail_buffer
            if price >= trail_sl:
                result.update({
                    'ExitPrice': price,
                    'TrailingStop': trail_sl,
                    'PnL': round((entry - price) * size, 2)
                })
                break

    return result
