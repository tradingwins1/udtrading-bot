
from ibkr_client import connect_ibkr, disconnect_ibkr, place_order
from discord_alert import send_alert

def execute_trades(signals_df, symbol, asset_type):
    connect_ibkr()

    latest_signal = signals_df.iloc[-1] if not signals_df.empty else None
    if latest_signal is not None and latest_signal['signal'] in ['buy', 'sell']:
        side = latest_signal['signal']
        entry = float(latest_signal['close'])
        sl = entry + 0.5 if side == 'sell' else entry - 0.5
        tp = entry - 1.0 if side == 'sell' else entry + 1.0

        print(f"[Execution] {side.upper()} signal for {symbol}")
        place_order(symbol, asset_type, side.upper(), 1)

        send_alert(
            message=f"[Executed] {side.upper()} {symbol} @ {entry}",
            side=side,
            entry=entry,
            sl=sl,
            tp=tp
        )
    else:
        message = f"[Execution] No actionable signal for {symbol}"
        print(message)
        send_alert(message, side="N/A", entry=0.0, sl=0.0, tp=0.0)

    disconnect_ibkr()
