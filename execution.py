
from ibkr_client import connect_ibkr, place_order, disconnect_ibkr
from discord_alert import send_alert

def execute_trades(df, symbol, asset_type="stock", test_mode=False):
    print("🚀 Executing trade...")

    connect_ibkr()

    if 'signal' not in df.columns or df.empty:
        print(f"[Execution] No actionable signal for {symbol}")
        send_alert(f"[Execution] No actionable signal for {symbol}")
        disconnect_ibkr()
        return

    latest = df.iloc[-1]
    signal = latest['signal']
    price = latest['close']
    sl = price + 0.5 if signal == "SELL" else price - 0.5
    tp = price - 1.0 if signal == "SELL" else price + 1.0
    confidence = 8

    if signal == "BUY":
        if not test_mode:
            place_order(symbol, asset_type, "BUY", 1)
        alert_msg = f"[LIVE ALERT] BUY {symbol} | Entry: {price}, SL: {sl}, TP: {tp}, TF: 5m, Confidence: {confidence}/10"
    elif signal == "SELL":
        if not test_mode:
            place_order(symbol, asset_type, "SELL", 1)
        alert_msg = f"[LIVE ALERT] SELL {symbol} | Entry: {price}, SL: {sl}, TP: {tp}, TF: 5m, Confidence: {confidence}/10"
    else:
        alert_msg = f"[Execution] No actionable signal for {symbol}"

    print(alert_msg)
    send_alert(alert_msg)
    disconnect_ibkr()
