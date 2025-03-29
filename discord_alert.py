
def send_alert(symbol, side, entry, sl, tp, confidence=8, tf='5m'):
    alert_msg = f"[LIVE ALERT] {side} {symbol} | Entry: {entry}, SL: {sl}, TP: {tp}, TF: {tf}, Confidence: {confidence}/10"
    print(alert_msg)
