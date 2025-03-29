
import pandas as pd

def smc_strategy(df):
    if df.empty:
        return pd.DataFrame()

    # === Calculate Indicators ===
    df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['rsi'] = compute_rsi(df['close'])

    # === Core Strategy Logic ===
    df['long_signal'] = (
        (df['ema_9'] > df['ema_21']) &
        (df['close'] > df['vwap']) &
        (df['rsi'] > 50)
    )

    df['short_signal'] = (
        (df['ema_9'] < df['ema_21']) &
        (df['close'] < df['vwap']) &
        (df['rsi'] < 50)
    )

    df['signal'] = None
    df.loc[df['long_signal'], 'signal'] = 'buy'
    df.loc[df['short_signal'], 'signal'] = 'sell'

    return df[['Datetime', 'open', 'high', 'low', 'close', 'volume', 'signal']]

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
