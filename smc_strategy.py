import pandas as pd
from bos_detector import detect_bos
from fvg_detector import detect_fvg
from strategy import (
    wick_to_body_ratio,
    is_high_impact_news_nearby,
    detect_volume_spike,
    is_near_pdh_or_pdl
)
from config import load_config

config = load_config()

def smc_strategy(df):
    df = df.copy()
    df.reset_index(inplace=True)
    df['date'] = pd.to_datetime(df['Datetime'])  # Yahoo/Binance data fix
    df['date'] = df['date'].dt.tz_localize(None)  # Remove timezone

    bos_signals = detect_bos(df)
    fvg_signals = detect_fvg(df)

    entries = []

    for i, row in df.iterrows():
        trade_time = pd.to_datetime(row['date'])

        # Optional news filter
        if config["flags"].get("enable_news_filter", False):
            if is_high_impact_news_nearby(trade_time):
                continue

        # Optional PDH/PDL sweep avoidance
        if config["flags"].get("avoid_pdh_pdl_liquidity", False):
            if is_near_pdh_or_pdl(df, i):
                continue

        # Wick-body ratio
        wick_ratio = wick_to_body_ratio(row)
        wick_body_ok = wick_ratio >= config['filters']['wick_body_ratio_threshold']

        # Volume spike
        volume_ok = detect_volume_spike(df, i, multiplier=config['filters']['volume_spike_multiplier'])

        bos_ok = bos_signals.loc[i] if i in bos_signals.index else False
        fvg_count = int(fvg_signals.loc[i]['FVG Count']) if i in fvg_signals.index and 'FVG Count' in fvg_signals.columns else 0



        # HTF trend filter (assume HTF already precomputed in df for now)
        htf_ok = True
        if config["flags"].get("enable_htf_trend", False):
            htf_ok = row.get("htf_trend", "up") == "up" if row.get("Signal") == "BUY" else "down"

        if wick_body_ok and volume_ok and htf_ok and (bos_ok or fvg_count > 0):
            entries.append({
                "Date": row['date'],
                "Signal": "BUY" if row['close'] > row['open'] else "SELL",
                "EntryPrice": row['close'],
                "Confidence": "A+ (SMC BOS + FVG)" if bos_ok and fvg_count else "B (SMC only)",
                "BOS Count": int(bos_ok),
                "FVG Count": fvg_count
            })

    return pd.DataFrame(entries)
