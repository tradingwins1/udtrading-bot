
"""
bot_trainer.py

This module analyzes past trade performance and updates the AI bot's strategy parameters
to improve win rate. It aims to maintain a win rate >= 60%.
"""

import pandas as pd

def assess_trade_performance(trade_log_path='trade_log.csv', min_trades=10):
    try:
        df = pd.read_csv(trade_log_path, parse_dates=['Date'])
        if len(df) < min_trades:
            print("Not enough trades to assess. Waiting for more data...")
            return None

        win_rate = (df['PnL'] > 0).mean() * 100
        avg_pnl = df['PnL'].mean()
        losing_patterns = df[df['PnL'] <= 0]['Comment'].value_counts().to_dict() if 'Comment' in df.columns else {}

        print(f"Win Rate: {win_rate:.2f}% | Avg PnL: ${avg_pnl:.2f}")
        print("Most common loss triggers:", losing_patterns)

        adjustments = {}

        if win_rate < 60:
            adjustments['tighten_entry'] = True
            adjustments['require_volume_confirmation'] = True
            if 'wick rejection' in ''.join(losing_patterns.keys()).lower():
                adjustments['apply_wick_filter'] = True
            if 'pdh' in ''.join(losing_patterns.keys()).lower() or 'pdl' in ''.join(losing_patterns.keys()).lower():
                adjustments['avoid_pdh_pdl_traps'] = True

        return adjustments

    except Exception as e:
        print("Error in assessing performance:", e)
        return None
