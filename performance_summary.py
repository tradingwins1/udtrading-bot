import pandas as pd

def print_summary(log_path="trade_log.csv"):
    try:
        df = pd.read_csv(log_path)

        if df.empty:
            print("⚠️ Trade log is empty.")
            return

        total_trades = len(df)
        wins = df[df['PnL'] > 0]
        losses = df[df['PnL'] < 0]
        break_even = df[df['PnL'] == 0]

        win_rate = round(len(wins) / total_trades * 100, 2) if total_trades > 0 else 0
        total_pnl = round(df['PnL'].sum(), 2)
        avg_pnl = round(df['PnL'].mean(), 2)

        print("\n📈 Daily Trade Summary:")
        print(f"   Total Trades    : {total_trades}")
        print(f"   Winning Trades  : {len(wins)}")
        print(f"   Losing Trades   : {len(losses)}")
        print(f"   Break Even      : {len(break_even)}")
        print(f"   Win Rate (%)    : {win_rate}")
        print(f"   Total PnL ($)   : {total_pnl}")
        print(f"   Avg PnL/Trade   : {avg_pnl}")

    except FileNotFoundError:
        print("⚠️ No trade log found.")
