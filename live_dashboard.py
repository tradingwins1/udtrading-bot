import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_pnl_dashboard(csv_path='trade_log.csv', output_path='pnl_dashboard.png'):
    df = pd.read_csv(csv_path)
    if df.empty:
        print("⚠️ No trades to plot.")
        return

    df['PnL'] = df['PnL'].cumsum()
    df['Trade #'] = range(1, len(df) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(df['Trade #'], df['PnL'], marker='o', linestyle='-')
    plt.title('Cumulative PnL Over Trades')
    plt.xlabel('Trade Number')
    plt.ylabel('Cumulative PnL ($)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
