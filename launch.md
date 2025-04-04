# 🚀 Launch Instructions for AI Trading Bot

Follow these steps to run your AI bot from scratch:

### 1. Set up Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API and Bot Settings

Edit the `config.json`:
- IBKR / Binance keys
- Symbols
- Risk settings
- Discord webhook

### 3. Start Paper Trading Mode

```bash
python main.py
```

Bot will fetch live data, analyze trades, and simulate execution with alerting.

### 4. Monitor Live Trades

- Discord alert logs
- Terminal logs with price/action tracking
- Trade journal files (CSV or dashboard logs)
