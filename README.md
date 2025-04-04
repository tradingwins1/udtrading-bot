# AI Trading Bot (Modular Strategy)

This is a fully automated trading bot designed for Futures, Forex, and Crypto using a modular approach that includes:

- Smart Money Concepts (SMC)
- Pop and Fade entry logic
- Break of Structure (BOS), Fair Value Gap (FVG)
- Zone-based entry with volume and wick filters
- Margin check logic for IBKR Pro

## 📦 Features

- Modular logic for precision entries
- Real-time market data integration
- Discord alerts and journaling
- Multi-asset support (MNQ, MCL, USDJPY, ETHUSD, etc.)
- Paper trading & backtest-ready

## 📁 Project Structure

- `smc_strategy.py` – Main strategy logic
- `entry_precision.py` – Pop and Fade + wick/body filters
- `live_tracker.py` – Real-time SL/TP check
- `execution.py` – Trade execution handler
- `risk_manager.py` – Risk and position size logic
- `config.json` – Bot configuration
- `discord_alert.py` – Sends alerts to Discord

## 🧪 Setup

1. Clone the repo
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create and edit `config.json` for your API keys (IBKR / Binance)
4. Run the bot:
   ```bash
   python main.py
   ```
