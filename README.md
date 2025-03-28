
# 🧠 AI Trading Bot (SMC-Based Scalper & Swing System)

This AI bot is designed for high-quality scalping and swing trading across Stocks, Futures, and Cryptos. It incorporates Smart Money Concepts (SMC), Break of Structure (BOS), Fair Value Gaps (FVG), and more — aiming for a 60%+ win rate using adaptive logic and real-time data.

---

## ✅ Features

- Break & Retest + BOS + FVG + Liquidity Sweep logic
- Red folder news avoidance
- HTF confluence and wick/body ratio filters
- Volume confirmation and timing constraints (NY trading hours only for stocks/futures)
- Live SL/TP tracking with Discord alerts
- Auto trade logging, risk management, performance summary
- Self-assessing logic via `bot_trainer.py`

---

## 🔧 Folder Structure

```
TradingBot/
├── bos_detector.py
├── bot_trainer.py
├── connect_ibkr.py
├── data_feed.py
├── discord_alert.py
├── execution.py
├── fvg_detector.py
├── live_dashboard.py
├── live_tracker.py
├── performance_summary.py
├── risk_manager.py
├── smc_strategy.py
├── strategy.py
├── test_discord_alert.py
├── trade_log.csv
├── trade_logger.py
├── trailing_stop.py
```

---

## 🛠 How to Use

### 1. Install Dependencies
```bash
pip install ib_insync pandas numpy requests matplotlib
```

### 2. Run the Bot
```bash
python data_feed.py
```

### 3. Live Monitoring (Optional)
```bash
python live_tracker.py
```

### 4. View Daily PnL
```bash
python performance_summary.py
```

### 5. Train and Refine Strategy
```bash
python bot_trainer.py
```

---

## ⏰ Schedule Daily Execution (CRON)
Example: Run `data_feed.py` every day at 9:30 AM EST
```bash
crontab -l | { cat; echo "30 9 * * * /usr/bin/python3 /path/to/data_feed.py"; } | crontab -
```

---

## 📈 Discord Alerts
Make sure to add your `SCALPING_ALERT_WEBHOOK` and `SWING_ALERT_WEBHOOK` in `discord_alert.py`.

---

## 📁 Logging
All trades are recorded in:
```
trade_log.csv
```

---

## 👨‍🏫 Strategy Goals
- Focus on A+ trades only (SMC + BOS + Volume)
- 60% minimum win rate
- 1:2 Risk:Reward per trade
- Strict time filtering and HTF alignment

---

## 📩 Questions?
Ping the assistant for live strategy review, debugging, or logic upgrades.
