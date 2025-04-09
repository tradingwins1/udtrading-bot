import schedule
import asyncio
import time
from datetime import datetime
from bot.ibkr_client import IBKRTrader
from bot.red_news_filter import is_red_folder_event_today
from bot.live_tracker import add_live_trade, track_live_trades
from bot.price_feed import get_latest_price
from bot.strategy import run_single_asset
import pytz
import os
from dotenv import load_dotenv
from bot.db import init_db, log_trade
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

init_db()
load_dotenv()
trader = IBKRTrader()

trade_counter = {
    "scalp": 0,
    "swing": 0,
    "last_date": datetime.now().date()
}

def reset_trade_counter():
    today = datetime.now().date()
    if trade_counter["last_date"] != today:
        trade_counter["scalp"] = 0
        trade_counter["swing"] = 0
        trade_counter["last_date"] = today

def is_scalping_window():
    now = datetime.now(pytz.timezone("US/Central"))
    if now.hour == 8 and now.minute < 30:
        logger.info("Scalping automation only starts at 8:30 AM CST")
        return False
    elif now.hour == 10 and now.minute > 30:
        logger.info("Scalping automation ends after 10:30 AM CST")
        return False
    elif now.hour < 8 or now.hour > 10:
        logger.info("Scalping window is from 8:30 AM to 10:30 AM CST only")
        return False
    return True

def is_swing_window():
    now = datetime.now(pytz.timezone("US/Central"))
    if now.hour < 7:
        logger.info("Swing scanning only starts at 7:00 AM CST")
        return False
    elif now.hour > 11:
        logger.info("Swing scanning ends after 11:00 AM CST")
        return False
    return True

def map_symbol(symbol):
    symbol_map = {
        "USDJPY": "JPY=X",
        "EURUSD": "EURUSD=X",
        "XAUUSD": "XAUUSD=X"
    }
    return symbol_map.get(symbol.upper(), symbol)

async def track_open_trades():
    all_assets = ["TSLA", "AAPL", "NVDA", "AMD", "USDJPY", "EURUSD", "XAUUSD"]
    for asset in all_assets:
        mapped_symbol = map_symbol(asset)
        latest_price = get_latest_price(mapped_symbol)
        if latest_price:
            track_live_trades()  # Updated to call the function directly

scalping_assets = [
    ("TSLA", "stock"),
    ("AAPL", "stock"),
    ("NVDA", "stock"),
    ("AMD", "stock")
]

async def run_scalping():
    logger.info("Running scalping automation")
    reset_trade_counter()
    if not is_scalping_window():
        logger.info("Outside scalping hours")
        return
    if is_red_folder_event_today():
        logger.info("Red folder news detected, skipping scalping today")
        return
    for symbol, asset_type in scalping_assets:
        if trade_counter["scalp"] >= 2:
            logger.info("Max scalps reached for today")
            return
        trade = await trader.place_order(symbol, asset_type, "BUY", 1)
        if trade:
            add_live_trade(symbol, trade.orderStatus.avgFillPrice, trade.order.stopLoss, trade.order.takeProfit, "BUY")
            log_trade({
                "asset": symbol,
                "direction": "BUY",
                "entry_price": trade.orderStatus.avgFillPrice,
                "exit_price": None,
                "stop_loss": trade.order.stopLoss,
                "take_profit": trade.order.takeProfit,
                "rr_ratio": None,
                "result": "Pending",
                "setup_type": "scalp",
                "confidence_score": None,
                "notes": ""
            })
            trade_counter["scalp"] += 1
    await track_open_trades()

swing_assets = [
    ("USDJPY", "forex"),
    ("EURUSD", "forex"),
    ("XAUUSD", "forex")
]

async def run_swing():
    logger.info("Running swing setup")
    reset_trade_counter()
    if not is_swing_window():
        logger.info("Outside swing hours")
        return
    if is_red_folder_event_today():
        logger.info("Red folder news detected, skipping swing today")
        return
    for symbol, asset_type in swing_assets:
        if trade_counter["swing"] >= 2:
            logger.info("Swing trade already executed today")
            return
        trade = await trader.place_order(symbol, asset_type, "BUY", 1)
        if trade:
            add_live_trade(symbol, trade.orderStatus.avgFillPrice, trade.order.stopLoss, trade.order.takeProfit, "BUY")
            log_trade({
                "asset": symbol,
                "direction": "BUY",
                "entry_price": trade.orderStatus.avgFillPrice,
                "exit_price": None,
                "stop_loss": trade.order.stopLoss,
                "take_profit": trade.order.takeProfit,
                "rr_ratio": None,
                "result": "Pending",
                "setup_type": "swing",
                "confidence_score": None,
                "notes": ""
            })
            trade_counter["swing"] += 1
    await track_open_trades()

async def main():
    await trader.connect()
    schedule.every(5).minutes.do(lambda: asyncio.create_task(run_scalping()))
    schedule.every(1).hours.do(lambda: asyncio.create_task(run_swing()))
    logger.info("AI Scheduler running...")
    while True:
        schedule.run_pending()
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())