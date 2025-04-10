import asyncio
from ibkr_client import IBKRTrader
from ai_scheduler import main as run_scheduler
from ib_insync import Forex, Future, Crypto

def main():
    # Initialize IBKR trader
    trader = IBKRTrader()

    # Run the scheduler
    try:
        asyncio.run(run_scheduler())
    except KeyboardInterrupt:
        trader.disconnect()

if __name__ == "__main__":
    main()