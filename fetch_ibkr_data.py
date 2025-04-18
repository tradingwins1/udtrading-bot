# fetch_ibkr_data.py
import asyncio
from ibkr_client import IBKRTrader
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def fetch_data():
    trader = IBKRTrader(mode="TWS")  # Use TWS for local testing
    try:
        # Connect to IBKR
        await trader.connect()

        # Fetch data for TSLA (1-minute and 5-minute)
        for bar_size in ["1 min", "5 mins"]:
            tsla_data = await trader.fetch_historical_data(
                symbol="TSLA",
                asset_type="stock",
                duration="3 M",  # Fetch 3 months of data
                bar_size=bar_size
            )
            if tsla_data is None:
                logger.error(f"Failed to fetch {bar_size} data for TSLA")

        # Fetch data for QQQ (1-minute and 5-minute)
        for bar_size in ["1 min", "5 mins"]:
            qqq_data = await trader.fetch_historical_data(
                symbol="QQQ",
                asset_type="stock",
                duration="3 M",
                bar_size=bar_size
            )
            if qqq_data is None:
                logger.error(f"Failed to fetch {bar_size} data for QQQ")

        logger.info("Successfully fetched 1-minute and 5-minute data for TSLA and QQQ")

    except Exception as e:
        logger.error(f"Error in fetching data: %s", e)
    finally:
        await trader.disconnect()

if __name__ == "__main__":
    asyncio.run(fetch_data())