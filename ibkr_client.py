# ibkr_client.py
from ib_async import IB, Stock, Forex, Future, MarketOrder, LimitOrder
import asyncio
import pandas as pd
import logging
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define ib at the module level for compatibility with execution.py
ib = IB()

async def connect_ibkr(host=None, port=None, client_id=None, mode="TWS"):
    """
    Connect to IBKR (TWS or IB Gateway) with retry logic.

    Args:
        host (str): Host address (default from .env or localhost).
        port (int): Port number (default from .env or based on mode).
        client_id (int): Client ID (default from .env or 1).
        mode (str): "TWS" for Trader Workstation, "Gateway" for IB Gateway.
    """
    # Default values
    host = host or os.getenv("IBKR_HOST", "127.0.0.1")
    if port is None:
        if mode == "TWS":
            port = int(os.getenv("IBKR_PORT", 7497))  # Default TWS paper trading port
        else:  # Gateway
            port = int(os.getenv("IBKR_PORT", 4002))  # Default Gateway paper trading port
    client_id = client_id or int(os.getenv("IBKR_CLIENT_ID", 1))

    if not ib.isConnected():
        for attempt in range(3):
            try:
                await ib.connectAsync(host, port, clientId=client_id)
                logger.info(f"Connected to IBKR ({mode}) at {host}:{port} with client ID {client_id}")
                return ib
            except Exception as e:
                logger.error(f"Connection attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(5)
        raise Exception(f"Failed to connect to IBKR ({mode}) after retries")
    return ib

class IBKRTrader:
    """Handles IBKR connections and trading operations asynchronously."""
    def __init__(self, mode="TWS"):
        self.ib = ib  # Use the module-level ib instance
        self.mode = mode  # "TWS" or "Gateway"

    async def connect(self, host=None, port=None, client_id=None):
        """Connect to IBKR (TWS or Gateway) with retry logic."""
        await connect_ibkr(host=host, port=port, client_id=client_id, mode=self.mode)

    async def disconnect(self):
        """Disconnect from IBKR."""
        if self.ib.isConnected():
            self.ib.disconnect()
            logger.info(f"Disconnected from IBKR ({self.mode})")

    async def resolve_contract(self, symbol, asset_type):
        """Resolve contract details for a given symbol and asset type."""
        if asset_type == "futures":
            future = Future(symbol=symbol, exchange='GLOBEX', currency='USD')
            contracts = await self.ib.reqContractDetailsAsync(future)
            if contracts:
                sorted_contracts = sorted(contracts, key=lambda c: c.contract.lastTradeDateOrContractMonth)
                return sorted_contracts[0].contract
            today = datetime.now()
            expiry = (today.replace(day=1) + timedelta(days=32)).strftime('%Y%m')
            return Future(symbol=symbol, lastTradeDateOrContractMonth=expiry, exchange='GLOBEX', currency='USD')
        elif asset_type == "forex":
            return Forex(symbol)
        elif asset_type == "stock":
            contract = Stock(symbol, exchange='SMART', currency='USD')
            await self.ib.qualifyContractsAsync(contract)
            return contract
        logger.warning(f"Unsupported asset type: {asset_type}")
        return None

    async def fetch_historical_data(self, symbol, asset_type, duration="1 M", bar_size="5 mins", end_date=None):
        """Fetch historical data for a given symbol and save it to a CSV file."""
        contract = await self.resolve_contract(symbol, asset_type)
        if contract is None:
            logger.error(f"Could not resolve contract for {symbol}")
            return None

        if end_date is None:
            end_date = datetime.now()

        try:
            bars = await self.ib.reqHistoricalDataAsync(
                contract=contract,
                endDateTime=end_date,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )
            if not bars:
                logger.error(f"No historical data returned for {symbol}")
                return None

            # Convert bars to DataFrame
            df = pd.DataFrame(
                [(bar.date, bar.open, bar.high, bar.low, bar.close, bar.volume) for bar in bars],
                columns=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
            )
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            df.set_index('Datetime', inplace=True)
            df.index = df.index.tz_localize('US/Eastern')

            # Save to CSV
            output_dir = "data"
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{symbol}_{duration.replace(' ', '')}_{bar_size.replace(' ', '')}.csv")
            df.to_csv(output_file)
            logger.info(f"Historical data for {symbol} saved to {output_file}")
            return df

        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: %s", e)
            return None

    async def place_order(self, symbol, asset_type, direction, quantity, sl=None, tp=None):
        """Place a market order with optional SL/TP and return trade details."""
        contract = await self.resolve_contract(symbol, asset_type)
        if contract is None:
            logger.error(f"Could not resolve contract for {symbol}")
            return None

        order = MarketOrder(direction.upper(), quantity)
        # Stub SL/TP (requires bracket order for actual implementation)
        if sl or tp:
            logger.warning(f"SL/TP not fully implemented: SL={sl}, TP={tp}")

        try:
            trade = self.ib.placeOrder(contract, order)
            logger.info(f"Executed {direction.upper()} order for {symbol} (Order ID: {trade.order.orderId})")
            for _ in range(5):
                await asyncio.sleep(2)
                status = trade.orderStatus.status
                logger.info(f"Trade status for {symbol}: {status}")
                if status in ['Filled', 'Cancelled', 'Inactive']:
                    break
            # Mock SL/TP attributes for compatibility
            trade.order.stopLoss = sl if sl else None
            trade.order.takeProfit = tp if tp else None
            return trade
        except Exception as e:
            logger.error(f"Error placing order for {symbol}: %s", e)
            return None