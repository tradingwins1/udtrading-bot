<<<<<<< HEAD
from ib_async import IB, Stock, Forex, Future, MarketOrder
import asyncio
import random
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IBKRTrader:
    """Handles IBKR Gateway connections and trading operations asynchronously."""

    def __init__(self):
        self.ib = IB()

    async def connect(self):
        """Connect to IBKR Gateway with retry logic."""
        if not self.ib.isConnected():
            client_id = random.randint(1000, 9999)
            for attempt in range(3):
                try:
                    await self.ib.connectAsync('127.0.0.1', 4004, clientId=client_id)
                    logger.info("Connected to IBKR Gateway")
                    return
                except Exception as e:
                    logger.error(f"Connection attempt {attempt + 1} failed: {e}")
                    await asyncio.sleep(5)
            raise Exception("Failed to connect to IBKR after retries")

    async def disconnect(self):
        """Disconnect from IBKR Gateway."""
        if self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnected from IBKR")

    async def resolve_contract(self, symbol, asset_type):
        """Resolve contract details for a given symbol and asset type."""
        if asset_type == "futures":
            # Fetch all contracts and select the nearest active one
            future = Future(symbol=symbol, exchange='GLOBEX', currency='USD')
            contracts = await self.ib.reqContractDetailsAsync(future)
            if contracts:
                # Sort by expiry and pick the nearest future contract
                sorted_contracts = sorted(contracts, key=lambda c: c.contract.lastTradeDateOrContractMonth)
                return sorted_contracts[0].contract
            # Fallback to next month if no contracts found
            today = datetime.now()
            expiry = (today.replace(day=1) + datetime.timedelta(days=32)).strftime('%Y%m')
            return Future(symbol=symbol, lastTradeDateOrContractMonth=expiry, exchange='GLOBEX', currency='USD')
        elif asset_type == "forex":
            return Forex(symbol)
        elif asset_type == "stock":
            contract = Stock(symbol, exchange='SMART', currency='USD')
            await self.ib.qualifyContractsAsync(contract)  # Ensure contract is valid
            return contract
        logger.warning(f"Unsupported asset type: {asset_type}")
        return None

    async def place_order(self, symbol, asset_type, direction, quantity):
        """Place a market order and return trade details."""
        contract = await self.resolve_contract(symbol, asset_type)
        if contract is None:
            logger.error(f"Could not resolve contract for {symbol}")
            return None

        order = MarketOrder(direction.upper(), quantity)
        try:
            trade = self.ib.placeOrder(contract, order)
            logger.info(f"Executed {direction.upper()} order for {symbol} (Order ID: {trade.order.orderId})")
            # Wait for order to process with a timeout
            for _ in range(5):  # Max 10 seconds
                await asyncio.sleep(2)
                status = trade.orderStatus.status
                logger.info(f"Trade status for {symbol}: {status}")
                if status in ['Filled', 'Cancelled', 'Inactive']:
                    break
            return trade  # Return trade object for further tracking
        except Exception as e:
            logger.error(f"Error placing order for {symbol}: {e}")
            return None
=======

from ib_insync import *
import random
ib = IB()

def connect_ibkr():
    if not ib.isConnected():
        client_id = random.randint(1000, 9999)  # Ensures it's unique every run
        ib.connect('127.0.0.1', 7497, clientId=client_id)
    print("✅ Connected to IBKR TWS")

def disconnect_ibkr():
    if ib.isConnected():
        ib.disconnect()
        print("🔌 Disconnected from IBKR")

def resolve_contract(symbol, asset_type):
    if asset_type == "futures":
        # Auto-determine expiry
        import datetime
        today = datetime.date.today()
        expiry = (today.replace(day=1) + datetime.timedelta(days=32)).strftime('%Y%m')  # Next month

        # Smart contract resolution fallback
        contracts = ib.reqContractDetails(Future(symbol=symbol, exchange='GLOBEX'))
        if contracts:
            return contracts[0].contract
        else:
            return Future(symbol=symbol, lastTradeDateOrContractMonth=expiry, exchange='GLOBEX', currency='USD')

    elif asset_type == "forex":
        return Forex(symbol)

    elif asset_type == "stock":
        return Stock(symbol, exchange='SMART', currency='USD')

    return None

def place_order(symbol, asset_type, direction, quantity):
    contract = resolve_contract(symbol, asset_type)
    if contract is None:
        print(f"❌ Could not resolve contract for {symbol}")
        return

    order = MarketOrder(direction.upper(), quantity)
    trade = ib.placeOrder(contract, order)
    print(f"[IBKR] Executed {direction.upper()} order for {symbol}")
>>>>>>> c12cab3eb014edf455e1f1b6569173c8b901b0f3
