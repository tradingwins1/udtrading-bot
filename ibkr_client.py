
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
