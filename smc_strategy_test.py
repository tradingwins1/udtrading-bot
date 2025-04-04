from ib_insync import *

# Create an IB connection
ib = IB()
ib.connect('127.0.0.1', 4002, clientId=101)

# Define a simple stock order
stock = Stock('TSLA', 'SMART', 'USD')
order = MarketOrder('BUY', 1)  # Buy 1 share

# Place the order
trade = ib.placeOrder(stock, order)

# Wait for the trade to complete
ib.sleep(2)
print(f"Order Status: {trade.orderStatus.status}")

# Disconnect
ib.disconnect()
