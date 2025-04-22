import strategy_wrapper
print(strategy_wrapper.__file__)
from strategy_wrapper import UGBacktestStrategy
strategy = UGBacktestStrategy()
print(hasattr(strategy, 'load_data'))