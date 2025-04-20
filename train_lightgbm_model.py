import pandas as pd
import lightgbm as lgb
import ta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load data
data = pd.read_csv('data/TSLA_3M_5min_mock.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'], utc=True)
data.set_index('timestamp', inplace=True)

# Prepare features
data['RSI'] = ta.momentum.RSIIndicator(data['close'], window=14).rsi()
data['ATR'] = ta.volatility.AverageTrueRange(data['high'], data['low'], data['close'], window=14).average_true_range()
data['EMA50'] = ta.trend.EMAIndicator(data['close'], window=50).ema_indicator()
data['EMA200'] = ta.trend.EMAIndicator(data['close'], window=200).ema_indicator()
data['trend_bullish'] = (data['EMA50'] > data['EMA200']).astype(int)
data['volume_sma'] = data['volume'].rolling(window=20).mean()
data['high_volume'] = (data['volume'] > data['volume_sma']).astype(int)

# Create a target: 1 if the next close is higher, 0 otherwise
data['target'] = (data['close'].shift(-1) > data['close']).astype(int)

# Drop rows with NaN values
data = data.dropna()

# Features and target
features = ['RSI', 'ATR', 'trend_bullish', 'high_volume']
X = data[features]
y = data['target']

# Split into train and test (80% train, 20% test)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Create LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Define parameters
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

# Train the model
logger.info("Training LightGBM model...")
model = lgb.train(params, train_data, num_boost_round=100, valid_sets=[test_data])

# Save the model
model.save_model('lightgbm_trade_predictor.txt')
logger.info("LightGBM model saved to lightgbm_trade_predictor.txt")