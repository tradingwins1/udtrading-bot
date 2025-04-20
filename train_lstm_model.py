import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load data
data = pd.read_csv('data/TSLA_3M_5min_mock.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'], utc=True)
data.set_index('timestamp', inplace=True)

# Prepare features (close prices) and target (1 if next close is higher, 0 otherwise)
data['target'] = (data['close'].shift(-1) > data['close']).astype(int)
data = data.dropna()

# Create sequences for LSTM
sequence_length = 10
X, y = [], []
for i in range(len(data) - sequence_length):
    X.append(data['close'].iloc[i:i+sequence_length].values)
    y.append(data['target'].iloc[i+sequence_length])
X = np.array(X)
y = np.array(y)

# Reshape for LSTM [samples, time steps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split into train and test
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(sequence_length, 1)))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
logger.info("Training LSTM model...")
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Save the model
model.save('lstm_pattern_detector.h5')
logger.info("LSTM model saved to lstm_pattern_detector.h5")