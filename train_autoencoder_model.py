import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load data
data = pd.read_csv('data/TSLA_3M_5min_mock.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'], utc=True)
data.set_index('timestamp', inplace=True)

# Prepare features (normalize close prices)
X = (data['close'] - data['close'].mean()) / data['close'].std()
X = X.values.reshape(-1, 1)

# Build Autoencoder
input_layer = Input(shape=(1,))
encoded = Dense(32, activation='relu')(input_layer)
decoded = Dense(1, activation='linear')(encoded)
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the model
logger.info("Training Autoencoder model...")
autoencoder.fit(X, X, epochs=5, batch_size=32, validation_split=0.2, verbose=1)

# Save the model
autoencoder.save('autoencoder_anomaly_detector.h5')
logger.info("Autoencoder model saved to autoencoder_anomaly_detector.h5")