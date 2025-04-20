import pandas as pd
from prophet import Prophet
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load data
data = pd.read_csv('data/TSLA_3M_5min_mock.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'], utc=True)
# Remove timezone from the timestamp column
data['timestamp'] = data['timestamp'].dt.tz_localize(None)
df = data[['timestamp', 'close']].rename(columns={'timestamp': 'ds', 'close': 'y'})

# Train Prophet model
logger.info("Training Prophet model...")
model = Prophet()
model.fit(df)

# Save the model
with open('prophet_price_predictor.pkl', 'wb') as f:
    pickle.dump(model, f)
logger.info("Prophet model saved to prophet_price_predictor.pkl")