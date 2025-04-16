# DataFetcher (Enhanced)
# --------------------------------------------------
# - Pulls 5-min TSLA data via Alpha Vantage API
# - Cleans & caches data for backtesting
# - Placeholder added for future sentiment analysis

import pandas as pd
import requests
from dotenv import load_dotenv
import os
import logging

logger = logging.getLogger(__name__)

class DataFetcher:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not self.api_key:
            logger.error("Alpha Vantage API key not found in .env file")
            raise ValueError("Alpha Vantage API key not found in .env file")

    def fetch_5min_tsla(self, cache_file='tsla_5min.csv'):
        """
        Fetch 5-minute TSLA data from Alpha Vantage and cache it.
        """
        logger.debug("Fetching 5-minute TSLA data from Alpha Vantage...")
        try:
            # Initialize an empty DataFrame to store all data
            all_data = pd.DataFrame()
            # Alpha Vantage provides 1 month of intraday data per call, fetch multiple slices
            for year in range(1, 3):  # Adjust range based on data needs
                for month in range(1, 13):
                    slice_param = f"year{year}month{month}"
                    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol=TSLA&interval=5min&slice={slice_param}&apikey={self.api_key}"
                    response = requests.get(url)
                    response.raise_for_status()  # Raise an exception for bad status codes
                    data = pd.read_csv(response.text.splitlines(), delimiter=',')
                    all_data = pd.concat([all_data, data], ignore_index=True)
                    logger.debug("Fetched data for slice %s: %s rows", slice_param, len(data))
            
            # Clean and format the data
            all_data = self.clean_data(all_data)
            
            # Cache the data
            cache_path = os.path.join('data', cache_file)
            all_data.to_csv(cache_path)
            logger.info("Data cached to %s", cache_path)
            
            return all_data
        except Exception as e:
            logger.error("Error fetching data from Alpha Vantage: %s", e)
            raise

    def clean_data(self, df):
        """
        Clean and format the fetched data.
        """
        logger.debug("Cleaning fetched data...")
        try:
            # Rename columns to match expected format
            df = df.rename(columns={
                'time': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            # Convert Date to datetime and set as index
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            # Sort by date
            df = df.sort_index()
            # Ensure numeric types
            df = df.astype({'Open': float, 'High': float, 'Low': float, 'Close': float, 'Volume': float})
            logger.debug("Data cleaned successfully: %s rows", len(df))
            return df
        except Exception as e:
            logger.error("Error cleaning data: %s", e)
            raise

    # Placeholder for future sentiment data integration (as recommended in UG_Trading_Bot_Analysis_Report.pdf)
    def fetch_sentiment_data(self):
        """
        Placeholder for fetching sentiment data (e.g., FinBERT for news/X sentiment).
        """
        logger.debug("Sentiment data fetching not implemented yet.")
        pass