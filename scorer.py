import pandas as pd
import numpy as np
import xgboost as xgb
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from ta.trend import SMAIndicator
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)

def check_3_bar_pattern(candles_df):
    logger.debug("Placeholder: check_3_bar_pattern")
    return False

def check_pop_and_fade(candles_df):
    logger.debug("Placeholder: check_pop_and_fade")
    return False

def check_vwap_bounce(candles_df):
    logger.debug("Placeholder: check_vwap_bounce")
    return False

def check_moving_average(candles_df):
    logger.debug("Placeholder: check_moving_average")
    return False

def check_dip_and_rip(candles_df):
    logger.debug("Placeholder: check_dip_and_rip")
    return False

def check_mean_reversal(candles_df):
    logger.debug("Placeholder: check_mean_reversal")
    return False

def extract_features(candles_df):
    """
    Extract features for scoring setups.
    """
    logger.debug("Extracting features from DataFrame with columns: %s", list(candles_df.columns))
    try:
        features = {}
        candles_df = candles_df.copy()
        
        # Calculate technical indicators
        candles_df['rsi'] = RSIIndicator(candles_df['Close'], window=14).rsi()
        candles_df['atr'] = AverageTrueRange(candles_df['High'], candles_df['Low'], candles_df['Close'], window=14).average_true_range()
        candles_df['sma20'] = SMAIndicator(candles_df['Close'], window=20).sma_indicator()
        candles_df['sma200'] = SMAIndicator(candles_df['Close'], window=200).sma_indicator()
        
        # Latest values
        latest = candles_df.iloc[-1]
        features['rsi'] = latest['rsi']
        features['atr'] = latest['atr']
        features['price_above_sma20'] = 1 if latest['Close'] > latest['sma20'] else 0
        features['price_above_sma200'] = 1 if latest['Close'] > latest['sma200'] else 0
        features['volume_spike'] = 1 if latest['Volume'] > candles_df['Volume'].shift(1).mean() * 1.05 else 0
        features['atr_ratio'] = latest['atr'] / candles_df['atr'].mean() if candles_df['atr'].mean() != 0 else 1.0
        
        return features
    except Exception as e:
        logger.error("Error extracting features: %s", e)
        return {}

def score_setup(candles_df):
    """
    Score a trading setup using a combination of ML model and rule-based checks.
    Returns a score between 0 and 10.
    """
    logger.debug("Scoring setup for DataFrame with shape: %s", candles_df.shape)
    try:
        model_path = os.path.join('models', 'setup_scorer.pkl')
        model = xgb.XGBRegressor()
        
        try:
            model.load_model(model_path)
            features = extract_features(candles_df)
            if not features:
                logger.warning("No features extracted, returning default score")
                return 0.0
            feature_array = np.array([list(features.values())])
            score = model.predict(feature_array)[0]
            score = np.clip(score, 0, 10)
            logger.debug("Model-based score: %s", score)
            return float(score)
        except Exception as e:
            logger.warning("Failed to load XGBoost model: %s. Falling back to rule-based scoring.", e)
            # Rule-based scoring as fallback
            features = extract_features(candles_df)
            score = 0.0
            if features.get('rsi', 0) > 70:
                score += 2.0  # Overbought
            elif features.get('rsi', 0) < 30:
                score += 2.0  # Oversold
            if features.get('volume_spike', 0):
                score += 2.0  # Volume spike
            if features.get('price_above_sma20', 0) and features.get('price_above_sma200', 0):
                score += 2.0  # Strong trend alignment
            if features.get('atr_ratio', 0) < 1.0:
                score += 1.0  # Low volatility, safer entry
            logger.debug("Rule-based score: %s", score)
            return float(np.clip(score, 0, 10))
    except Exception as e:
        logger.error("Error in scoring setup: %s", e)
        return 0.0

def train_scorer_model(data=None, labels=None):
    """
    Train the XGBoost model using historical trade data.
    """
    logger.debug("Training XGBoost model for scoring setups")
    try:
        # Load historical trade data from trade_logs.db
        from learn import load_trades
        trade_df = load_trades()
        if trade_df.empty:
            logger.error("No trade data available for training")
            return

        # Extract features for each trade
        features_list = []
        scores = []
        for idx, trade in trade_df.iterrows():
            # Mock candles_df based on trade data (since we don't have full OHLCV data)
            candles_data = {
                'Close': [trade['entry_price']] * 50 + [trade['exit_price']],
                'High': [trade['entry_price'] * 1.01] * 50 + [trade['exit_price'] * 1.01],
                'Low': [trade['entry_price'] * 0.99] * 50 + [trade['exit_price'] * 0.99],
                'Volume': [1000] * 51  # Mock volume
            }
            candles_df = pd.DataFrame(candles_data)
            features = extract_features(candles_df)
            if features:
                features_list.append(list(features.values()))
                # Score based on PNL: 10 for large wins, 0 for large losses
                score = np.clip((trade['pnl'] + 200) / 40, 0, 10)  # Map PNL [-200, 200] to [0, 10]
                scores.append(score)

        if not features_list or len(features_list) < 50:
            logger.error("Insufficient data for training: %s samples", len(features_list))
            return

        # Train XGBoost model
        X = np.array(features_list)
        y = np.array(scores)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        logger.info("Model trained, MSE on test set: %s", mse)

        # Save model
        os.makedirs('models', exist_ok=True)
        model.save_model(os.path.join('models', 'setup_scorer.pkl'))
        logger.info("Model trained and saved to models/setup_scorer.pkl")
    except Exception as e:
        logger.error("Error training model: %s", e)