import pandas as pd
import numpy as np
import xgboost as xgb
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.trend import SMAIndicator, MACD
from ta.volume import VolumeWeightedAveragePrice
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)

def check_3_bar_pattern(candles_df):
    if len(candles_df) < 3:
        return False
    last_three = candles_df.iloc[-3:]
    bar_1_bearish = last_three.iloc[0]['Close'] < last_three.iloc[0]['Open']
    bar_2_bullish = last_three.iloc[1]['Close'] > last_three.iloc[1]['Open']
    bar_3_bullish = last_three.iloc[2]['Close'] > last_three.iloc[2]['Open']
    return bar_1_bearish and bar_2_bullish and bar_3_bullish

def check_pop_and_fade(candles_df):
    if len(candles_df) < 5:
        return False
    last_five = candles_df.iloc[-5:]
    sma20 = SMAIndicator(last_five['Close'], window=20).sma_indicator()
    sma200 = SMAIndicator(last_five['Close'], window=200).sma_indicator()
    trend = 'down' if sma20.iloc[-1] < sma200.iloc[-1] else 'up'
    spike = last_five.iloc[-2]['Close'] > last_five.iloc[-3]['Close'] * 1.02
    reversal = last_five.iloc[-1]['Close'] < last_five.iloc[-2]['Close']
    return spike and reversal and trend == 'down'

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
        candles_df['sma50'] = SMAIndicator(candles_df['Close'], window=50).sma_indicator()  # For trend direction
        candles_df['sma200'] = SMAIndicator(candles_df['Close'], window=200).sma_indicator()
        candles_df['macd'] = MACD(candles_df['Close']).macd()
        candles_df['vwap'] = VolumeWeightedAveragePrice(candles_df['High'], candles_df['Low'], candles_df['Close'], candles_df['Volume']).volume_weighted_average_price()
        candles_df['bb_upper'] = BollingerBands(candles_df['Close'], window=20).bollinger_hband()
        candles_df['bb_lower'] = BollingerBands(candles_df['Close'], window=20).bollinger_lband()
        candles_df['stoch_k'] = StochasticOscillator(candles_df['High'], candles_df['Low'], candles_df['Close'], window=14).stoch()
        
        # Latest values
        latest = candles_df.iloc[-1]
        features['rsi'] = latest['rsi']
        features['atr'] = latest['atr']
        features['price_above_sma20'] = 1 if latest['Close'] > latest['sma20'] else 0
        features['price_above_sma200'] = 1 if latest['Close'] > latest['sma200'] else 0
        features['price_below_sma20'] = 1 if latest['Close'] < latest['sma20'] else 0
        features['price_below_sma200'] = 1 if latest['Close'] < latest['sma200'] else 0
        features['volume_spike'] = 1 if latest['Volume'] > candles_df['Volume'].shift(1).mean() * 1.05 else 0
        features['atr_ratio'] = latest['atr'] / candles_df['atr'].mean() if candles_df['atr'].mean() != 0 else 1.0
        features['macd'] = latest['macd']
        features['price_above_vwap'] = 1 if latest['Close'] > latest['vwap'] else 0
        features['price_above_bb_upper'] = 1 if latest['Close'] > latest['bb_upper'] else 0
        features['price_below_bb_lower'] = 1 if latest['Close'] < latest['bb_lower'] else 0
        features['stoch_k'] = latest['stoch_k']
        # Check for trend direction (bearish if SMA50 is declining)
        sma50_shifted = candles_df['sma50'].shift(1)
        if sma50_shifted.isna().all():
            features['is_bearish_trend'] = 0  # Default to 0 if all values are NaN
        else:
            # Compare latest SMA50 with the previous SMA50 value
            features['is_bearish_trend'] = 1 if latest['sma50'] < sma50_shifted.iloc[-1] else 0
        
        return features
    except Exception as e:
        logger.error("Error extracting features: %s", e)
        return {}

def score_setup(candles_df, direction=None):
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
            # Adjust score based on trend direction
            if direction == 'long' and features.get('is_bearish_trend', 0):
                score *= 0.8  # Penalize long signals in bearish trends by 20%
            score = np.clip(score, 0, 10)
            logger.debug("Model-based score: %s", score)
            return float(score)
        except Exception as e:
            logger.warning("Failed to load XGBoost model: %s. Falling back to rule-based scoring.", e)
            # Rule-based scoring as fallback
            features = extract_features(candles_df)
            score = 0.0
            # RSI conditions
            if features.get('rsi', 0) > 70:
                score += 2.0  # Overbought
            elif features.get('rsi', 0) < 30:
                score += 2.0  # Oversold
            # Volume spike
            if features.get('volume_spike', 0):
                score += 2.0  # Volume spike indicates momentum
            # Trend alignment
            if features.get('price_above_sma20', 0) and features.get('price_above_sma200', 0):
                score += 2.0  # Strong bullish trend
            elif features.get('price_below_sma20', 0) and features.get('price_below_sma200', 0):
                score -= 1.0  # Strong bearish trend, penalize for mismatch with direction
            # Volatility: reward low volatility, penalize high volatility
            if features.get('atr_ratio', 0) < 1.0:
                score += 1.0  # Low volatility, safer entry
            elif features.get('atr_ratio', 0) > 2.0:
                score -= 1.0  # High volatility, riskier entry
            # MACD
            if features.get('macd', 0) > 0:
                score += 1.0  # Bullish MACD
            # VWAP
            if features.get('price_above_vwap', 0):
                score += 1.0  # Bullish VWAP
            # Bollinger Bands
            if features.get('price_above_bb_upper', 0):
                score -= 1.0  # Overextended, potential reversal
            elif features.get('price_below_bb_lower', 0):
                score += 1.0  # Potential bounce
            # Stochastic Oscillator
            if features.get('stoch_k', 0) > 80:
                score -= 1.0  # Overbought
            elif features.get('stoch_k', 0) < 20:
                score += 1.0  # Oversold
            # Trend direction adjustment
            if direction == 'long' and features.get('is_bearish_trend', 0):
                score *= 0.8  # Penalize long signals in bearish trends by 20%
            # Confluence from check_* functions
            if check_3_bar_pattern(candles_df):
                score += 1.0  # 3-bar pattern adds confluence
            if check_pop_and_fade(candles_df):
                score += 1.0  # Pop and fade pattern adds confluence
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

        # Log the distribution of winning and losing trades
        num_wins = len(trade_df[trade_df['pnl'] > 0])
        num_losses = len(trade_df[trade_df['pnl'] <= 0])
        logger.info("Dataset distribution: %s winning trades, %s losing trades", num_wins, num_losses)

        # Balance the dataset
        trade_df_wins = trade_df[trade_df['pnl'] > 0]
        trade_df_losses = trade_df[trade_df['pnl'] <= 0]
        # Sample the minimum number available to balance the dataset
        sample_size = min(len(trade_df_wins) // 2, len(trade_df_losses))
        if sample_size == 0:
            logger.warning("Not enough data to balance dataset. Proceeding with original dataset.")
        else:
            trade_df_wins = trade_df_wins.sample(n=sample_size, random_state=42)
            trade_df_losses = trade_df_losses.sample(n=sample_size, random_state=42)
            trade_df = pd.concat([trade_df_wins, trade_df_losses])
            logger.info("Balanced dataset: %s winning trades, %s losing trades", len(trade_df_wins), len(trade_df_losses))

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