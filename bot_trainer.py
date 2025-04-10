"""
bot_trainer.py

This module analyzes past trade performance and updates the AI bot's strategy parameters
to improve win rate. It aims to maintain a win rate >= 60% using advanced learning with scikit-learn.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from trend_detector import determine_trend
from utils import check_3_bar_pattern, check_pop_and_fade, calculate_atr

# Model file path for saving/loading the trained model
MODEL_PATH = "training/models/trade_predictor.pkl"

def extract_features(trade_data, candles_df):
    """
    Extract features from trade data and candle data for model training.
    trade_data: Dict with trade details (e.g., entry_price, stop_loss, take_profit, outcome).
    candles_df: DataFrame with candle data at the time of the trade.
    Returns: Dict with features.
    """
    trend = determine_trend(candles_df)
    pattern = check_3_bar_pattern(candles_df)
    pop_fade = check_pop_and_fade(candles_df, trend)
    atr = calculate_atr(candles_df)
    avg_atr = calculate_atr(candles_df[-50:])  # Average over last 50 candles
    volume_spike = candles_df['volume'].iloc[-1] > 1.5 * candles_df['volume'][-10:-1].mean()

    features = {
        "confidence_score": trade_data.get('confidence_score', 5.0),
        "setup_type_swing": 1 if trade_data.get('setup_type') == "swing" else 0,
        "setup_type_scalp": 1 if trade_data.get('setup_type') == "scalp" else 0,
        "has_3_bar_pattern": 1 if pattern else 0,
        "has_pop_fade": 1 if pop_fade else 0,
        "volume_spike": 1 if volume_spike else 0,
        "atr_ratio": atr / avg_atr if avg_atr != 0 else 1.0,
        "trend_up": 1 if trend == "up" else 0,
        "trend_down": 1 if trend == "down" else 0,
        "rr_ratio": trade_data.get('rr_ratio', 2.0),
    }
    return features

def evaluate_trade(trade_data, candles_df):
    """
    Evaluate a trade's outcome and log features for learning.
    trade_data: Dict with trade details (e.g., entry_price, stop_loss, take_profit, outcome).
    candles_df: DataFrame with candle data at the time of the trade.
    Returns: Dict with evaluation metrics.
    """
    features = extract_features(trade_data, candles_df)
    win = trade_data['result'] == "win"
    features['win'] = 1 if win else 0

    # Log features to CSV for training
    feature_df = pd.DataFrame([features])
    feature_df.to_csv("training/trade_log.csv", mode='a', header=not pd.io.common.file_exists("training/trade_log.csv"), index=False)

    return {
        "setup_quality": features['has_3_bar_pattern'] + features['has_pop_fade'],
        "win_rate": features['win']
    }

def train_model(trade_log_path='training/trade_log.csv', min_trades=50):
    """
    Train a Random Forest model to predict trade outcomes.
    Returns: Trained model or None if not enough data.
    """
    try:
        df = pd.read_csv(trade_log_path)
        if len(df) < min_trades:
            print(f"Not enough trades to train model. Need {min_trades}, have {len(df)}.")
            return None

        # Features and target
        X = df.drop(columns=['win'])
        y = df['win']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Random Forest
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy:.2f}")

        # Save model
        joblib.dump(model, MODEL_PATH)
        return model

    except Exception as e:
        print(f"Error in training model: {e}")
        return None

def assess_trade_performance(trade_log_path='training/trade_log.csv', min_trades=10):
    """
    Assess trade performance and suggest adjustments.
    Returns: Dict with adjustments.
    """
    try:
        df = pd.read_csv(trade_log_path)
        if len(df) < min_trades:
            print("Not enough trades to assess. Waiting for more data...")
            return None

        win_rate = df['win'].mean() * 100
        avg_confidence = df['confidence_score'].mean()
        print(f"Win Rate: {win_rate:.2f}% | Avg Confidence: {avg_confidence:.2f}")

        # Train or load model
        model = train_model(trade_log_path)
        if model is None and pd.io.common.file_exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)

        adjustments = {}

        if win_rate < 60:
            adjustments['tighten_entry'] = True
            adjustments['require_volume_confirmation'] = True

            # Use model to identify important features
            if model:
                feature_importances = pd.Series(model.feature_importances_, index=df.drop(columns=['win']).columns)
                print("Feature Importances:", feature_importances.sort_values(ascending=False))

                # Adjust scoring weights based on feature importance
                if 'has_3_bar_pattern' in feature_importances and feature_importances['has_3_bar_pattern'] < 0.1:
                    adjustments['reduce_3_bar_weight'] = True
                if 'has_pop_fade' in feature_importances and feature_importances['has_pop_fade'] < 0.1:
                    adjustments['reduce_pop_fade_weight'] = True
                if 'volume_spike' in feature_importances and feature_importances['volume_spike'] > 0.2:
                    adjustments['increase_volume_requirement'] = True
                if 'atr_ratio' in feature_importances and feature_importances['atr_ratio'] < 0.1:
                    adjustments['avoid_low_atr'] = True

        return adjustments

    except Exception as e:
        print("Error in assessing performance:", e)
        return None

def update_strategy(adjustments):
    """
    Update strategy parameters based on model insights.
    adjustments: Dict with suggested adjustments.
    """
    if not adjustments:
        return

    # Example: Adjust scoring weights in scorer.py
    if 'reduce_3_bar_weight' in adjustments:
        print("Reducing weight of 3-bar pattern in scoring...")
        # Modify scorer.py logic (e.g., reduce confluence weight)
    if 'reduce_pop_fade_weight' in adjustments:
        print("Reducing weight of Pop and Fade Out in scoring...")
    if 'increase_volume_requirement' in adjustments:
        print("Increasing volume spike requirement to 2x average...")
        # Update smc_strategy.py to require 2x volume
    if 'avoid_low_atr' in adjustments:
        print("Avoiding trades with low ATR...")
        # Update smc_strategy.py to filter low ATR trades