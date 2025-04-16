# bot_trainer.py (Enhanced)
# --------------------------------------------------
# - Logs win rate and feature importance
# - Suggests strategy changes if win rate < 60%
# - Dumps best model and prints evaluation summary

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from ta.volatility import BollingerBands
from ta.momentum import StochasticOscillator, RSIIndicator

# Mock trend_detector and utils for compatibility
def determine_trend(candles_df):
    sma20 = candles_df['close'].rolling(window=20).mean()
    sma200 = candles_df['close'].rolling(window=200).mean()
    if sma20.iloc[-1] > sma200.iloc[-1]:
        return 'up'
    elif sma20.iloc[-1] < sma200.iloc[-1]:
        return 'down'
    return 'neutral'

def check_3_bar_pattern(candles_df):
    if len(candles_df) < 3:
        return False
    last_three = candles_df.iloc[-3:]
    bar_1_bearish = last_three.iloc[0]['close'] < last_three.iloc[0]['open']
    bar_2_bullish = last_three.iloc[1]['close'] > last_three.iloc[1]['open']
    bar_3_bullish = last_three.iloc[2]['close'] > last_three.iloc[2]['open']
    return bar_1_bearish and bar_2_bullish and bar_3_bullish

def check_pop_and_fade(candles_df, trend):
    if len(candles_df) < 5:
        return False
    last_five = candles_df.iloc[-5:]
    spike = last_five.iloc[-2]['close'] > last_five.iloc[-3]['close'] * 1.02
    reversal = last_five.iloc[-1]['close'] < last_five.iloc[-2]['close']
    return spike and reversal and trend == 'down'

def calculate_atr(candles_df):
    if len(candles_df) < 14:
        return 0.0
    atr = (candles_df['high'].iloc[-14:] - candles_df['low'].iloc[-14:]).mean()
    return atr

# Import scorer functions
from scorer import check_dip_and_rip, check_mean_reversal

MODEL_PATH = "models/trade_predictor.pkl"

def extract_features(trade_data, candles_df):
    """
    Extract features for trade prediction.
    """
    trend = determine_trend(candles_df)
    pattern = check_3_bar_pattern(candles_df)
    pop_fade = check_pop_and_fade(candles_df, trend)
    dip_and_rip = check_dip_and_rip(candles_df)
    mean_reversal = check_mean_reversal(candles_df)
    atr = calculate_atr(candles_df)
    avg_atr = calculate_atr(candles_df[-50:]) if len(candles_df) >= 50 else atr
    volume_spike = candles_df['volume'].iloc[-1] > 1.5 * candles_df['volume'].iloc[-10:-1].mean()
    
    # Ensure column names are lowercase
    candles_df = candles_df.rename(columns=lambda x: x.lower())
    
    # Calculate indicators
    candles_df['rsi'] = RSIIndicator(candles_df['close'], window=14).rsi()
    candles_df['stoch'] = StochasticOscillator(candles_df['high'], candles_df['low'], candles_df['close'], window=14).stoch()
    bb = BollingerBands(candles_df['close'], window=20)
    candles_df['bb_high'] = bb.bollinger_hband()
    candles_df['bb_low'] = bb.bollinger_lband()
    
    curr = candles_df.iloc[-1]
    
    features = {
        'confidence_score': trade_data.get('confidence_score', 5.0),
        'setup_type_swing': 1 if trade_data.get('setup_type') == 'swing' else 0,
        'setup_type_scalp': 1 if trade_data.get('setup_type') == 'scalp' else 0,
        'has_3_bar_pattern': 1 if pattern else 0,
        'has_pop_fade': 1 if pop_fade else 0,
        'has_dip_and_rip': 1 if dip_and_rip else 0,
        'has_mean_reversal': 1 if mean_reversal else 0,
        'volume_spike': 1 if volume_spike else 0,
        'atr_ratio': atr / avg_atr if avg_atr != 0 else 1.0,
        'trend_up': 1 if trend == 'up' else 0,
        'trend_down': 1 if trend == 'down' else 0,
        'rr_ratio': trade_data.get('rr_ratio', 2.0),
        'rsi': curr['rsi'],
        'stoch': curr['stoch'],
        'bb_distance': (curr['close'] - curr['bb_low']) / (curr['bb_high'] - curr['bb_low']) if (curr['bb_high'] - curr['bb_low']) != 0 else 0.5
    }
    return features

def evaluate_trade(trade_data, candles_df):
    """
    Evaluate trade and log features.
    """
    features = extract_features(trade_data, candles_df)
    win = trade_data['result'] == 'win'
    features['win'] = 1 if win else 0
    
    feature_df = pd.DataFrame([features])
    # Ensure the training directory exists
    os.makedirs("training", exist_ok=True)
    feature_df.to_csv("training/trade_log.csv", mode='a', header=not os.path.exists("training/trade_log.csv"), index=False)
    
    return {
        'setup_quality': sum([features['has_3_bar_pattern'], features['has_pop_fade'], 
                             features['has_dip_and_rip'], features['has_mean_reversal']]),
        'win_rate': features['win']
    }

def train_model(trade_log_path='training/trade_log.csv', min_trades=50):
    """
    Train XGBoost model with hyperparameter tuning.
    """
    try:
        df = pd.read_csv(trade_log_path)
        if len(df) < min_trades:
            print(f"Not enough trades. Need {min_trades}, have {len(df)}.")
            return None

        X = df.drop(columns=['win'])
        y = df['win']

        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }
        model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X, y)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Model accuracy: {accuracy:.2f}")

        joblib.dump(best_model, MODEL_PATH)
        return best_model
    except Exception as e:
        print(f"Error training model: {e}")
        return None

def assess_trade_performance(trade_log_path='training/trade_log.csv', min_trades=10):
    """
    Assess performance and suggest adjustments.
    """
    try:
        df = pd.read_csv(trade_log_path)
        if len(df) < min_trades:
            print("Not enough trades to assess.")
            return None

        win_rate = df['win'].mean() * 100
        print(f"Win Rate: {win_rate:.2f}%")

        model = train_model(trade_log_path)
        if model is None and os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)

        adjustments = {}
        if win_rate < 60:
            adjustments['tighten_entry'] = True
            adjustments['require_volume_confirmation'] = True

            if model:
                feature_importances = pd.Series(model.feature_importances_, index=df.drop(columns=['win']).columns)
                print("Feature Importances:", feature_importances.sort_values(ascending=False))

                if feature_importances.get('has_3_bar_pattern', 0) < 0.1:
                    adjustments['reduce_3_bar_weight'] = True
                if feature_importances.get('has_pop_fade', 0) < 0.1:
                    adjustments['reduce_pop_fade_weight'] = True
                if feature_importances.get('has_dip_and_rip', 0) < 0.1:
                    adjustments['reduce_dip_and_rip_weight'] = True
                if feature_importances.get('has_mean_reversal', 0) < 0.1:
                    adjustments['reduce_mean_reversal_weight'] = True
                if feature_importances.get('volume_spike', 0) > 0.2:
                    adjustments['increase_volume_requirement'] = True
                if feature_importances.get('atr_ratio', 0) < 0.1:
                    adjustments['avoid_low_atr'] = True

                print("\n📊 Strategy Adjustment Suggestions:")
                for key, value in adjustments.items():
                    print(f"  - {key}: {value}")

        return adjustments
    except Exception as e:
        print(f"Error assessing performance: %s", e)
        return None

def update_strategy(adjustments):
    """
    Update strategy based on adjustments.
    """
    if not adjustments:
        return

    from scorer import update_weights
    update_weights(adjustments)