# ai_utils.py (Enhanced)
# --------------------------------------------------
# - Adds try/except to avoid missing timestamps
# - Logs skipped trades if entry_time not found
# - Prepares cleaner AI dataset with indicators

import pandas as pd
import numpy as np
from ta.momentum import StochasticOscillator
from ta.trend import ADXIndicator
from ta.volatility import BollingerBands
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import logging

logger = logging.getLogger(__name__)

def prepare_ai_dataset(trades, data):
    """Prepare AI dataset with indicators for training."""
    df = data.copy()
    try:
        df['ADX'] = ADXIndicator(df['High'], df['Low'], df['Close'], window=14).adx()
        df['Stoch'] = StochasticOscillator(df['High'], df['Low'], df['Close'], window=14).stoch()
        bb = BollingerBands(df['Close'], window=20)
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Low'] = bb.bollinger_lband()
        df['Volatility'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(252 * 78)
    except Exception as e:
        logger.error("Error calculating indicators: %s", e)
        return pd.DataFrame()

    ai_data = []
    trades['entry_time'] = pd.to_datetime(trades['entry_time'])
    df.index = pd.to_datetime(df.index)

    for _, trade in trades.iterrows():
        entry_date = trade['entry_time']
        try:
            if entry_date in df.index:
                features = df.loc[entry_date].to_dict()
                features.update({
                    'trade_type': trade['type'],
                    'signal_type': trade['signal_type'],
                    'entry_price': trade['entry_price'],
                    'exit_price': trade['exit_price'],
                    'pnl': trade['pnl'],
                    'size': trade['size'],
                    'holding_period': trade['holding_period'],
                    'rsi_entry': trade['rsi_entry'],
                    'atr_entry': trade['atr_entry'],
                    'confidence_score': trade['confidence_score'],
                    'reason': trade['reason'],
                    'target': 1 if trade['pnl'] > 0 else 0
                })
                ai_data.append(features)
            else:
                logger.warning("Skipped trade: %s not in candles data", entry_date)
        except Exception as e:
            logger.error("Error processing trade at %s: %s", entry_date, e)
            continue

    result = pd.DataFrame(ai_data).dropna()
    logger.info("Prepared AI dataset with %s rows", len(result))
    return result

def train_xgboost_model(ai_data):
    """Train XGBoost classifier on AI dataset."""
    try:
        feature_cols = ['RSI', 'ADX', 'Stoch', 'BB_High', 'BB_Low', 'Volatility', 
                        'ATR', 'rsi_entry', 'atr_entry', 'holding_period', 'confidence_score']
        X = ai_data[feature_cols]
        y = ai_data['target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info("XGBoost model trained with accuracy: %.4f", accuracy)
        print("Accuracy:", accuracy)
        print("\nClassification Report:\n", classification_report(y_test, y_pred))

        importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        logger.info("Feature Importance:\n%s", importance)
        print("\nFeature Importance:\n", importance)

        return model, accuracy
    except Exception as e:
        logger.error("Error training XGBoost model: %s", e)
        return None, 0.0