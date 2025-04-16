import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib
import os
from scorer import train_scorer_model
from bot_trainer import train_model, extract_features

def train_models():
    # Train scorer model (setup_scorer.pkl)
    train_scorer_model()

    # Train trade predictor model (trade_predictor.pkl)
    trade_log_path = 'training/trade_log.csv'
    if os.path.exists(trade_log_path):
        train_model(trade_log_path=trade_log_path)
    else:
        print("No trade log found for training trade predictor model.")

if __name__ == "__main__":
    train_models()