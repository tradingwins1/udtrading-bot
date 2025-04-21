import logging
import pandas as pd
import numpy as np
import ta
from ib_insync import *
import lightgbm as lgb
from tensorflow.keras.models import load_model
from prophet import Prophet
from datetime import datetime, timedelta
import requests
import pickle
from scipy.stats import norm
import os
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class UGBacktestStrategy:
    def __init__(self):
        self.tsla_5min_data = None
        self.tsla_1min_data = None
        self.qqq_5min_data = None
        self.qqq_1min_data = None
        self.trades = []
        self.rejections = []
        self.trade_features = []
        self.lgb_model = None
        self.lstm_model = None
        self.autoencoder = None
        self.dqn_model = None
        self.prophet_model = None
        self.initial_capital = 15000
        self.capital_risk_percent = 0.01
        self.scaler = MinMaxScaler()

    def log_rejection(self, timestamp, reason, rr_ratio=None, lgbm_score=None, expected_profit=None):
        rejection = {
            'timestamp': timestamp,
            'reason': reason,
            'rr_ratio': rr_ratio,
            'lgbm_score': lgbm_score,
            'expected_profit': expected_profit
        }
        self.rejections.append(rejection)
        logger.debug("Trade rejected: timestamp=%s, reason=%s, rr_ratio=%.2f, lgbm_score=%.2f, expected_profit=%.2f",
                     timestamp, reason, rr_ratio if rr_ratio is not None else -1,
                     lgbm_score if lgbm_score is not None else -1,
                     expected_profit if expected_profit is not None else -1)

    def load_data(self, tsla_5min_path, tsla_1min_path, qqq_5min_path, qqq_1min_path):
        self.tsla_5min_data = pd.read_csv(tsla_5min_path)
        logger.info("Columns in %s: %s", tsla_5min_path, self.tsla_5min_data.columns.tolist())
        self.tsla_5min_data['timestamp'] = pd.to_datetime(self.tsla_5min_data['timestamp'], utc=True)
        self.tsla_5min_data.set_index('timestamp', inplace=True)
        logger.info("First few index values in %s: %s", tsla_5min_path, self.tsla_5min_data.index[:5].tolist())
        logger.info("Loaded %s with shape: %s", tsla_5min_path, self.tsla_5min_data.shape)

        self.tsla_1min_data = pd.read_csv(tsla_1min_path)
        self.tsla_1min_data['timestamp'] = pd.to_datetime(self.tsla_1min_data['timestamp'], utc=True)
        self.tsla_1min_data.set_index('timestamp', inplace=True)
        logger.info("Loaded %s with shape: %s", tsla_1min_path, self.tsla_1min_data.shape)

        self.qqq_5min_data = pd.read_csv(qqq_5min_path)
        self.qqq_5min_data['timestamp'] = pd.to_datetime(self.qqq_5min_data['timestamp'], utc=True)
        self.qqq_5min_data.set_index('timestamp', inplace=True)
        logger.info("Loaded %s with shape: %s", qqq_5min_path, self.qqq_5min_data.shape)

        self.qqq_1min_data = pd.read_csv(qqq_1min_path)
        self.qqq_1min_data['timestamp'] = pd.to_datetime(self.qqq_1min_data['timestamp'], utc=True)
        self.qqq_1min_data.set_index('timestamp', inplace=True)
        logger.info("Loaded %s with shape: %s", qqq_1min_path, self.qqq_1min_data.shape)

    def initialize_models(self):
        # Attempt to load existing LightGBM model
        try:
            self.lgb_model = pickle.load(open("models/lgb_model.pkl", "rb"))
            logger.info("LightGBM model loaded successfully.")
        except Exception as e:
            logger.warning("Failed to load LightGBM model: %s", e)
            self.lgb_model = None

        # Train a new model using synthetic and real data
        logger.info("Training a new LightGBM model with synthetic and real data...")
        synthetic_data = {
            'rsi_entry': np.random.uniform(30, 70, 10000),
            'atr_entry': np.random.uniform(0.1, 0.5, 10000),  # Widened range
            'adx_entry': np.random.uniform(10, 50, 10000),  # Widened range
            'volume_ratio': np.random.uniform(0.5, 3.0, 10000),  # Widened range
            'prophet_movement': np.random.uniform(-0.1, 0.1, 10000),  # Widened range
            'mft_bullish': np.random.choice([0, 1], 10000),
            'macd': np.random.uniform(-0.5, 0.5, 10000),  # Adjusted for TSLA
            'bollinger_width': np.random.uniform(0.01, 0.2, 10000),  # Adjusted for TSLA
            'vwap_distance': np.random.uniform(-0.2, 0.2, 10000),  # Adjusted for TSLA
            'result': np.random.choice([0, 1], 10000, p=[0.3, 0.7])
        }
        synthetic_df = pd.DataFrame(synthetic_data)

        # Include real trade data if available
        real_data = []
        try:
            real_df = pd.read_csv('trade_features.csv')
            real_data = {
                'rsi_entry': real_df['rsi_entry'],
                'atr_entry': real_df['atr_entry'],
                'adx_entry': real_df['adx_entry'],
                'volume_ratio': real_df['volume_ratio'],
                'prophet_movement': self.tsla_5min_data['prophet_movement'].iloc[:len(real_df)],
                'mft_bullish': self.tsla_5min_data['mft_bullish'].iloc[:len(real_df)].astype(int),
                'macd': self.tsla_5min_data['macd'].iloc[:len(real_df)],
                'bollinger_width': self.tsla_5min_data['bollinger_width'].iloc[:len(real_df)],
                'vwap_distance': self.tsla_5min_data['vwap_distance'].iloc[:len(real_df)],
                'result': real_df['result']
            }
            real_df = pd.DataFrame(real_data).dropna()
            real_df = resample(real_df, replace=True, n_samples=10000, random_state=42)  # Increased to 10000
            features_df = pd.concat([synthetic_df, real_df], ignore_index=True)
        except Exception:
            features_df = synthetic_df

        # Scale features
        feature_columns = ['rsi_entry', 'atr_entry', 'adx_entry', 'volume_ratio', 'prophet_movement', 'mft_bullish', 'macd', 'bollinger_width', 'vwap_distance']
        X = features_df[feature_columns]
        y = features_df['result']
        X_scaled = self.scaler.fit_transform(X)

        # Train the model with cross-validation and tuned hyperparameters
        lgb_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=7, num_leaves=31, random_state=42)
        cv_scores = cross_val_score(lgb_model, X_scaled, y, cv=5, scoring='accuracy')
        logger.info("Cross-validation accuracy: %.2f ± %.2f", cv_scores.mean(), cv_scores.std())
        lgb_model.fit(X_scaled, y)
        self.lgb_model = lgb_model
        # Log feature importance
        feature_importance = pd.DataFrame({'feature': feature_columns, 'importance': lgb_model.feature_importances_})
        logger.info("Feature importance:\n%s", feature_importance.sort_values(by='importance', ascending=False))
        # Save the model
        os.makedirs("models", exist_ok=True)
        with open("models/lgb_model.pkl", "wb") as f:
            pickle.dump(lgb_model, f)
        logger.info("New LightGBM model trained and saved to models/lgb_model.pkl")

        try:
            self.lstm_model = load_model("models/lstm_model.h5")
            logger.info("LSTM model loaded successfully.")
        except Exception as e:
            logger.warning("Failed to load LSTM model: %s", e)
            self.lstm_model = None

        try:
            self.autoencoder = load_model("models/autoencoder_model.h5")
            logger.info("Autoencoder model loaded successfully.")
        except Exception as e:
            logger.warning("Failed to load Autoencoder model: %s", e)
            self.autoencoder = None

        try:
            self.dqn_model = load_model("models/dqn_trading_model.h5")
            logger.info("DQN model loaded successfully.")
        except Exception as e:
            logger.warning("Failed to load DQN model: %s", e)
            self.dqn_model = None

        try:
            self.prophet_model = Prophet()
            logger.info("Prophet model initialized.")
        except Exception as e:
            logger.warning("Failed to load Prophet model: %s", e)
            self.prophet_model = None

    def get_mtf_trend(self, data):
        data_60min = data.resample('60min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        ema50_60min = ta.trend.EMAIndicator(data_60min['close'], window=50).ema_indicator()
        ema200_60min = ta.trend.EMAIndicator(data_60min['close'], window=200).ema_indicator()
        data_60min['mft_bullish'] = ema50_60min > ema200_60min
        data_60min['mft_bearish'] = ema50_60min < ema200_60min
        data['mft_bullish'] = data_60min['mft_bullish'].reindex(data.index, method='ffill')
        data['mft_bearish'] = data_60min['mft_bearish'].reindex(data.index, method='ffill')
        return data

    def detect_break_and_retest(self, data):
        support = data['close'].rolling(window=20).min()
        resistance = data['close'].rolling(window=20).max()
        breakout = (data['close'] > resistance.shift(1)) | (data['close'] < support.shift(1))
        retest = (data['close'] > support * 0.98) & (data['close'] < resistance * 1.02)
        signals = pd.DataFrame(index=data.index)
        signals['break_and_retest_long'] = breakout & retest & (data['close'] > resistance.shift(1))
        signals['break_and_retest_short'] = breakout & retest & (data['close'] < support.shift(1))
        return signals

    def detect_ug_signals(self, data):
        ema50 = ta.trend.EMAIndicator(data['close'], window=50).ema_indicator()
        ema200 = ta.trend.EMAIndicator(data['close'], window=200).ema_indicator()
        signals = pd.DataFrame(index=data.index)
        signals['bos_long'] = (data['close'] > ema50) & (ema50 > ema200) & (data['close'] > data['close'].shift(1))
        signals['bos_short'] = (data['close'] < ema50) & (ema50 < ema200) & (data['close'] < data['close'].shift(1))
        return signals

    def detect_gap_fill_reversal(self, data):
        signals = pd.DataFrame(index=data.index)
        signals['gap_fill_long'] = False
        signals['gap_fill_short'] = False
        return signals

    def detect_opening_range_signals(self, data):
        signals = pd.DataFrame(index=data.index)
        signals['orb_long'] = (data['close'] > data['open']) & (data['close'] > data['close'].shift(1))
        signals['orb_short'] = (data['close'] < data['open']) & (data['close'] < data['close'].shift(1))
        return signals

    def predict_trade_success(self, data_slice, confluences, direction):
        if self.lgb_model is None:
            return 0.5
        features = {
            'rsi_entry': data_slice['RSI'].iloc[-1],
            'atr_entry': data_slice['ATR'].iloc[-1],
            'adx_entry': data_slice['ADX'].iloc[-1],
            'volume_ratio': data_slice['volume'].iloc[-1] / data_slice['volume_sma'].iloc[-1],
            'prophet_movement': data_slice['prophet_movement'].iloc[-1],
            'mft_bullish': 1 if data_slice['mft_bullish'].iloc[-1] else 0,
            'macd': data_slice['macd'].iloc[-1],
            'bollinger_width': data_slice['bollinger_width'].iloc[-1],
            'vwap_distance': data_slice['vwap_distance'].iloc[-1]
        }
        feature_df = pd.DataFrame([features])
        required_features = ['rsi_entry', 'atr_entry', 'adx_entry', 'volume_ratio', 'prophet_movement', 'mft_bullish', 'macd', 'bollinger_width', 'vwap_distance']
        if not all(col in feature_df.columns for col in required_features):
            logger.error("Missing required features in predict_trade_success: %s", feature_df.columns.tolist())
            return 0.5
        feature_scaled = self.scaler.transform(feature_df[required_features])
        prob = self.lgb_model.predict_proba(feature_scaled)[0][1]  # Probability of win
        return prob

    def detect_pattern(self, data_slice):
        if self.lstm_model is None:
            return True
        return True

    def detect_anomaly(self, data_slice, confluences):
        if self.autoencoder is None:
            return False
        return False

    def precompute_prophet_predictions(self, data):
        if self.prophet_model is None:
            logger.warning("Prophet model not loaded, setting price movements to 0.")
            return np.zeros(len(data))
        try:
            df = data[['close']].reset_index().rename(columns={'timestamp': 'ds', 'close': 'y'})
            df['ds'] = df['ds'].dt.tz_localize(None)
            logger.info("Running Prophet prediction for %d timestamps...", len(df))
            forecast = self.prophet_model.predict(df)
            future_prices = forecast['yhat'].values
            current_prices = df['y'].values
            movements = (future_prices - current_prices) / current_prices
            logger.info("Prophet predictions completed.")
            return movements
        except Exception as e:
            logger.error("Error in Prophet predictions: %s", e)
            return np.zeros(len(data))

    def calculate_option_metrics(self, stock_price, buy_strike, sell_strike, days_to_expiry, iv=0.4):
        risk_free_rate = 0.01
        time_to_expiry = days_to_expiry / 365

        def calculate_d1(S, K, T, r, sigma):
            return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

        def calculate_delta(S, K, T, r, sigma):
            if T <= 0:
                return 1.0 if S > K else 0.0
            d1 = calculate_d1(S, K, T, r, sigma)
            delta = norm.cdf(d1)
            return max(0, min(1, delta))

        buy_delta = calculate_delta(stock_price, buy_strike, time_to_expiry, risk_free_rate, iv)
        sell_delta = calculate_delta(stock_price, sell_strike, time_to_expiry, risk_free_rate, iv)
        spread_delta = buy_delta - sell_delta

        buy_premium = max(0, (stock_price - buy_strike) * buy_delta + (iv * stock_price * (days_to_expiry / 365) ** 0.5))
        sell_premium = max(0, (stock_price - sell_strike) * sell_delta + (iv * stock_price * (days_to_expiry / 365) ** 0.5))
        net_cost = (buy_premium - sell_premium) * 100
        net_cost *= 1.05  # 5% slippage

        return spread_delta, net_cost

    def close_position(self, timestamp, stock_price, reason, signal_type, entry_time, entry_stock_price, size, direction, data_slice, trade):
        buy_strike = trade['buy_strike']
        sell_strike = trade['sell_strike']
        days_to_expiry = max(0, (trade['expiry_date'] - timestamp).total_seconds() / (24 * 3600))

        try:
            _, current_value = self.calculate_option_metrics(stock_price, buy_strike, sell_strike, days_to_expiry, iv=trade['iv'])
        except Exception as e:
            logger.error("Error in calculate_option_metrics: %s", e)
            current_value = 0

        trade['exit_time'] = timestamp
        trade['exit_stock_price'] = stock_price
        trade['exit_value'] = current_value
        trade['pnl'] = current_value - trade['entry_value']
        trade['holding_period'] = (timestamp - trade['entry_time']).total_seconds() / 3600
        trade['reason'] = reason
        trade['result'] = 'win' if trade['pnl'] > 0 else 'loss'
        trade['rr_ratio'] = (trade['target_value'] - trade['entry_value']) / trade['max_loss']
        trade['exit_bar'] = data_slice.index[-1] if isinstance(data_slice.index, pd.Index) else timestamp
        logger.info("Trade closed: signal_type=%s, pnl=%.2f, holding_period=%.2f hours, exit_value=%.2f, reason=%s", 
                    signal_type, trade['pnl'], trade['holding_period'], trade['exit_value'], reason)

        # Log features for model retraining
        self.trade_features.append({
            'rsi_entry': trade['rsi_entry'],
            'atr_entry': trade['atr_entry'],
            'adx_entry': trade.get('adx_entry', 0),
            'volume_ratio': trade.get('volume_ratio', 1),
            'macd': trade.get('macd', 0),
            'bollinger_width': trade.get('bollinger_width', 0),
            'vwap_distance': trade.get('vwap_distance', 0),
            'signal_type': trade['signal_type'],
            'direction': trade['direction'],
            'result': 1 if trade['result'] == 'win' else 0
        })

    def run(self):
        self.tsla_5min_data = self.get_mtf_trend(self.tsla_5min_data)
        self.tsla_5min_data['time'] = self.tsla_5min_data.index.time
        self.tsla_5min_data['is_trading_window'] = (self.tsla_5min_data['time'] >= pd.Timestamp('09:00').time()) & (self.tsla_5min_data['time'] <= pd.Timestamp('17:00').time())
        self.tsla_5min_data['RSI'] = ta.momentum.RSIIndicator(self.tsla_5min_data['close'], window=14).rsi()
        self.tsla_5min_data['ATR'] = ta.volatility.AverageTrueRange(self.tsla_5min_data['high'], self.tsla_5min_data['low'], self.tsla_5min_data['close'], window=14).average_true_range()
        self.tsla_5min_data['SMA20'] = self.tsla_5min_data['close'].rolling(window=20).mean()
        self.tsla_5min_data['EMA50'] = ta.trend.EMAIndicator(self.tsla_5min_data['close'], window=50).ema_indicator()
        self.tsla_5min_data['EMA200'] = ta.trend.EMAIndicator(self.tsla_5min_data['close'], window=200).ema_indicator()
        self.tsla_5min_data['trend_bullish'] = self.tsla_5min_data['EMA50'] > self.tsla_5min_data['EMA200']
        self.tsla_5min_data['trend_bearish'] = self.tsla_5min_data['EMA50'] < self.tsla_5min_data['EMA200']
        self.tsla_5min_data['volume_sma'] = self.tsla_5min_data['volume'].rolling(window=20).mean()
        self.tsla_5min_data['high_volume'] = self.tsla_5min_data['volume'] > self.tsla_5min_data['volume_sma']
        self.tsla_5min_data['ADX'] = ta.trend.ADXIndicator(self.tsla_5min_data['high'], self.tsla_5min_data['low'], self.tsla_5min_data['close'], window=14).adx()
        self.tsla_5min_data['ATR_SMA20'] = self.tsla_5min_data['ATR'].rolling(window=20).mean()
        self.tsla_5min_data['macd'] = ta.trend.MACD(self.tsla_5min_data['close']).macd()
        bbands = ta.volatility.BollingerBands(self.tsla_5min_data['close'], window=20)
        self.tsla_5min_data['bollinger_width'] = (bbands.bollinger_hband() - bbands.bollinger_lband()) / bbands.bollinger_mavg()
        vwap = (self.tsla_5min_data['close'] * self.tsla_5min_data['volume']).cumsum() / self.tsla_5min_data['volume'].cumsum()
        self.tsla_5min_data['vwap_distance'] = (self.tsla_5min_data['close'] - vwap) / vwap
        self.tsla_5min_data['support'] = self.tsla_5min_data['close'].rolling(window=20).min()
        self.tsla_5min_data['resistance'] = self.tsla_5min_data['close'].rolling(window=20).max()

        prophet_movements = self.precompute_prophet_predictions(self.tsla_5min_data)
        self.tsla_5min_data['prophet_movement'] = prophet_movements

        signals = pd.DataFrame(index=self.tsla_5min_data.index)
        bos_signals = self.detect_ug_signals(self.tsla_5min_data)
        gap_signals = self.detect_gap_fill_reversal(self.tsla_5min_data)
        orb_signals = self.detect_opening_range_signals(self.tsla_5min_data)
        break_retest_signals = self.detect_break_and_retest(self.tsla_5min_data)

        signals['bos_long'] = bos_signals['bos_long']
        signals['bos_short'] = bos_signals['bos_short']
        signals['gap_fill_long'] = gap_signals['gap_fill_long']
        signals['gap_fill_short'] = gap_signals['gap_fill_short']
        signals['orb_long'] = orb_signals['orb_long']
        signals['orb_short'] = orb_signals['orb_short']
        signals['break_and_retest_long'] = break_retest_signals['break_and_retest_long']
        signals['break_and_retest_short'] = break_retest_signals['break_and_retest_short']

        signal_timestamps = signals.index.tolist()

        bos_signals_df = signals[signals['bos_long'] | signals['bos_short']]
        if bos_signals_df.index.duplicated().any():
            logger.warning("Duplicates found in bos_signals_df: %s", bos_signals_df.index[bos_signals_df.index.duplicated()].tolist())
        gap_signals_df = signals[signals['gap_fill_long'] | signals['gap_fill_short']]
        if gap_signals_df.empty:
            logger.warning("No gap fill reversal signals generated.")
        else:
            logger.info("Merged signals: total=%d", len(gap_signals_df))
        orb_signals_df = signals[signals['orb_long'] | signals['orb_short']]
        signals = signals.groupby(signals.index).first()
        if signals.empty:
            logger.info("No signals generated. Check data and signal logic.")
        else:
            logger.info("ORB signals generated: %d, timestamps: %s", len(signals[signals['orb_long'] | signals['orb_short']]), signals[signals['orb_long'] | signals['orb_short']].index[:5].tolist())
            logger.info("BOS signals generated: %d, timestamps: %s", len(signals[signals['bos_long'] | signals['bos_short']]), signals[signals['bos_long'] | signals['bos_short']].index[:5].tolist())
            logger.info("Gap signals generated: %d, timestamps: %s", len(signals[signals['gap_fill_long'] | signals['gap_fill_short']]), signals[signals['gap_fill_long'] | signals['gap_fill_short']].index[:5].tolist())
            if signals.index.duplicated().any():
                logger.warning("Duplicates in signals before merging: %s", signals.index[signals.index.duplicated()].tolist())
            signals = signals.groupby(signals.index).first()
            logger.info("Total signals after merging: %d, timestamps: %s", len(signals), signals.index[:5].tolist())

        daily_trades = 0
        daily_pnl = 0
        daily_losers = 0
        equity = self.initial_capital
        logger.info("Starting backtest with initial equity: %.2f", equity)

        # For threshold analysis
        threshold_counts = {0.2: 0, 0.4: 0, 0.6: 0, 0.65: 0}

        for i in range(len(self.tsla_5min_data)):
            if i % 1000 == 0:
                progress = (i / len(self.tsla_5min_data)) * 100
                eta = (len(self.tsla_5min_data) - i) * 0.001
                logger.info("Processing bar %d/%d (%.2f%%), ETA: %.2f seconds, equity: %.2f", i, len(self.tsla_5min_data), progress, eta, equity)

            current = self.tsla_5min_data.iloc[i]
            timestamp = self.tsla_5min_data.index[i]
            logger.debug("Processing bar %d, timestamp=%s", i, timestamp)

            if timestamp.date() != self.tsla_5min_data.index[i-1].date() if i > 0 else True:
                daily_trades = 0
                daily_pnl = 0
                daily_losers = 0
                logger.debug("New day at bar %d, resetting daily limits", i)

            if not current['is_trading_window']:
                logger.debug("Skipping bar %d: Outside trading window (09:00 - 17:00 UTC), timestamp=%s", i, timestamp)
                self.log_rejection(timestamp, "Outside trading window")
                continue
            logger.debug("Bar %d within trading window", i)

            if daily_trades >= 30:  # Increased to 30
                logger.debug("Skipping bar %d: Daily trade limit reached (%d trades)", i, daily_trades)
                self.log_rejection(timestamp, "Daily trade limit reached")
                continue
            if daily_pnl <= -self.initial_capital * 0.05:
                logger.debug("Skipping bar %d: Daily PNL limit reached (%.2f)", i, daily_pnl)
                self.log_rejection(timestamp, "Daily PNL limit reached")
                continue
            if daily_losers >= 15:
                logger.debug("Skipping bar %d: Daily loser limit reached (%d losers)", i, daily_losers)
                self.log_rejection(timestamp, "Daily loser limit reached")
                continue

            # Check and close open trades
            for trade in self.trades:
                if trade.get('exit_time') is not None:
                    continue
                entry_time = trade['entry_time']
                entry_stock_price = trade['entry_stock_price']
                size = trade['size']
                direction = trade['direction']
                signal_type = trade['signal_type']
                entry_value = trade['entry_value']
                max_loss = trade['max_loss']
                target_value = trade['target_value']
                buy_strike = trade['buy_strike']
                sell_strike = trade['sell_strike']
                iv_entry = trade['iv']
                data_slice = self.tsla_5min_data.loc[entry_time:timestamp]

                days_to_expiry = max(0, (trade['expiry_date'] - timestamp).total_seconds() / (24 * 3600))
                _, current_value = self.calculate_option_metrics(current['close'], buy_strike, sell_strike, days_to_expiry, iv=iv_entry)
                current_iv = iv_entry

                # Trailing Stop
                profit_potential = target_value - entry_value
                if current_value >= entry_value + 0.1 * profit_potential:  # Adjusted to 10%
                    trailing_stop = current_value - (trade['atr_entry'] * 0.2)  # Trail at 0.2×ATR
                    if current_value <= trailing_stop:
                        self.close_position(timestamp, current['close'], "Trailing Stop Hit", signal_type, entry_time, entry_stock_price, size, direction, data_slice, trade)
                        trade['closed'] = True
                        daily_pnl += trade['pnl']
                        equity += trade['exit_value']
                        logger.info("Equity updated after trailing stop: %.2f", equity)
                        if trade['result'] == 'loss':
                            daily_losers += 1
                        else:
                            daily_losers = 0
                        continue

                # Break of Structure Invalidation
                recent_support = data_slice['support'].iloc[-1]
                recent_resistance = data_slice['resistance'].iloc[-1]
                if direction == 'long' and current['close'] < recent_support:
                    self.close_position(timestamp, current['close'], "Break of Structure Invalidation", signal_type, entry_time, entry_stock_price, size, direction, data_slice, trade)
                    trade['closed'] = True
                    daily_pnl += trade['pnl']
                    equity += trade['exit_value']
                    logger.info("Equity updated after structure invalidation: %.2f", equity)
                    if trade['result'] == 'loss':
                        daily_losers += 1
                    continue
                if direction == 'short' and current['close'] > recent_resistance:
                    self.close_position(timestamp, current['close'], "Break of Structure Invalidation", signal_type, entry_time, entry_stock_price, size, direction, data_slice, trade)
                    trade['closed'] = True
                    daily_pnl += trade['pnl']
                    equity += trade['exit_value']
                    logger.info("Equity updated after structure invalidation: %.2f", equity)
                    if trade['result'] == 'loss':
                        daily_losers += 1
                    continue

                if current_value >= target_value:
                    self.close_position(timestamp, current['close'], "Take-Profit Hit", signal_type, entry_time, entry_stock_price, size, direction, data_slice, trade)
                    trade['closed'] = True
                    daily_pnl += trade['pnl']
                    equity += trade['exit_value']
                    logger.info("Equity updated after take-profit: %.2f", equity)
                    if trade['result'] == 'loss':
                        daily_losers += 1
                    else:
                        daily_losers = 0
                    continue

                if current_value <= entry_value - max_loss:
                    self.close_position(timestamp, current['close'], "Stop-Loss Hit", signal_type, entry_time, entry_stock_price, size, direction, data_slice, trade)
                    trade['closed'] = True
                    daily_pnl += trade['pnl']
                    equity += trade['exit_value']
                    logger.info("Equity updated after stop-loss: %.2f", equity)
                    if trade['result'] == 'loss':
                        daily_losers += 1
                    continue

                if current_iv >= iv_entry * 1.1:  # Adjusted to 1.1
                    self.close_position(timestamp, current['close'], "IV Spike Exit", signal_type, entry_time, entry_stock_price, size, direction, data_slice, trade)
                    trade['closed'] = True
                    daily_pnl += trade['pnl']
                    equity += trade['exit_value']
                    logger.info("Equity updated after IV spike exit: %.2f", equity)
                    if trade['result'] == 'loss':
                        daily_losers += 1
                    continue

                holding_period = (timestamp - entry_time).total_seconds() / 3600
                if holding_period >= 24:  # Increased to 24 hours
                    self.close_position(timestamp, current['close'], "Timeout", signal_type, entry_time, entry_stock_price, size, direction, data_slice, trade)
                    trade['closed'] = True
                    daily_pnl += trade['pnl']
                    equity += trade['exit_value']
                    logger.info("Equity updated after timeout: %.2f", equity)
                    if trade['result'] == 'loss':
                        daily_losers += 1
                    continue

            # Remove closed trades
            self.trades = [trade for trade in self.trades if not trade.get('closed')]

            time_window = timedelta(minutes=10)
            matching_signals = signals[(signals.index >= (timestamp - time_window)) & (signals.index <= (timestamp + time_window))]
            if matching_signals.empty:
                logger.debug("No matching signals for bar %d, timestamp=%s", i, timestamp)
                self.log_rejection(timestamp, "No matching signals")
                continue
            logger.debug("Found %d matching signals for bar %d, timestamp=%s", len(matching_signals), i, timestamp)

            # Process signals and select one direction per signal type
            signal_types = matching_signals[['bos_long', 'bos_short', 'gap_fill_long', 'gap_fill_short', 'orb_long', 'orb_short', 'break_and_retest_long', 'break_and_retest_short']].any()
            for signal_type in signal_types.index:
                if not signal_types[signal_type]:
                    continue
                long_signal = signal_type.endswith('_long')
                short_signal = signal_type.endswith('_short')
                if long_signal or short_signal:
                    signal_name = signal_type.replace('_long', '').replace('_short', '')
                    long_prob = 0
                    short_prob = 0
                    if long_signal:
                        data_slice = self.tsla_5min_data.loc[timestamp - timedelta(minutes=30):timestamp]
                        confluences = {
                            'ORB 5-min Break': signal_type.startswith("orb"),
                            'Liquidity Sweep': data_slice['volume'].iloc[-1] > data_slice['volume'].mean(),
                            'Uptrend': current['trend_bullish'],
                            'Downtrend': current['trend_bearish'],
                            'QQQ Aligned': True
                        }
                        long_prob = self.predict_trade_success(data_slice, confluences, 'long')
                    if short_signal:
                        data_slice = self.tsla_5min_data.loc[timestamp - timedelta(minutes=30):timestamp]
                        confluences = {
                            'ORB 5-min Break': signal_type.startswith("orb"),
                            'Liquidity Sweep': data_slice['volume'].iloc[-1] > data_slice['volume'].mean(),
                            'Uptrend': current['trend_bullish'],
                            'Downtrend': current['trend_bearish'],
                            'QQQ Aligned': True
                        }
                        short_prob = self.predict_trade_success(data_slice, confluences, 'short')
                    # Select direction
                    if long_prob > short_prob and (current['trend_bullish'] or long_prob >= 0.65):
                        direction = 'long'
                        lgb_prob = long_prob
                        signal_type_full = f"{signal_name} (Long)"
                    elif short_prob > long_prob and (current['trend_bearish'] or short_prob >= 0.65):
                        direction = 'short'
                        lgb_prob = short_prob
                        signal_type_full = f"{signal_name} (Short)"
                    else:
                        logger.debug("Skipping signal %s at bar %d: No clear directional edge (long_prob=%.2f, short_prob=%.2f)", signal_type, i, long_prob, short_prob)
                        self.log_rejection(timestamp, "No clear directional edge", lgbm_score=max(long_prob, short_prob))
                        continue

                    # Process selected signal
                    for trade in self.trades:
                        if trade['signal_type'] == signal_type_full and trade['direction'] == direction and (timestamp - trade['entry_time']).total_seconds() < 600:  # Increased to 600 seconds
                            logger.debug("Skipping trade at bar %d: Duplicate signal %s already open", i, signal_type_full)
                            self.log_rejection(timestamp, "Duplicate signal already open")
                            break
                    else:
                        rsi = current['RSI']
                        adx = current['ADX']
                        strong_trend = adx > 25
                        if strong_trend:
                            momentum_up = rsi > 35  # Relaxed to 35
                            momentum_down = rsi < 65  # Relaxed to 65
                        else:
                            momentum_up = rsi > 35
                            momentum_down = rsi < 65

                        # EMA Crossover Confirmation
                        ema_crossover_long = current['EMA50'] > current['EMA200']
                        ema_crossover_short = current['EMA50'] < current['EMA200']

                        if direction == 'long' and not (momentum_up and ema_crossover_long):
                            logger.debug("Skipping trade at bar %d: RSI (%.2f) or EMA crossover not confirmed", i, rsi)
                            self.log_rejection(timestamp, "RSI or EMA crossover not confirmed")
                            continue
                        if direction == 'short' and not (momentum_down and ema_crossover_short):
                            logger.debug("Skipping trade at bar %d: RSI (%.2f) or EMA crossover not confirmed", i, rsi)
                            self.log_rejection(timestamp, "RSI or EMA crossover not confirmed")
                            continue

                        qqq_aligned = True
                        is_trend_day = current['trend_bullish'] or current['trend_bearish']
                        high_volatility = current['ATR'] > current['ATR_SMA20']
                        if not is_trend_day:
                            qqq_slice = self.qqq_5min_data.loc[timestamp - timedelta(minutes=1000):timestamp]
                            logger.debug("QQQ slice size for bar %d: %d bars", i, len(qqq_slice))
                            if not qqq_slice.empty:
                                qqq_ema50 = ta.trend.EMAIndicator(qqq_slice['Close'], window=50).ema_indicator().iloc[-1]
                                qqq_ema200 = ta.trend.EMAIndicator(qqq_slice['Close'], window=200).ema_indicator().iloc[-1]
                                if not pd.isna(qqq_ema50) and not pd.isna(qqq_ema200):
                                    if high_volatility:
                                        qqq_aligned = (direction == 'long' and qqq_ema50 > qqq_ema200 * 0.99) or (direction == 'short' and qqq_ema50 < qqq_ema200 * 1.01)
                                    else:
                                        qqq_aligned = (direction == 'long' and qqq_ema50 > qqq_ema200) or (direction == 'short' and qqq_ema50 < qqq_ema200)
                                else:
                                    logger.debug("QQQ EMAs are NaN at bar %d (EMA50=%s, EMA200=%s), allowing trade", i,
                                                 'nan' if pd.isna(qqq_ema50) else f"{qq_ema50:.2f}",
                                                 'nan' if pd.isna(qqq_ema200) else f"{qq_ema200:.2f}")
                        if not qqq_aligned:
                            logger.debug("Skipping trade at bar %d: QQQ trend not aligned (EMA50=%.2f, EMA200=%.2f)", i, qqq_ema50, qqq_ema200)
                            self.log_rejection(timestamp, "QQQ trend not aligned")
                            continue

                        data_slice = self.tsla_5min_data.loc[timestamp - timedelta(minutes=30):timestamp]
                        confluences = {
                            'ORB 5-min Break': signal_type_full.startswith("Opening Range Break"),
                            'Liquidity Sweep': data_slice['volume'].iloc[-1] > data_slice['volume'].mean(),
                            'Uptrend': current['trend_bullish'],
                            'Downtrend': current['trend_bearish'],
                            'QQQ Aligned': qqq_aligned,
                        }

                        if not self.detect_pattern(data_slice):
                            logger.debug("Skipping trade at bar %d: LSTM detected low-probability pattern", i)
                            self.log_rejection(timestamp, "LSTM detected low-probability pattern")
                            continue
                        if self.detect_anomaly(data_slice, confluences):
                            logger.debug("Skipping trade at bar %d: Autoencoder detected anomaly", i)
                            self.log_rejection(timestamp, "Autoencoder detected anomaly")
                            continue

                        # Dynamic LightGBM threshold
                        rr_ratio = (target_value - entry_value) / max_loss if 'max_loss' in locals() else 1
                        lgbm_threshold = 0.65 if rr_ratio >= 3 else 0.7
                        if lgb_prob < lgbm_threshold:
                            logger.debug("Skipping trade at bar %d: LightGBM probability %.2f below %.2f", i, lgb_prob, lgbm_threshold)
                            self.log_rejection(timestamp, f"LightGBM probability below {lgbm_threshold}", lgbm_score=lgb_prob)
                            continue

                        confidence_score = lgb_prob * 10
                        if confidence_score < 3.0:
                            logger.debug("Skipping trade at bar %d: Confidence score %.2f below 3.0 (lgb_prob=%.2f)", i, confidence_score, lgb_prob)
                            self.log_rejection(timestamp, "Confidence score below 3.0", lgbm_score=lgb_prob)
                            continue

                        stock_price = current['close']
                        buy_strike = stock_price + 1
                        sell_strike = stock_price + 5
                        days_to_expiry = 7
                        iv = 0.4
                        spread_delta, net_cost = self.calculate_option_metrics(stock_price, buy_strike, sell_strike, days_to_expiry, iv)

                        if spread_delta < 0.03:
                            logger.debug("Skipping trade at bar %d: Spread delta %.2f below 0.03", i, spread_delta)
                            self.log_rejection(timestamp, "Spread delta below 0.03")
                            continue
                        if iv > 0.5:
                            if signal_type_full.startswith("Opening Range Break") and timestamp.time() < pd.Timestamp('10:00').time():
                                logger.debug("Skipping trade at bar %d: IV %.2f above 0.5 with short expected hold", i, iv)
                                self.log_rejection(timestamp, "IV above 0.5 with short expected hold")
                                continue
                            logger.debug("Allowing trade despite high IV %.2f due to signal type or time", i, iv)

                        # Dynamic SL/TP based on ATR, LightGBM probability, and volatility
                        bar_atr = current['ATR']
                        volatility_factor = min(2.0, max(0.5, bar_atr / current['ATR_SMA20']))
                        if lgb_prob > 0.9:
                            sl_atr_mult = 2.5 * volatility_factor
                            tp_atr_mult = 10.0 * volatility_factor
                        elif lgb_prob > 0.85:
                            sl_atr_mult = 3.5 * volatility_factor
                            tp_atr_mult = 8.0 * volatility_factor
                        else:
                            sl_atr_mult = 5.0 * volatility_factor
                            tp_atr_mult = 6.0 * volatility_factor

                        size = max(1, int(equity * 0.01 / net_cost))
                        net_cost_total = net_cost * size
                        entry_value = net_cost * size
                        max_loss = bar_atr * sl_atr_mult * 100 * size
                        target_value = entry_value + (bar_atr * tp_atr_mult * 100 * size)

                        # Debug logging for profit calculation
                        logger.debug("Profit calculation: bar=%d, net_cost=%.2f, size=%d, entry_value=%.2f, bar_atr=%.2f, tp_atr_mult=%.2f, target_value=%.2f",
                                     i, net_cost, size, entry_value, bar_atr, tp_atr_mult, target_value)

                        if net_cost_total > equity:
                            logger.debug("Skipping trade at bar %d: Insufficient equity (%.2f required, %.2f available)", i, net_cost_total, equity)
                            self.log_rejection(timestamp, "Insufficient equity")
                            continue

                        # Capital Exposure Cap
                        if net_cost_total > 0.05 * equity:
                            logger.debug("Skipping trade at bar %d: Capital exposure exceeds 5%% (%.2f required, %.2f allowed)", i, net_cost_total, 0.05 * equity)
                            self.log_rejection(timestamp, "Capital exposure exceeds 5%")
                            continue

                        # Reward Floor: Adjust based on IV
                        expected_profit = target_value - entry_value
                        reward_floor = 40
                        if iv > 0.4:
                            reward_floor *= 1.2
                        if expected_profit < reward_floor:
                            if lgb_prob > 0.65:
                                logger.debug("High-confidence trade (lgb_prob=%.2f) below reward floor, proceeding", lgb_prob)
                            else:
                                logger.debug("Skipping trade at bar %d: Expected profit %.2f below $%.2f", i, expected_profit, reward_floor)
                                self.log_rejection(timestamp, "Expected profit below reward floor", expected_profit=expected_profit)
                                continue

                        max_profit = ((sell_strike - buy_strike) * 100 - net_cost) * size
                        rr_ratio = max_profit / max_loss

                        # Dynamic R:R Threshold
                        rr_threshold = 0.9
                        if iv < 0.35:
                            rr_threshold *= 0.9
                            logger.debug("Adjusting R:R threshold for low IV (%.2f): %.2f", iv, rr_threshold)
                        if pd.Timestamp('09:30').time() <= timestamp.time() <= pd.Timestamp('10:00').time():
                            rr_threshold *= 0.95
                            logger.debug("Adjusting R:R threshold for opening range: %.2f", rr_threshold)
                        if lgb_prob >= 0.6:
                            rr_threshold *= 0.9
                            logger.debug("Adjusting R:R threshold for high LightGBM prob (%.2f): %.2f", lgb_prob, rr_threshold)

                        # Soft Filtering with Trade Score
                        normalized_r_r = min(rr_ratio / 1.5, 1.0)
                        iv_score = 1.0 - min(iv / 0.5, 1.0)
                        trade_score = lgb_prob * 0.5 + normalized_r_r * 0.3 + iv_score * 0.2
                        logger.debug("Trade score: lgb_prob=%.2f, normalized_r_r=%.2f, iv_score=%.2f, trade_score=%.2f", lgb_prob, normalized_r_r, iv_score, trade_score)

                        if rr_ratio < 0.8:
                            logger.debug("Skipping trade at bar %d: R:R ratio %.2f below minimum 0.8", i, rr_ratio)
                            self.log_rejection(timestamp, "R:R ratio below minimum 0.8", rr_ratio=rr_ratio)
                            continue

                        if rr_ratio < rr_threshold and trade_score <= 0.75:
                            logger.debug("Skipping trade at bar %d: R:R ratio %.2f below threshold %.2f and trade_score %.2f <= 0.75", i, rr_ratio, rr_threshold, trade_score)
                            self.log_rejection(timestamp, "R:R ratio below threshold and trade score too low", rr_ratio=rr_ratio)
                            continue

                        if size > 0 and (trade_score > 0.75 or rr_ratio >= rr_threshold):
                            trade = {
                                'entry_time': timestamp,
                                'entry_stock_price': stock_price,
                                'entry_value': entry_value,
                                'max_loss': max_loss,
                                'target_value': target_value,
                                'size': size,
                                'direction': direction,
                                'rsi_entry': current['RSI'],
                                'atr_entry': current['ATR'],
                                'adx_entry': current['ADX'],
                                'volume_ratio': current['volume'] / current['volume_sma'],
                                'macd': current['macd'],
                                'bollinger_width': current['bollinger_width'],
                                'vwap_distance': current['vwap_distance'],
                                'signal_type': signal_type_full,
                                'confidence_score': confidence_score,
                                'buy_strike': buy_strike,
                                'sell_strike': sell_strike,
                                'iv': iv,
                                'expiry_date': timestamp + timedelta(days=7),
                                'confluences': confluences,
                                'entry_bar': i,
                                'exit_time': None,
                                'closed': False
                            }
                            self.trades.append(trade)
                            equity -= net_cost_total
                            daily_trades += 1
                            logger.info("Opened position: bar=%d, type=%s, size=%d, signal=%s, confidence=%.2f, net_cost=%.2f, equity=%.2f, rr_ratio=%.2f, trade_score=%.2f", 
                                        i, direction, size, signal_type_full, confidence_score, net_cost_total, equity, rr_ratio, trade_score)

        # Log threshold analysis
        logger.info("Threshold analysis - trades passing thresholds: 0.2: %d, 0.4: %d, 0.6: %d, 0.65: %d", 
                    threshold_counts[0.2], threshold_counts[0.4], threshold_counts[0.6], threshold_counts[0.65])

        # Ensure all trades are closed at the end
        last_timestamp = self.tsla_5min_data.index[-1]
        last_price = self.tsla_5min_data['close'].iloc[-1]
        last_data_slice = self.tsla_5min_data.iloc[-1:]
        for trade in self.trades:
            if trade['exit_time'] is None:
                self.close_position(
                    last_timestamp,
                    last_price,
                    "End of Backtest",
                    trade['signal_type'],
                    trade['entry_time'],
                    trade['entry_stock_price'],
                    trade['size'],
                    trade['direction'],
                    last_data_slice,
                    trade
                )
                equity += trade['exit_value']
                logger.info("Equity updated after end of backtest: %.2f", equity)

        logger.info("Strategy run completed: %d trades executed", len(self.trades))

    def save_trades(self, output_path):
        trades_df = pd.DataFrame(self.trades)
        if not trades_df.empty:
            trades_df.to_csv(output_path, index=False)
            logger.info("Backtest completed. Trades saved to %s", output_path)
        else:
            logger.warning("No trades to save.")

    def save_rejections(self, output_path):
        rejections_df = pd.DataFrame(self.rejections)
        if not rejections_df.empty:
            rejections_df.to_csv(output_path, index=False)
            logger.info("Rejection log saved to %s", output_path)
        else:
            logger.warning("No rejections to save.")

    def save_trade_features(self, output_path):
        features_df = pd.DataFrame(self.trade_features)
        if not features_df.empty:
            features_df.to_csv(output_path, index=False)
            logger.info("Trade features saved to %s for model retraining", output_path)
        else:
            logger.warning("No trade features to save.")

if __name__ == "__main__":
    strategy = UGBacktestStrategy()
    strategy.load_data(
        'data/TSLA_3M_5min_mock.csv',
        'data/TSLA_3M_1min_mock.csv',
        'data/QQQ_3M_5min_mock.csv',
        'data/QQQ_3M_1min_mock.csv'
    )
    strategy.initialize_models()
    strategy.run()
    strategy.save_trades('trades_output.csv')
    strategy.save_rejections('rejections_log.csv')
    strategy.save_trade_features('trade_features.csv')