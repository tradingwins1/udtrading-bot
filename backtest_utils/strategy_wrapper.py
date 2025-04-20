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

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class UGBacktestStrategy:
    def __init__(self):
        self.tsla_5min_data = None
        self.tsla_1min_data = None
        self.qqq_5min_data = None
        self.qqq_1min_data = None
        self.trades = []
        self.lgb_model = None
        self.lstm_model = None
        self.autoencoder = None
        self.dqn_model = None
        self.prophet_model = None

    def load_data(self, tsla_5min_path, tsla_1min_path, qqq_5min_path, qqq_1min_path):
        # Load TSLA 5-min data
        self.tsla_5min_data = pd.read_csv(tsla_5min_path)
        logger.info("Columns in %s: %s", tsla_5min_path, self.tsla_5min_data.columns.tolist())
        self.tsla_5min_data['timestamp'] = pd.to_datetime(self.tsla_5min_data['timestamp'], utc=True)
        self.tsla_5min_data.set_index('timestamp', inplace=True)
        logger.info("First few index values in %s: %s", tsla_5min_path, self.tsla_5min_data.index[:5].tolist())
        logger.info("Loaded %s with shape: %s", tsla_5min_path, self.tsla_5min_data.shape)

        # Load TSLA 1-min data
        self.tsla_1min_data = pd.read_csv(tsla_1min_path)
        logger.info("Columns in %s: %s", tsla_1min_path, self.tsla_1min_data.columns.tolist())
        self.tsla_1min_data['timestamp'] = pd.to_datetime(self.tsla_1min_data['timestamp'], utc=True)
        if not isinstance(self.tsla_1min_data.index, pd.DatetimeIndex):
            logger.warning("Index in %s is not a DatetimeIndex. Attempting to convert.", tsla_1min_path)
            logger.info("Sample index values before conversion in %s: %s", tsla_1min_path, self.tsla_1min_data['timestamp'].head(5).tolist())
            self.tsla_1min_data.set_index('timestamp', inplace=True)
        logger.info("First few index values in %s: %s", tsla_1min_path, self.tsla_1min_data.index[:5].tolist())
        logger.info("Loaded %s with shape: %s", tsla_1min_path, self.tsla_1min_data.shape)

        # Load QQQ 5-min data
        self.qqq_5min_data = pd.read_csv(qqq_5min_path)
        logger.info("Columns in %s: %s", qqq_5min_path, self.qqq_5min_data.columns.tolist())
        self.qqq_5min_data['timestamp'] = pd.to_datetime(self.qqq_5min_data['timestamp'], utc=True)
        self.qqq_5min_data.set_index('timestamp', inplace=True)
        logger.info("First few index values in %s: %s", qqq_5min_path, self.qqq_5min_data.index[:5].tolist())
        logger.info("Loaded %s with shape: %s", qqq_5min_path, self.qqq_5min_data.shape)

        # Load QQQ 1-min data
        self.qqq_1min_data = pd.read_csv(qqq_1min_path)
        logger.info("Columns in %s: %s", qqq_1min_path, self.qqq_1min_data.columns.tolist())
        self.qqq_1min_data['timestamp'] = pd.to_datetime(self.qqq_1min_data['timestamp'], utc=True)
        if not isinstance(self.qqq_1min_data.index, pd.DatetimeIndex):
            logger.warning("Index in %s is not a DatetimeIndex. Attempting to convert.", qqq_1min_path)
            logger.info("Sample index values before conversion in %s: %s", qqq_1min_path, self.qqq_1min_data['timestamp'].head(5).tolist())
            self.qqq_1min_data.set_index('timestamp', inplace=True)
        logger.info("First few index values in %s: %s", qqq_1min_path, self.qqq_1min_data.index[:5].tolist())
        logger.info("Loaded %s with shape: %s", qqq_1min_path, self.qqq_1min_data.shape)

    def initialize_models(self):
        try:
            self.lgb_model = lgb.Booster(model_file='lightgbm_trade_predictor.txt')
            logger.debug("LightGBM model loaded successfully")
        except Exception as e:
            logger.error("Failed to load LightGBM model: %s", e)
            self.lgb_model = None

        try:
            self.lstm_model = load_model('lstm_pattern_detector.h5')
            logger.debug("LSTM model loaded successfully")
        except Exception as e:
            logger.error("Failed to load LSTM model: %s", e)
            self.lstm_model = None

        try:
            self.autoencoder = load_model('autoencoder_anomaly_detector.h5')
            logger.debug("Autoencoder model loaded successfully")
        except Exception as e:
            logger.error("Failed to load Autoencoder model: %s", e)
            self.autoencoder = None

        try:
            self.dqn_model = load_model('dqn_trading_model.h5')
            logger.debug("DQN model loaded successfully")
        except Exception as e:
            logger.error("Failed to load DQN model: %s", e)
            self.dqn_model = None

        try:
            # Load Prophet model using pickle
            with open('prophet_price_predictor.pkl', 'rb') as f:
                self.prophet_model = pickle.load(f)
            logger.debug("Prophet model loaded successfully")
        except Exception as e:
            logger.error("Failed to load Prophet model: %s", e)
            self.prophet_model = None

    def get_mtf_trend(self, data):
        # Resample to 60-minute for MTF trend confirmation
        data_60min = data.resample('60min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        ema50_60min = ta.trend.EMAIndicator(data_60min['close'], window=50).ema_indicator()
        ema200_60min = ta.trend.EMAIndicator(data_60min['close'], window=200).ema_indicator()
        data_60min['mft_bullish'] = ema50_60min > ema200_60min
        data_60min['mft_bearish'] = ema50_60min < ema200_60min
        # Resample back to 5-minute
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
        # Placeholder for BOS signals (simplified)
        ema50 = ta.trend.EMAIndicator(data['close'], window=50).ema_indicator()
        ema200 = ta.trend.EMAIndicator(data['close'], window=200).ema_indicator()
        signals = pd.DataFrame(index=data.index)
        signals['bos_long'] = (data['close'] > ema50) & (ema50 > ema200) & (data['close'] > data['close'].shift(1))
        signals['bos_short'] = (data['close'] < ema50) & (ema50 < ema200) & (data['close'] < data['close'].shift(1))
        return signals

    def detect_gap_fill_reversal(self, data):
        # Placeholder for gap fill reversal signals
        signals = pd.DataFrame(index=data.index)
        signals['gap_fill_long'] = False
        signals['gap_fill_short'] = False
        return signals

    def detect_opening_range_signals(self, data):
        # Placeholder for ORB signals
        signals = pd.DataFrame(index=data.index)
        signals['orb_long'] = (data['close'] > data['open']) & (data['close'] > data['close'].shift(1))
        signals['orb_short'] = (data['close'] < data['open']) & (data['close'] < data['close'].shift(1))
        return signals

    def predict_trade_success(self, data_slice, confluences, direction):
        if self.lgb_model is None:
            return 0.5  # Default probability if model not loaded
        features = {
            'rsi': data_slice['RSI'].iloc[-1],
            'atr': data_slice['ATR'].iloc[-1],
        }
        feature_df = pd.DataFrame([features])
        prob = self.lgb_model.predict(feature_df)[0]
        return prob

    def detect_pattern(self, data_slice):
        if self.lstm_model is None:
            return True  # Default to allow trade if model not loaded
        return True  # Simplified for now

    def detect_anomaly(self, data_slice, confluences):
        if self.autoencoder is None:
            return False  # Default to no anomaly if model not loaded
        return False  # Simplified for now

    def precompute_prophet_predictions(self, data):
        """Precompute Prophet predictions for the entire dataset."""
        if self.prophet_model is None:
            logger.warning("Prophet model not loaded, setting price movements to 0.")
            return np.zeros(len(data))

        try:
            # Prepare data for Prophet
            df = data[['close']].reset_index().rename(columns={'timestamp': 'ds', 'close': 'y'})
            df['ds'] = df['ds'].dt.tz_localize(None)  # Remove timezone for Prophet

            # Predict for all timestamps
            logger.info("Running Prophet prediction for %d timestamps...", len(df))
            forecast = self.prophet_model.predict(df)
            future_prices = forecast['yhat'].values
            current_prices = df['y'].values

            # Compute price movements as percentage change
            movements = (future_prices - current_prices) / current_prices
            logger.info("Prophet predictions completed.")
            return movements
        except Exception as e:
            logger.error("Error in precomputing Prophet predictions: %s", e)
            return np.zeros(len(data))

    def close_position(self, timestamp, price, reason, signal_type, entry_time, entry_price, size, direction, data_slice):
        if not self.trades or self.trades[-1].get('exit_time') is not None:
            return
        trade = self.trades[-1]
        trade['exit_time'] = timestamp
        trade['exit_price'] = price
        trade['pnl'] = (price - trade['entry_price']) * size if direction == 'long' else (trade['entry_price'] - price) * size
        trade['holding_period'] = (timestamp - trade['entry_time']).total_seconds() / 3600  # in hours
        trade['reason'] = reason
        trade['result'] = 'win' if trade['pnl'] > 0 else 'loss'
        trade['rr_ratio'] = (trade['tp2'] - trade['entry_price']) / (trade['entry_price'] - trade['sl']) if direction == 'long' else (trade['entry_price'] - trade['tp2']) / (trade['sl'] - trade['entry_price'])
        trade['exit_bar'] = data_slice.index[-1]
        logger.info("Trade closed: signal_type=%s, pnl=%.2f, holding_period=%.2f hours", signal_type, trade['pnl'], trade['holding_period'])

    def run(self):
        # Prepare data with indicators
        self.tsla_5min_data = self.get_mtf_trend(self.tsla_5min_data)  # Add MTF trend
        # Adjust trading window for UTC (9:00 AM to 9:00 PM UTC to cover the data range)
        self.tsla_5min_data['time'] = self.tsla_5min_data.index.time
        self.tsla_5min_data['is_trading_window'] = (self.tsla_5min_data['time'] >= pd.Timestamp('09:00').time()) & (self.tsla_5min_data['time'] <= pd.Timestamp('21:00').time())
        self.tsla_5min_data['RSI'] = ta.momentum.RSIIndicator(self.tsla_5min_data['close'], window=14).rsi()
        self.tsla_5min_data['ATR'] = ta.volatility.AverageTrueRange(self.tsla_5min_data['high'], self.tsla_5min_data['low'], self.tsla_5min_data['close'], window=14).average_true_range()
        self.tsla_5min_data['SMA20'] = self.tsla_5min_data['close'].rolling(window=20).mean()
        self.tsla_5min_data['EMA50'] = ta.trend.EMAIndicator(self.tsla_5min_data['close'], window=50).ema_indicator()
        self.tsla_5min_data['EMA200'] = ta.trend.EMAIndicator(self.tsla_5min_data['close'], window=200).ema_indicator()
        self.tsla_5min_data['trend_bullish'] = self.tsla_5min_data['EMA50'] > self.tsla_5min_data['EMA200']
        self.tsla_5min_data['trend_bearish'] = self.tsla_5min_data['EMA50'] < self.tsla_5min_data['EMA200']
        self.tsla_5min_data['volume_sma'] = self.tsla_5min_data['volume'].rolling(window=20).mean()
        self.tsla_5min_data['high_volume'] = self.tsla_5min_data['volume'] > self.tsla_5min_data['volume_sma']

        # Precompute Prophet predictions
        prophet_movements = self.precompute_prophet_predictions(self.tsla_5min_data)
        self.tsla_5min_data['prophet_movement'] = prophet_movements

        # Generate signals
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

        # Log signal generation
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

        # Main backtest loop
        daily_trades = 0
        daily_pnl = 0
        daily_losers = 0
        equity = 1000000  # Same as Pine Script initial capital
        for i in range(len(self.tsla_5min_data)):
            if i % 1000 == 0:
                progress = (i / len(self.tsla_5min_data)) * 100
                eta = (len(self.tsla_5min_data) - i) * 0.001
                logger.info("Processing bar %d/%d (%.2f%%), ETA: %.2f seconds", i, len(self.tsla_5min_data), progress, eta)

            current = self.tsla_5min_data.iloc[i]
            timestamp = self.tsla_5min_data.index[i]
            logger.debug("Processing bar %d, timestamp=%s", i, timestamp)

            if timestamp.date() != self.tsla_5min_data.index[i-1].date() if i > 0 else True:
                daily_trades = 0
                daily_pnl = 0
                daily_losers = 0
                logger.debug("New day at bar %d, resetting daily limits", i)

            # Skip bars outside trading window
            if not current['is_trading_window']:
                logger.debug("Skipping bar %d: Outside trading window (09:00 - 21:00 UTC), timestamp=%s", i, timestamp)
                continue
            logger.debug("Bar %d within trading window", i)

            # Daily limits
            if daily_trades >= 3:
                logger.debug("Skipping bar %d: Daily trade limit reached (%d trades)", i, daily_trades)
                continue
            if daily_pnl <= -500:
                logger.debug("Skipping bar %d: Daily PNL limit reached (%.2f)", i, daily_pnl)
                continue
            if daily_losers >= 3:
                logger.debug("Skipping bar %d: Daily loser limit reached (%d losers)", i, daily_losers)
                continue

            # Match signals within a time window
            time_window = timedelta(minutes=10)
            matching_signals = signals[(signals.index >= (timestamp - time_window)) & (signals.index <= (timestamp + time_window))]
            if matching_signals.empty:
                logger.debug("No matching signals for bar %d, timestamp=%s", i, timestamp)
                continue
            logger.debug("Found %d matching signals for bar %d, timestamp=%s", len(matching_signals), i, timestamp)

            # Check open positions
            if self.trades:  # Only proceed if there are trades
                for trade in self.trades:
                    if trade['exit_time'] is not None:
                        continue
                    entry_time = trade['entry_time']
                    entry_price = trade['entry_price']
                    size = trade['size']
                    direction = trade['direction']
                    signal_type = trade['signal_type']
                    sl = trade['sl']
                    tp1 = trade['tp1']
                    tp2 = trade['tp2']
                    data_slice = self.tsla_5min_data.loc[entry_time:timestamp]

                    # Trailing stop
                    atr = current['ATR']
                    trail_multiplier = 1.2  # Same as Pine Script for stocks
                    trail_distance = atr * trail_multiplier
                    trail_offset = trail_distance / 2
                    if direction == 'long':
                        trail_stop = max(current['close'] - trail_distance, entry_price - trail_distance)
                        if current['close'] <= trail_stop:
                            self.close_position(timestamp, current['close'], "Trailing Stop", signal_type, entry_time, entry_price, size, direction, data_slice)
                            continue
                    else:
                        trail_stop = min(current['close'] + trail_distance, entry_price + trail_distance)
                        if current['close'] >= trail_stop:
                            self.close_position(timestamp, current['close'], "Trailing Stop", signal_type, entry_time, entry_price, size, direction, data_slice)
                            continue

                    # Timeout after 3 hours
                    holding_period = (timestamp - entry_time).total_seconds() / 3600  # in hours
                    if holding_period > 3:
                        self.close_position(timestamp, current['close'], "Timeout", signal_type, entry_time, entry_price, size, direction, data_slice)
                        continue

                    # Check SL/TP
                    if direction == 'long':
                        if current['close'] <= sl:
                            self.close_position(timestamp, current['close'], "SL Hit", signal_type, entry_time, entry_price, size, direction, data_slice)
                        elif current['close'] >= tp2:
                            self.close_position(timestamp, current['close'], "TP2 Hit", signal_type, entry_time, entry_price, size, direction, data_slice)
                        elif current['close'] >= tp1:
                            self.close_position(timestamp, current['close'], "TP1 Hit", signal_type, entry_time, entry_price, size, direction, data_slice)
                    else:
                        if current['close'] >= sl:
                            self.close_position(timestamp, current['close'], "SL Hit", signal_type, entry_time, entry_price, size, direction, data_slice)
                        elif current['close'] <= tp2:
                            self.close_position(timestamp, current['close'], "TP2 Hit", signal_type, entry_time, entry_price, size, direction, data_slice)
                        elif current['close'] <= tp1:
                            self.close_position(timestamp, current['close'], "TP1 Hit", signal_type, entry_time, entry_price, size, direction, data_slice)

            # Open new positions if no open trades
            if self.trades and self.trades[-1].get('exit_time') is None:
                continue

            for signal_idx, signal in matching_signals.iterrows():
                signal_type = None
                direction = None
                if signal['bos_long']:
                    signal_type = "Break of Structure (Long)"
                    direction = 'long'
                elif signal['bos_short']:
                    signal_type = "Break of Structure (Short)"
                    direction = 'short'
                elif signal['gap_fill_long']:
                    signal_type = "Gap Fill Reversal (Long)"
                    direction = 'long'
                elif signal['gap_fill_short']:
                    signal_type = "Gap Fill Reversal (Short)"
                    direction = 'short'
                elif signal['orb_long']:
                    signal_type = "Opening Range Break (Long)"
                    direction = 'long'
                elif signal['orb_short']:
                    signal_type = "Opening Range Break (Short)"
                    direction = 'short'
                elif signal['break_and_retest_long']:
                    signal_type = "Break and Retest (Long)"
                    direction = 'long'
                elif signal['break_and_retest_short']:
                    signal_type = "Break and Retest (Short)"
                    direction = 'short'

                if not signal_type or not direction:
                    logger.debug("Skipping signal at bar %d: No signal type or direction", i)
                    continue

                # MTF trend confirmation - Made optional
                if direction == 'long' and not current['mft_bullish']:
                    logger.debug("Allowing trade at bar %d despite MTF trend not bullish (EMA50_60min vs EMA200_60min)", i)
                if direction == 'short' and not current['mft_bearish']:
                    logger.debug("Allowing trade at bar %d despite MTF trend not bearish (EMA50_60min vs EMA200_60min)", i)

                # Momentum confirmation (RSI) - Relaxed further
                rsi = current['RSI']
                momentum_up = rsi > 30  # Relaxed from 40
                momentum_down = rsi < 70  # Relaxed from 60
                if direction == 'long' and not momentum_up:
                    logger.debug("Skipping trade at bar %d: RSI not above 30 (%.2f)", i, rsi)
                    continue
                if direction == 'short' and not momentum_down:
                    logger.debug("Skipping trade at bar %d: RSI not below 70 (%.2f)", i, rsi)
                    continue

                # QQQ alignment - Made optional if EMAs are NaN
                qqq_aligned = True  # Default to True if calculation fails
                qqq_slice = self.qqq_5min_data.loc[timestamp - timedelta(minutes=1000):timestamp]
                logger.debug("QQQ slice size for bar %d: %d bars", i, len(qqq_slice))
                if not qqq_slice.empty:
                    qqq_ema50 = ta.trend.EMAIndicator(qqq_slice['Close'], window=50).ema_indicator().iloc[-1]
                    qqq_ema200 = ta.trend.EMAIndicator(qqq_slice['Close'], window=200).ema_indicator().iloc[-1]
                    if not pd.isna(qqq_ema50) and not pd.isna(qqq_ema200):
                        qqq_aligned = (direction == 'long' and qqq_ema50 > qqq_ema200) or (direction == 'short' and qqq_ema50 < qqq_ema200)
                    else:
                        logger.debug("QQQ EMAs are NaN at bar %d (EMA50=%s, EMA200=%s), allowing trade", i,
                                     'nan' if pd.isna(qqq_ema50) else f"{qqq_ema50:.2f}",
                                     'nan' if pd.isna(qqq_ema200) else f"{qqq_ema200:.2f}")
                if not qqq_aligned:
                    logger.debug("Skipping trade at bar %d: QQQ trend not aligned (EMA50=%.2f, EMA200=%.2f)", i, qqq_ema50, qqq_ema200)
                    continue

                data_slice = self.tsla_5min_data.loc[timestamp - timedelta(minutes=30):timestamp]
                confluences = {
                    'ORB 5-min Break': signal_type.startswith("Opening Range Break"),
                    'Liquidity Sweep': data_slice['volume'].iloc[-1] > data_slice['volume'].mean(),
                    'Uptrend': current['trend_bullish'],
                    'Downtrend': current['trend_bearish'],
                    'QQQ Aligned': qqq_aligned,
                }

                # AI filters
                if not self.detect_pattern(data_slice):
                    logger.debug("Skipping trade at bar %d: LSTM detected low-probability pattern", i)
                    continue
                if self.detect_anomaly(data_slice, confluences):
                    logger.debug("Skipping trade at bar %d: Autoencoder detected anomaly", i)
                    continue

                # Use precomputed Prophet prediction
                price_movement = current['prophet_movement']
                if (direction == 'long' and price_movement < 0) or (direction == 'short' and price_movement > 0):
                    logger.debug("Skipping trade at bar %d: Predicted price movement against direction (%.2f%%)", i, price_movement * 100)
                    continue

                lgb_prob = self.predict_trade_success(data_slice, confluences, direction)
                if lgb_prob < 0.6:
                    logger.debug("Skipping trade at bar %d: LightGBM probability %.2f below 0.6", i, lgb_prob)
                    continue

                confidence_score = lgb_prob * 10
                if confidence_score < 6.0:
                    logger.debug("Skipping trade at bar %d: Confidence score %.2f below 6.0 (lgb_prob=%.2f)", i, confidence_score, lgb_prob)
                    continue

                # Calculate position size (2% of equity)
                equity = 1000000  # Same as Pine Script
                size = int((equity * 0.02) / current['close'])

                # ATR-based SL/TP
                atr = current['ATR']
                atr_multiplier_sl = 1.5  # Same as Pine Script for stocks
                atr_multiplier_tp = 3.0  # Same as Pine Script for stocks
                sl = current['close'] * (1 - 0.005) if direction == 'long' else current['close'] * (1 + 0.005)  # 0.5% SL
                tp1 = current['close'] * (1 + 0.01) if direction == 'long' else current['close'] * (1 - 0.01)  # 1% TP1
                tp2 = current['close'] * (1 + 0.015) if direction == 'long' else current['close'] * (1 - 0.015)  # 1.5% TP2

                # Check R:R ratio
                rr_ratio = (tp2 - current['close']) / (current['close'] - sl) if direction == 'long' else (current['close'] - tp2) / (sl - current['close'])
                if rr_ratio < 1.0:
                    logger.debug("Skipping trade at bar %d: R:R ratio %.2f below 1.0", i, rr_ratio)
                    continue

                if size > 0 and confidence_score >= 6.0:
                    trade = {
                        'entry_time': timestamp,
                        'entry_price': current['close'],
                        'pnl': 0,
                        'size': size,
                        'direction': direction,
                        'rsi_entry': current['RSI'],
                        'atr_entry': current['ATR'],
                        'signal_type': signal_type,
                        'confidence_score': confidence_score,
                        'sl': sl,
                        'tp1': tp1,
                        'tp2': tp2,
                        'confluences': confluences,
                        'entry_bar': i,
                        'exit_time': None,  # Initialize exit_time as None for new trades
                    }
                    self.trades.append(trade)
                    daily_trades += 1
                    logger.info("Opened position: bar=%d, type=%s, size=%d, signal=%s, confidence=%.2f, TP1=%.2f, TP2=%.2f", i, direction, size, signal_type, confidence_score, tp1, tp2)

        logger.info("Strategy run completed: %d trades executed", len(self.trades))

    def save_trades(self, output_path):
        trades_df = pd.DataFrame(self.trades)
        if not trades_df.empty:
            trades_df.to_csv(output_path, index=False)
            logger.info("Backtest completed. Trades saved to %s", output_path)
        else:
            logger.warning("No trades to save.")

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