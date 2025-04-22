import logging
import pandas as pd
import numpy as np
import ta
from ib_insync import *
from prophet import Prophet
from datetime import datetime, timedelta
import requests
import pickle
from scipy.stats import norm
import os
import math

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class UGBacktestStrategy:
    def __init__(self):
        self.initial_capital = 100000
        self.trades = []
        self.rejections = []
        self.trade_features = []
        self.prophet_model = Prophet()
        self.last_trade_time = None
        self.pending_signals = []
        logger.info("Initialized UGBacktestStrategy with initial capital: %.2f", self.initial_capital)

    def load_data(self, tsla_5min_path, tsla_1min_path, qqq_5min_path, qqq_1min_path):
        def standardize_columns(df, path):
            df.columns = df.columns.str.lower()
            expected_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in expected_columns):
                missing = [col for col in expected_columns if col not in df.columns]
                logger.error("Missing columns in %s: %s", path, missing)
                raise KeyError(f"Missing columns in {path}: {missing}")
            logger.info("Columns in %s: %s", path, df.columns.tolist())
            return df

        self.tsla_5min_data = standardize_columns(pd.read_csv(tsla_5min_path), tsla_5min_path)
        self.tsla_5min_data['timestamp'] = pd.to_datetime(self.tsla_5min_data['timestamp'], utc=True)
        self.tsla_5min_data.set_index('timestamp', inplace=True)
        logger.info("Loaded %s with shape: %s", tsla_5min_path, self.tsla_5min_data.shape)

        self.tsla_1min_data = standardize_columns(pd.read_csv(tsla_1min_path), tsla_1min_path)
        self.tsla_1min_data['timestamp'] = pd.to_datetime(self.tsla_1min_data['timestamp'], utc=True)
        self.tsla_1min_data.set_index('timestamp', inplace=True)
        logger.info("Loaded %s with shape: %s", tsla_1min_path, self.tsla_1min_data.shape)

        self.qqq_5min_data = standardize_columns(pd.read_csv(qqq_5min_path), qqq_5min_path)
        self.qqq_5min_data['timestamp'] = pd.to_datetime(self.qqq_5min_data['timestamp'], utc=True)
        self.qqq_5min_data.set_index('timestamp', inplace=True)
        logger.info("Loaded %s with shape: %s", qqq_5min_path, self.qqq_5min_data.shape)

        self.qqq_1min_data = standardize_columns(pd.read_csv(qqq_1min_path), qqq_1min_path)
        self.qqq_1min_data['timestamp'] = pd.to_datetime(self.qqq_1min_data['timestamp'], utc=True)
        self.qqq_1min_data.set_index('timestamp', inplace=True)
        logger.info("Loaded %s with shape: %s", qqq_1min_path, self.qqq_1min_data.shape)

        return self.tsla_5min_data

    def log_rejection(self, timestamp, reason, **kwargs):
        logger.debug("Rejection at %s: %s, kwargs: %s", timestamp, reason, kwargs)
        self.rejections.append({'timestamp': timestamp, 'reason': reason, **kwargs})

    def get_mtf_trend(self, data, timeframe='5min'):
        logger.debug("Calculating multi-timeframe trend for %s", timeframe)
        if timeframe != '5min':
            data = data.resample(timeframe).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
        data['sma50'] = data['close'].rolling(window=50).mean()
        data['sma200'] = data['close'].rolling(window=200).mean()
        data['trend'] = np.where(data['sma50'] > data['sma200'], 'bullish',
                                np.where(data['sma50'] < data['sma200'], 'bearish', 'neutral'))
        return data

    def identify_liquidity_zones(self, data, timeframe='D'):
        logger.debug("Identifying liquidity zones for %s", timeframe)
        data_tf = data.resample(timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        data_tf['swing_high'] = data_tf['high'].rolling(window=5, center=True).max()
        data_tf['swing_low'] = data_tf['low'].rolling(window=5, center=True).min()
        data_tf['fvg_up'] = np.where((data_tf['low'].shift(-1) > data_tf['high']) & (data_tf['close'].shift(-1) > data_tf['open'].shift(-1)), 
                                     (data_tf['low'].shift(-1) + data_tf['high']) / 2, np.nan)
        data_tf['fvg_down'] = np.where((data_tf['high'].shift(-1) < data_tf['low']) & (data_tf['close'].shift(-1) < data_tf['open'].shift(-1)), 
                                       (data_tf['high'].shift(-1) + data_tf['low']) / 2, np.nan)
        return data_tf

    def calculate_key_levels(self, data_5min):
        logger.debug("Calculating key levels (PMH, PML, PDH, PDL)")
        data_5min['date'] = data_5min.index.date
        data_5min['time'] = data_5min.index.time
        key_levels = {}
        for date in data_5min['date'].unique():
            day_data = data_5min[data_5min['date'] == date]
            pre_market = day_data[(day_data['time'] >= pd.Timestamp('04:00').time()) & 
                                 (day_data['time'] < pd.Timestamp('09:30').time())]
            if not pre_market.empty:
                key_levels[date] = {
                    'PMH': pre_market['high'].max(),
                    'PML': pre_market['low'].min()
                }
            else:
                key_levels[date] = {'PMH': np.nan, 'PML': np.nan}
        
        daily_data = data_5min.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        daily_data['PDH'] = daily_data['high'].shift(1)
        daily_data['PDL'] = daily_data['low'].shift(1)
        return key_levels, daily_data[['PDH', 'PDL']]

    def detect_break_and_retest(self, data):
        logger.debug("Detecting break and retest signals")
        signals = pd.DataFrame(index=data.index, columns=['break_and_retest_long', 'break_and_retest_short'])
        signals['break_and_retest_long'] = False
        signals['break_and_retest_short'] = False
        highs = data['high'].rolling(window=20).max()
        lows = data['low'].rolling(window=20).min()
        for i in range(21, len(data)):
            if data['close'].iloc[i-1] > highs.iloc[i-2] and data['low'].iloc[i] <= highs.iloc[i-2]:
                signals.loc[data.index[i], 'break_and_retest_long'] = True
            if data['close'].iloc[i-1] < lows.iloc[i-2] and data['high'].iloc[i] >= lows.iloc[i-2]:
                signals.loc[data.index[i], 'break_and_retest_short'] = True
        return signals

    def detect_ug_signals(self, data):
        logger.debug("Detecting UG signals (Break of Structure)")
        signals = pd.DataFrame(index=data.index, columns=['bos_long', 'bos_short'])
        signals['bos_long'] = False
        signals['bos_short'] = False
        for i in range(3, len(data)):
            if (data['high'].iloc[i-2] > data['high'].iloc[i-3] and
                data['low'].iloc[i-1] > data['low'].iloc[i-2] and
                data['close'].iloc[i] > data['high'].iloc[i-2]):
                signals.loc[data.index[i], 'bos_long'] = True
            if (data['low'].iloc[i-2] < data['low'].iloc[i-3] and
                data['high'].iloc[i-1] < data['high'].iloc[i-2] and
                data['close'].iloc[i] < data['low'].iloc[i-2]):
                signals.loc[data.index[i], 'bos_short'] = True
        return signals

    def detect_opening_range_signals(self, data, timeframe='5min'):
        logger.debug("Detecting opening range signals for %s timeframe", timeframe)
        signals = pd.DataFrame(index=data.index, columns=['orb_long', 'orb_short'])
        signals['orb_long'] = False
        signals['orb_short'] = False
        data = data.copy()
        data['date'] = data.index.date
        data['time'] = data.index.time
        for date in data['date'].unique():
            day_data = data[data['date'] == date]
            orb_data = day_data[(day_data['time'] >= pd.Timestamp('09:30').time()) & 
                               (day_data['time'] <= pd.Timestamp('09:45').time())]
            if not orb_data.empty:
                orb_high = orb_data['high'].max()
                orb_low = orb_data['low'].min()
                post_orb = day_data[day_data['time'] >= pd.Timestamp('09:45').time()]
                for i in post_orb.index:
                    if post_orb.loc[i, 'close'] > orb_high:
                        signals.loc[i, 'orb_long'] = True
                    if post_orb.loc[i, 'close'] < orb_low:
                        signals.loc[i, 'orb_short'] = True
        return signals

    def detect_elliott_wave_confirmation(self, data_5min, data_1h, direction, signal_type):
        logger.debug("Detecting Elliott Wave confirmation for %s direction, signal_type=%s", direction, signal_type)
        data_5min = data_5min.copy()
        data_1h = data_1h.copy()

        wave_confirmation = False
        if len(data_5min) < 50:
            logger.debug("Insufficient data for Elliott Wave analysis")
            return wave_confirmation

        # Get recent price range
        recent_high = data_5min['high'].iloc[-50:].max()
        recent_low = data_5min['low'].iloc[-50:].min()
        current_price = data_5min['close'].iloc[-1]
        diff = recent_high - recent_low

        # For break_and_retest signals, restrict to Wave 2, Wave 4, or OB continuation
        if signal_type in ['break_and_retest (Long)', 'break_and_retest (Short)']:
            # Identify prior impulsive move
            prior_high = data_5min['high'].iloc[-50:-25].max()
            prior_low = data_5min['low'].iloc[-50:-25].min()
            impulse_diff = prior_high - prior_low

            # Fibonacci levels for retracement
            fib_382 = prior_high - impulse_diff * 0.382 if direction == 'long' else prior_low + impulse_diff * 0.382
            fib_50 = prior_high - impulse_diff * 0.5 if direction == 'long' else prior_low + impulse_diff * 0.5
            fib_618 = prior_high - impulse_diff * 0.618 if direction == 'long' else prior_low + impulse_diff * 0.618

            # Check if current price is in retracement zone (Wave 2 or 4)
            retracement_zone = fib_382 <= current_price <= fib_618 if direction == 'long' else fib_618 <= current_price <= fib_382

            # Check for OB continuation (high-volume zone retest)
            volume_sma = data_5min['volume'].rolling(window=20).mean().iloc[-1]
            high_volume = data_5min['volume'].iloc[-5:].mean() > volume_sma * 1.5
            price_near_prior_level = abs(current_price - prior_high) / impulse_diff < 0.1 if direction == 'long' else abs(current_price - prior_low) / impulse_diff < 0.1

            # Wave 2 or 4: Retracement in 38.2-61.8% zone, no overlap with prior low (Wave 4)
            wave_2_4 = retracement_zone and (current_price > prior_low if direction == 'long' else current_price < prior_high)
            # OB continuation: High-volume zone retest
            ob_continuation = high_volume and price_near_prior_level

            wave_confirmation = wave_2_4 or ob_continuation
            logger.debug("Break and retest EW check: wave_2_4=%s, ob_continuation=%s, retracement_zone=%s, high_volume=%s",
                         wave_2_4, ob_continuation, retracement_zone, high_volume)
        else:
            # Original logic for other signals (e.g., bos)
            fib_0 = recent_high if direction == 'long' else recent_low
            fib_786 = recent_high - diff * 0.786 if direction == 'long' else recent_low + diff * 0.786
            if direction == 'long':
                wave_confirmation = fib_786 <= current_price <= fib_0
            else:
                wave_confirmation = fib_0 <= current_price <= fib_786

        logger.debug("Elliott Wave confirmation: direction=%s, signal_type=%s, confirmed=%s",
                     direction, signal_type, wave_confirmation)
        return wave_confirmation

    def check_news_event(self, current, bar):
        logger.debug("Checking for news event at bar %d", bar)
        atr_ratio = current['atr'] / current['atr_sma20']
        volume_ratio = current['volume'] / current['volume_sma']
        news_event = atr_ratio > 2.0 or volume_ratio > 3.0
        if news_event:
            logger.debug("News event detected: ATR ratio=%.2f, Volume ratio=%.2f", atr_ratio, volume_ratio)
        return news_event

    def check_displacement_candle(self, data_slice, direction):
        logger.debug("Checking displacement candle for %s", direction)
        if len(data_slice) < 1:
            logger.debug("Insufficient data for displacement candle check")
            return False
        latest_candle = data_slice.iloc[-1]
        body_size = abs(latest_candle['close'] - latest_candle['open'])
        candle_range = latest_candle['high'] - latest_candle['low']
        body_ratio = body_size / candle_range if candle_range != 0 else 0
        if direction == 'long':
            close_at_high = (latest_candle['close'] - latest_candle['low']) / candle_range > 0.4 if candle_range != 0 else False
            return body_ratio > 0.3 and close_at_high
        else:
            close_at_low = (latest_candle['high'] - latest_candle['close']) / candle_range > 0.4 if candle_range != 0 else False
            return body_ratio > 0.3 and close_at_low

    def check_momentum(self, data_slice, direction):
        logger.debug("Checking momentum for %s", direction)
        if len(data_slice) < 20:
            logger.debug("Insufficient data for momentum check")
            return False
        latest = data_slice.iloc[-1]
        volume_spike = latest['volume'] > data_slice['volume'].rolling(window=20).mean().iloc[-1] * 1.05
        return volume_spike

    def calculate_option_metrics(self, stock_price, buy_strike, sell_strike, days_to_expiry, iv):
        logger.debug("Calculating option metrics: stock_price=%.2f, buy_strike=%.2f, sell_strike=%.2f, days=%.2f, iv=%.2f",
                     stock_price, buy_strike, sell_strike, days_to_expiry, iv)
        risk_free_rate = 0.04
        t = days_to_expiry / 365
        if t <= 0 or np.isnan(stock_price):
            return 0.05, (sell_strike - buy_strike) * 100 * 0.1, 0.5
        d1 = (math.log(stock_price / buy_strike) + (risk_free_rate + iv**2 / 2) * t) / (iv * math.sqrt(t))
        d2 = d1 - iv * math.sqrt(t)
        call_price = stock_price * norm.cdf(d1) - buy_strike * math.exp(-risk_free_rate * t) * norm.cdf(d2)
        d1_sell = (math.log(stock_price / sell_strike) + (risk_free_rate + iv**2 / 2) * t) / (iv * math.sqrt(t))
        d2_sell = d1_sell - iv * math.sqrt(t)
        call_price_sell = stock_price * norm.cdf(d1_sell) - sell_strike * math.exp(-risk_free_rate * t) * norm.cdf(d2_sell)
        spread_delta = norm.cdf(d1) - norm.cdf(d1_sell)
        net_cost = (call_price - call_price_sell) * 100
        delta = norm.cdf(d1)
        return spread_delta, net_cost, delta

    def close_position(self, timestamp, current_price, reason, signal_type, entry_time, entry_stock_price, size, direction, data_slice, trade):
        logger.debug("Closing position at %s: %s, signal_type=%s, direction=%s", timestamp, reason, signal_type, direction)
        days_to_expiry = max(0, (trade['expiry_date'] - timestamp).total_seconds() / (24 * 3600))
        _, current_value, _ = self.calculate_option_metrics(
            current_price, trade['buy_strike'], trade['sell_strike'], days_to_expiry, trade['iv']
        )
        current_value *= size
        trade['exit_time'] = timestamp
        trade['exit_value'] = current_value
        trade['pnl'] = current_value - trade['entry_value']
        trade['result'] = 'profit' if trade['pnl'] > 0 else 'loss'
        self.trade_features.append({
            'rsi_entry': trade['rsi_entry'],
            'atr_entry': trade['atr_entry'],
            'adx_entry': trade['adx_entry'],
            'pnl': trade['pnl'],
            'direction': trade['direction'],
            'signal_type': signal_type
        })

    def precompute_prophet_predictions(self, data):
        logger.debug("Precomputing Prophet predictions")
        df_prophet = data[['close']].reset_index().rename(columns={'timestamp': 'ds', 'close': 'y'})
        df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)
        self.prophet_model.fit(df_prophet)
        future = self.prophet_model.make_future_dataframe(periods=0)
        forecast = self.prophet_model.predict(future)
        predictions = forecast['yhat'].values
        return predictions[:len(data)]

    def run(self, start_bar, end_bar):
        # Compute multi-timeframe trends
        self.tsla_daily = self.get_mtf_trend(self.tsla_5min_data, 'D')
        self.tsla_4h = self.get_mtf_trend(self.tsla_5min_data, '4H')
        self.tsla_1h = self.get_mtf_trend(self.tsla_5min_data, '1H')
        self.qqq_daily = self.get_mtf_trend(self.qqq_5min_data, 'D')
        self.qqq_4h = self.get_mtf_trend(self.qqq_5min_data, '4H')
        self.qqq_1h = self.get_mtf_trend(self.qqq_5min_data, '1H')

        # Identify liquidity zones
        self.tsla_liquidity_zones = self.identify_liquidity_zones(self.tsla_5min_data, 'D')

        # Calculate key levels (PMH, PML, PDH, PDL)
        self.key_levels, self.daily_levels = self.calculate_key_levels(self.tsla_5min_data)

        # Compute technical indicators for 5min data
        self.tsla_5min_data = self.get_mtf_trend(self.tsla_5min_data)
        self.qqq_5min_data = self.get_mtf_trend(self.qqq_5min_data)
        self.tsla_5min_data['time'] = self.tsla_5min_data.index.time
        self.tsla_5min_data['date'] = self.tsla_5min_data.index.date
        self.tsla_5min_data['is_trading_window'] = (self.tsla_5min_data['time'] >= pd.Timestamp('09:00').time()) & (self.tsla_5min_data['time'] <= pd.Timestamp('17:00').time())
        self.tsla_5min_data['rsi'] = ta.momentum.RSIIndicator(self.tsla_5min_data['close'], window=14).rsi()
        self.tsla_5min_data['atr'] = ta.volatility.AverageTrueRange(self.tsla_5min_data['high'], self.tsla_5min_data['low'], self.tsla_5min_data['close'], window=14).average_true_range()
        self.tsla_5min_data['sma20'] = self.tsla_5min_data['close'].rolling(window=20).mean()
        self.tsla_5min_data['ema50'] = ta.trend.EMAIndicator(self.tsla_5min_data['close'], window=50).ema_indicator()
        self.tsla_5min_data['ema200'] = ta.trend.EMAIndicator(self.tsla_5min_data['close'], window=200).ema_indicator()
        self.tsla_5min_data['trend_bullish'] = self.tsla_5min_data['ema50'] > self.tsla_5min_data['ema200']
        self.tsla_5min_data['trend_bearish'] = self.tsla_5min_data['ema50'] < self.tsla_5min_data['ema200']
        self.tsla_5min_data['volume_sma'] = self.tsla_5min_data['volume'].rolling(window=20).mean()
        self.tsla_5min_data['high_volume'] = self.tsla_5min_data['volume'] > self.tsla_5min_data['volume_sma']
        self.tsla_5min_data['adx'] = ta.trend.ADXIndicator(self.tsla_5min_data['high'], self.tsla_5min_data['low'], self.tsla_5min_data['close'], window=14).adx()
        self.tsla_5min_data['atr_sma20'] = self.tsla_5min_data['atr'].rolling(window=20).mean()
        self.tsla_5min_data['rsi_slope'] = self.tsla_5min_data['rsi'].diff(5) / 5
        vwap = (self.tsla_5min_data['close'] * self.tsla_5min_data['volume']).cumsum() / self.tsla_5min_data['volume'].cumsum()
        self.tsla_5min_data['vwap'] = vwap
        self.tsla_5min_data['vwap_dev'] = (self.tsla_5min_data['close'] - vwap) / vwap

        # Precompute Prophet predictions
        prophet_movements = self.precompute_prophet_predictions(self.tsla_5min_data)
        self.tsla_5min_data['prophet_movement'] = prophet_movements

        # Detect trading signals
        signals = pd.DataFrame(index=self.tsla_5min_data.index)
        bos_signals = self.detect_ug_signals(self.tsla_5min_data)
        break_retest_signals = self.detect_break_and_retest(self.tsla_5min_data)
        orb_signals_5min = self.detect_opening_range_signals(self.tsla_5min_data, '5min')
        orb_signals_1min = self.detect_opening_range_signals(self.tsla_1min_data, '1min')

        signals['bos_long'] = bos_signals['bos_long']
        signals['bos_short'] = bos_signals['bos_short']
        signals['break_and_retest_long'] = break_retest_signals['break_and_retest_long']
        signals['break_and_retest_short'] = break_retest_signals['break_and_retest_short']
        signals['orb_5min_long'] = orb_signals_5min['orb_long']
        signals['orb_5min_short'] = orb_signals_5min['orb_short']
        signals['orb_1min_long'] = orb_signals_1min['orb_long']
        signals['orb_1min_short'] = orb_signals_1min['orb_short']

        equity = self.initial_capital
        open_positions = []

        # Process each bar within the specified range
        for i in range(max(200, start_bar), min(end_bar + 1, len(self.tsla_5min_data))):
            current = self.tsla_5min_data.iloc[i]
            timestamp = self.tsla_5min_data.index[i]
            bar = i
            current_date = current['date']

            if not current['is_trading_window']:
                logger.debug("Bar %d outside trading window", bar)
                continue

            logger.debug("Processing bar %d, timestamp=%s", bar, timestamp)

            # Manage open positions
            for trade in open_positions[:]:
                days_to_expiry = max(0, (trade['expiry_date'] - timestamp).total_seconds() / (24 * 3600))
                _, current_value, _ = self.calculate_option_metrics(
                    current['close'], trade['buy_strike'], trade['sell_strike'], days_to_expiry, trade['iv']
                )
                current_value *= trade['size']
                max_loss = trade['max_loss']
                target_profit = trade['entry_value'] * 8.0
                if 'highest_value' not in trade:
                    trade['highest_value'] = current_value
                else:
                    trade['highest_value'] = max(trade['highest_value'], current_value)
                trailing_stop = trade['highest_value'] * 0.70
                if current_value < trade['entry_value'] - max_loss:
                    self.close_position(
                        timestamp, current['close'], 'Loss Limit Exceeded', trade['signal_type'],
                        trade['entry_time'], trade['entry_stock_price'], trade['size'],
                        trade['direction'], self.tsla_5min_data.iloc[i-20:i+1], trade
                    )
                    self.trades.append(trade)
                    open_positions.remove(trade)
                    equity += current_value
                    logger.info("Equity updated after loss limit: %.2f", equity)
                elif trade['direction'] == 'long' and current_value < trailing_stop:
                    self.close_position(
                        timestamp, current['close'], 'Trailing Stop Hit', trade['signal_type'],
                        trade['entry_time'], trade['entry_stock_price'], trade['size'],
                        trade['direction'], self.tsla_5min_data.iloc[i-20:i+1], trade
                    )
                    self.trades.append(trade)
                    open_positions.remove(trade)
                    equity += current_value
                    logger.info("Equity updated after trailing stop: %.2f", equity)
                elif current_value >= target_profit:
                    self.close_position(
                        timestamp, current['close'], 'Profit Target Reached', trade['signal_type'],
                        trade['entry_time'], trade['entry_stock_price'], trade['size'],
                        trade['direction'], self.tsla_5min_data.iloc[i-20:i+1], trade
                    )
                    self.trades.append(trade)
                    open_positions.remove(trade)
                    equity += current_value
                    logger.info("Equity updated after profit target: %.2f", equity)
                else:
                    time_since_entry = (timestamp - trade['entry_time']).total_seconds()
                    if time_since_entry >= 21600 and current_value > trade['entry_value']:
                        self.close_position(
                            timestamp, current['close'], 'Time-based Profit Exit', trade['signal_type'],
                            trade['entry_time'], trade['entry_stock_price'], trade['size'],
                            trade['direction'], self.tsla_5min_data.iloc[i-20:i+1], trade
                        )
                        self.trades.append(trade)
                        open_positions.remove(trade)
                        equity += current_value
                        logger.info("Equity updated after time-based profit exit: %.2f", equity)

            # Commenting out minimum time between trades to allow more frequent trades
            # Enforce minimum time between trades
            # time_diff = 0
            # if self.last_trade_time is not None:
            #     time_diff = (timestamp - self.last_trade_time).total_seconds()
            # if time_diff < 1200:
            #     logger.debug("Skipping trade at bar %d: Recent trade within 1200 seconds", bar)
            #     continue

            # Check for news events
            if self.check_news_event(current, bar):
                self.log_rejection(timestamp, "News event detected, skipping trade", bar=bar)
                continue

            # Detect new signals
            matching_signals = []
            if signals['bos_long'].iloc[i]:
                matching_signals.append(('bos (Long)', 'long'))
            if signals['bos_short'].iloc[i]:
                matching_signals.append(('bos (Short)', 'short'))
            if signals['break_and_retest_long'].iloc[i] and (signals['orb_5min_long'].iloc[i] or signals['orb_1min_long'].iloc[i]):
                matching_signals.append(('break_and_retest (Long)', 'long'))
            if signals['break_and_retest_short'].iloc[i] and (signals['orb_5min_short'].iloc[i] or signals['orb_1min_short'].iloc[i]):
                matching_signals.append(('break_and_retest (Short)', 'short'))

            logger.debug("Found %d matching signals for bar %d, timestamp=%s", len(matching_signals), bar, timestamp)

            # Queue new signals
            for signal_type, direction in matching_signals:
                open_position_types = [trade['signal_type'] for trade in open_positions]
                if signal_type in open_position_types:
                    logger.debug("Skipping trade at bar %d: Already an open position with signal type %s", bar, signal_type)
                    continue
                confluences = {
                    'Uptrend': current['trend_bullish'],
                    'Downtrend': current['trend_bearish'],
                    'QQQ Aligned': True
                }
                self.pending_signals.append({
                    'signal_type': signal_type,
                    'direction': direction,
                    'confluences': confluences,
                    'bar': bar,
                    'timestamp': timestamp,
                    'wait_until': timestamp + timedelta(minutes=15)
                })

            # Process pending signals
            for pending in self.pending_signals[:]:
                if timestamp >= pending['wait_until']:
                    signal_type = pending['signal_type']
                    direction = pending['direction']
                    confluences = pending['confluences']
                    signal_bar = pending['bar']
                    signal_timestamp = pending['timestamp']

                    current_idx = i
                    if current_idx - signal_bar < 3:
                        continue

                    current_data = self.tsla_5min_data.iloc[i]
                    qqq_data = self.qqq_5min_data.loc[self.qqq_5min_data.index <= timestamp].iloc[-1] if timestamp in self.qqq_5min_data.index else self.qqq_5min_data.iloc[i]
                    confluences['Uptrend'] = current_data['trend_bullish']
                    confluences['Downtrend'] = current_data['trend_bearish']
                    confluences['QQQ Aligned'] = True

                    # Check trend compatibility for short trades
                    if direction == 'short':
                        if confluences['Uptrend'] or not confluences['Downtrend']:
                            self.log_rejection(timestamp, f"Trend mismatch for short: direction={direction}, uptrend={confluences['Uptrend']}, downtrend={confluences['Downtrend']}", signal_type=signal_type)
                            self.pending_signals.remove(pending)
                            continue

                    # Check displacement candle
                    if not self.check_displacement_candle(self.tsla_5min_data.iloc[i-1:i+1], direction):
                        self.log_rejection(timestamp, "No strong displacement candle after 15-min wait", signal_type=signal_type)
                        self.pending_signals.remove(pending)
                        continue

                    # Check momentum
                    if not self.check_momentum(self.tsla_5min_data.iloc[i-20:i+1], direction):
                        self.log_rejection(timestamp, "Insufficient momentum after 15-min wait", signal_type=signal_type)
                        self.pending_signals.remove(pending)
                        continue

                    # Check Elliott Wave confirmation for long trades or shorts in uptrend
                    if direction == 'long' or (direction == 'short' and confluences['Uptrend']):
                        data_1h = self.tsla_1min_data.resample('h').agg({
                            'open': 'first',
                            'high': 'max',
                            'low': 'min',
                            'close': 'last',
                            'volume': 'sum'
                        }).dropna()
                        if not self.detect_elliott_wave_confirmation(self.tsla_5min_data.iloc[i-50:i+1], data_1h, direction, signal_type):
                            self.log_rejection(timestamp, "No Elliott Wave confirmation after 15-min wait", signal_type=signal_type)
                            self.pending_signals.remove(pending)
                            continue

                    # Check momentum indicators
                    momentum_up = current_data['rsi'] < 80
                    momentum_down = current_data['rsi'] > 30
                    logger.debug("Momentum check after 15-min wait: RSI=%.2f, ADX=%.2f, momentum_up=%s, momentum_down=%s",
                                 current_data['rsi'], current_data['adx'], momentum_up, momentum_down)

                    if direction == 'long' and not momentum_up:
                        self.log_rejection(timestamp, "Insufficient momentum for long after 15-min wait", signal_type=signal_type)
                        self.pending_signals.remove(pending)
                        continue
                    if direction == 'short' and confluences['Uptrend']:
                        if not (current_data['rsi'] < 70 and current_data['adx'] < 35):
                            self.log_rejection(timestamp, "Insufficient correction indicators for short in uptrend after 15-min wait", signal_type=signal_type)
                            self.pending_signals.remove(pending)
                            continue
                    elif direction == 'short' and not momentum_down:
                        self.log_rejection(timestamp, "Insufficient momentum for short after 15-min wait", signal_type=signal_type)
                        self.pending_signals.remove(pending)
                        continue

                    # Calculate trade parameters
                    buy_strike = current_data['close'] + 1
                    sell_strike = buy_strike + 6
                    days_to_expiry = 7
                    iv = 0.4
                    spread_delta, net_cost, delta = self.calculate_option_metrics(
                        current_data['close'], buy_strike, sell_strike, days_to_expiry, iv
                    )
                    size = int(3 / max(spread_delta, 0.1))
                    size = max(1, min(size, 3))
                    entry_value = net_cost * size

                    # Set stop-loss based on the signal candle's low/high
                    signal_candle = self.tsla_5min_data.iloc[signal_bar]
                    if direction == 'long':
                        sl_price = signal_candle['low'] * 0.995  # 0.5% below the low
                        sl_value = self.calculate_option_metrics(
                            sl_price, buy_strike, sell_strike, days_to_expiry, iv
                        )[1] * size
                        max_loss = entry_value - sl_value
                        logger.debug("Stop-loss set: sl_price=%.2f, max_loss=%.2f", sl_price, max_loss)
                    else:
                        sl_price = signal_candle['high'] * 1.005  # 0.5% above the high
                        sl_value = self.calculate_option_metrics(
                            sl_price, buy_strike, sell_strike, days_to_expiry, iv
                        )[1] * size
                        max_loss = entry_value - sl_value
                        logger.debug("Stop-loss set: sl_price=%.2f, max_loss=%.2f", sl_price, max_loss)

                    current_date_str = current_date.strftime('%Y-%m-%d')
                    pmh = self.key_levels.get(current_date, {}).get('PMH', np.nan)
                    pml = self.key_levels.get(current_date, {}).get('PML', np.nan)
                    pdh = self.daily_levels.loc[current_date_str, 'PDH'] if current_date_str in self.daily_levels.index else np.nan
                    pdl = self.daily_levels.loc[current_date_str, 'PDL'] if current_date_str in self.daily_levels.index else np.nan

                    # Set take-profit levels
                    if direction == 'long':
                        tp1_price = pmh if not np.isnan(pmh) and current_data['close'] < pmh else (pml if not np.isnan(pml) else current_data['close'] * 1.005)
                        final_tp_price = pdh * 0.99 if not np.isnan(pdh) else current_data['close'] * 1.05
                        if current_data['atr'] / current_data['atr_sma20'] > 1.5:
                            tp1_price = current_data['close'] * (1 + current_data['vwap_dev'] * 0.5)
                    else:
                        tp1_price = pml if not np.isnan(pml) and current_data['close'] > pml else (pmh if not np.isnan(pmh) else current_data['close'] * 0.995)
                        final_tp_price = pdl * 1.01 if not np.isnan(pdl) else current_data['close'] * 0.95
                        if current_data['atr'] / current_data['atr_sma20'] > 1.5:
                            tp1_price = current_data['close'] * (1 - current_data['vwap_dev'] * 0.5)

                    _, tp1_value, _ = self.calculate_option_metrics(
                        tp1_price, buy_strike, sell_strike, days_to_expiry, iv
                    )
                    tp1_value *= size
                    _, final_tp_value, _ = self.calculate_option_metrics(
                        final_tp_price, buy_strike, sell_strike, days_to_expiry, iv
                    )
                    final_tp_value *= size

                    logger.debug("Profit calculation after 15-min wait: bar=%d, net_cost=%.2f, size=%d, entry_value=%.2f, tp1_value=%.2f, final_tp_value=%.2f",
                                 bar, net_cost, size, entry_value, tp1_value, final_tp_value)

                    # Create trade record
                    trade = {
                        'entry_time': timestamp,
                        'entry_stock_price': current_data['close'],
                        'entry_value': entry_value,
                        'max_loss': max_loss,
                        'tp1_value': tp1_value,
                        'target_value': final_tp_value,
                        'size': size,
                        'direction': direction,
                        'rsi_entry': current_data['rsi'],
                        'atr_entry': current_data['atr'],
                        'adx_entry': current_data['adx'],
                        'signal_type': signal_type,
                        'buy_strike': buy_strike,
                        'sell_strike': sell_strike,
                        'iv': iv,
                        'expiry_date': timestamp + timedelta(days=days_to_expiry),
                        'confluences': confluences,
                        'entry_bar': bar
                    }

                    open_positions.append(trade)
                    equity -= entry_value
                    self.last_trade_time = timestamp
                    logger.info("Opened position after 15-min wait: bar=%d, type=%s, size=%d, signal=%s, net_cost=%.2f, equity=%.2f",
                                bar, direction, size, signal_type, entry_value, equity)

                    self.pending_signals.remove(pending)

        # Close remaining open positions
        for trade in open_positions:
            days_to_expiry = max(0, (trade['expiry_date'] - self.tsla_5min_data.index[-1]).total_seconds() / (24 * 3600))
            _, current_value, _ = self.calculate_option_metrics(
                self.tsla_5min_data['close'].iloc[-1], trade['buy_strike'], trade['sell_strike'], days_to_expiry, trade['iv']
            )
            current_value *= trade['size']
            self.close_position(
                self.tsla_5min_data.index[-1], self.tsla_5min_data['close'].iloc[-1], 'End of Backtest',
                trade['signal_type'], trade['entry_time'], trade['entry_stock_price'], trade['size'],
                trade['direction'], self.tsla_5min_data.iloc[-20:], trade
            )
            self.trades.append(trade)
            equity += current_value
            logger.info("Equity updated after end of backtest: %.2f", equity)

        logger.info("Strategy run completed: %d trades executed", len(self.trades))

    def save_trades(self, filename):
        trades_df = pd.DataFrame(self.trades)
        trades_df.to_csv(filename, index=False)
        logger.info("Trades saved to %s", filename)