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
import yaml
from sklearn.metrics import mean_squared_error

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
    logging.warning("xgboost not installed, falling back to heuristic signal scoring")

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class UGBacktestStrategy:
    def __init__(self, config_path='config.yaml', live_mode=False):
        self.live_mode = live_mode
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.config.setdefault('initial_capital', 5000)
        self.config.setdefault('risk_per_trade', 0.01)
        self.config.setdefault('commission_per_contract', 0.65)
        self.config.setdefault('slippage_pct', 0.0005)
        self.config.setdefault('min_rr_ratio', 2.0)
        self.config.setdefault('trading_window_start', '09:00')
        self.config.setdefault('trading_window_end', '17:00')
        self.initial_capital = self.config['initial_capital']
        self.risk_per_trade = self.config['risk_per_trade']
        self.commission_per_contract = self.config['commission_per_contract']
        self.slippage_pct = self.config['slippage_pct']
        self.min_rr_ratio = self.config['min_rr_ratio']
        self.trading_window_start = pd.Timestamp(self.config['trading_window_start']).time()
        self.trading_window_end = pd.Timestamp(self.config['trading_window_end']).time()
        self.trades = []
        self.rejections = []
        self.trade_features = []
        self.equity_curve = []
        self.last_trade_time = None
        self.pending_signals = []
        self.resampled_cache = {}
        self.signal_scorer = None
        logger.info("Initialized UGBacktestStrategy: live_mode=%s, initial_capital=%.2f, risk_per_trade=%.4f, version=optimized",
                    live_mode, self.initial_capital, self.risk_per_trade)

    def load_data(self, tsla_5min_path, tsla_1min_path, qqq_5min_path, qqq_1min_path):
        def standardize_columns(df, path):
            df.columns = df.columns.str.lower()
            expected_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in expected_columns):
                missing = [col for col in expected_columns if col not in df.columns]
                logger.error("Missing columns in %s: %s", path, missing)
                raise KeyError(f"Missing columns in {path}: {missing}")
            if df.empty or df['timestamp'].isna().any():
                logger.error("Empty or invalid timestamp data in %s", path)
                raise ValueError(f"Invalid data in {path}")
            logger.info("Columns in %s: %s", path, df.columns.tolist())
            return df

        if self.live_mode:
            ib = IB()
            ib.connect('127.0.0.1', 7497, clientId=1)
            contract_tsla = Stock('TSLA', 'SMART', 'USD')
            contract_qqq = Stock('QQQ', 'SMART', 'USD')
            try:
                self.tsla_5min_data = ib.reqHistoricalData(
                    contract_tsla, endDateTime='', durationStr='1 D', barSizeSetting='5 mins', whatToShow='TRADES', useRTH=True
                ).to_df()
                self.tsla_1min_data = ib.reqHistoricalData(
                    contract_tsla, endDateTime='', durationStr='1 D', barSizeSetting='1 min', whatToShow='TRADES', useRTH=True
                ).to_df()
                self.qqq_5min_data = ib.reqHistoricalData(
                    contract_qqq, endDateTime='', durationStr='1 D', barSizeSetting='5 mins', whatToShow='TRADES', useRTH=True
                ).to_df()
                self.qqq_1min_data = ib.reqHistoricalData(
                    contract_qqq, endDateTime='', durationStr='1 D', barSizeSetting='1 min', whatToShow='TRADES', useRTH=True
                ).to_df()
            except Exception as e:
                logger.error("Failed to fetch live data: %s", str(e))
                raise
            finally:
                ib.disconnect()
        else:
            self.tsla_5min_data = standardize_columns(pd.read_csv(tsla_5min_path), tsla_5min_path)
            self.tsla_1min_data = standardize_columns(pd.read_csv(tsla_1min_path), tsla_1min_path)
            self.qqq_5min_data = standardize_columns(pd.read_csv(qqq_5min_path), qqq_5min_path)
            self.qqq_1min_data = standardize_columns(pd.read_csv(qqq_1min_path), qqq_1min_path)

        self.tsla_5min_data['timestamp'] = pd.to_datetime(self.tsla_5min_data['timestamp'], utc=True)
        self.tsla_5min_data.set_index('timestamp', inplace=True)
        self.tsla_1min_data['timestamp'] = pd.to_datetime(self.tsla_1min_data['timestamp'], utc=True)
        self.tsla_1min_data.set_index('timestamp', inplace=True)
        self.qqq_5min_data['timestamp'] = pd.to_datetime(self.qqq_5min_data['timestamp'], utc=True)
        self.qqq_5min_data.set_index('timestamp', inplace=True)
        self.qqq_1min_data['timestamp'] = pd.to_datetime(self.qqq_1min_data['timestamp'], utc=True)
        self.qqq_1min_data.set_index('timestamp', inplace=True)

        logger.info("Loaded TSLA 5min: %s, TSLA 1min: %s, QQQ 5min: %s, QQQ 1min: %s",
                    self.tsla_5min_data.shape, self.tsla_1min_data.shape, self.qqq_5min_data.shape, self.qqq_1min_data.shape)

        split_idx = int(len(self.tsla_5min_data) * 0.7)
        self.train_data = self.tsla_5min_data.iloc[:split_idx]
        self.test_data = self.tsla_5min_data.iloc[split_idx:]
        return self.tsla_5min_data

    def log_rejection(self, timestamp, reason, **kwargs):
        logger.debug("Rejection at %s: %s, kwargs: %s", timestamp, reason, kwargs)
        self.rejections.append({'timestamp': timestamp, 'reason': reason, **kwargs})

    def get_mtf_trend(self, data, timeframe='5min'):
        cache_key = f"{timeframe}_{id(data)}"
        if cache_key in self.resampled_cache:
            return self.resampled_cache[cache_key]
        logger.debug("Calculating multi-timeframe trend for %s", timeframe)
        if timeframe != '5min':
            timeframe = timeframe.lower()
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
        self.resampled_cache[cache_key] = data
        return data

    def identify_liquidity_zones(self, data, timeframe='D'):
        logger.debug("Identifying liquidity zones for %s", timeframe)
        data_tf = self.get_mtf_trend(data, timeframe)
        data_tf['swing_high'] = data_tf['high'].rolling(window=5, center=True).max()
        data_tf['swing_low'] = data_tf['low'].rolling(window=5, center=True).min()
        data_tf['fvg_up'] = np.where((data_tf['low'].shift(-1) > data_tf['high']) & (data_tf['close'].shift(-1) > data_tf['open'].shift(-1)),
                                     (data_tf['low'].shift(-1) + data_tf['high']) / 2, np.nan)
        data_tf['fvg_down'] = np.where((data_tf['high'].shift(-1) < data_tf['low']) & (data_tf['close'].shift(-1) < data_tf['open'].shift(-1)),
                                       (data_tf['high'].shift(-1) + data_tf['low']) / 2, np.nan)
        return data_tf

    def calculate_key_levels(self, data_5min):
        logger.debug("Calculating key levels (PMH, PML, PDH, PDL)")
        data_5min = data_5min.copy()
        data_5min['date'] = data_5min.index.date
        data_5min['time'] = data_5min.index.time
        key_levels = {}
        for date in data_5min['date'].unique():
            day_data = data_5min[data_5min['date'] == date]
            pre_market = day_data[(day_data['time'] >= pd.Timestamp('04:00').time()) &
                                  (day_data['time'] < pd.Timestamp('09:30').time())]
            key_levels[date] = {
                'PMH': pre_market['high'].max() if not pre_market.empty else data_5min['close'].mean(),
                'PML': pre_market['low'].min() if not pre_market.empty else data_5min['close'].mean()
            }

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
        highs = data['high'].rolling(window=20).max()
        lows = data['low'].rolling(window=20).min()
        signals['break_and_retest_long'] = (data['close'].shift(1) > highs.shift(2)) & (data['low'] <= highs.shift(2))
        signals['break_and_retest_short'] = (data['close'].shift(1) < lows.shift(2)) & (data['high'] >= lows.shift(2))
        return signals

    def detect_ug_signals(self, data):
        logger.debug("Detecting UG signals (Break of Structure)")
        signals = pd.DataFrame(index=data.index, columns=['bos_long', 'bos_short'])
        signals['bos_long'] = (data['high'].shift(2) > data['high'].shift(3)) & \
                             (data['low'].shift(1) > data['low'].shift(2)) & \
                             (data['close'] > data['high'].shift(2))
        signals['bos_short'] = (data['low'].shift(2) < data['low'].shift(3)) & \
                              (data['high'].shift(1) < data['high'].shift(2)) & \
                              (data['close'] < data['low'].shift(2))
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
                signals.loc[post_orb.index, 'orb_long'] = post_orb['close'] > orb_high
                signals.loc[post_orb.index, 'orb_short'] = post_orb['close'] < orb_low
        return signals

    def detect_elliott_wave_confirmation(self, data_5min, data_1h, direction, signal_type):
        logger.debug("Detecting Elliott Wave confirmation for %s direction, signal_type=%s", direction, signal_type)
        return True

    def check_news_event(self, current, bar):
        logger.debug("Checking for news event at bar %d", bar)
        atr_ratio = current['atr'] / current['atr_sma20'] if current['atr_sma20'] != 0 else 1.0
        volume_ratio = current['volume'] / current['volume_sma'] if current['volume_sma'] != 0 else 1.0
        return atr_ratio > 2.0 or volume_ratio > 3.0

    def check_displacement_candle(self, data_slice, direction):
        logger.debug("Checking displacement candle for %s", direction)
        if len(data_slice) < 1:
            return False
        latest_candle = data_slice.iloc[-1]
        body_size = abs(latest_candle['close'] - latest_candle['open'])
        candle_range = latest_candle['high'] - latest_candle['low']
        body_ratio = body_size / candle_range if candle_range != 0 else 0
        atr_20 = data_slice['atr'].rolling(window=20).mean().iloc[-1]
        return body_ratio > 0.015 or candle_range > 0.7 * atr_20

    def check_momentum(self, data_slice, direction):
        logger.debug("Checking momentum for %s", direction)
        if len(data_slice) < 20:
            return False
        latest = data_slice.iloc[-1]
        sma20 = data_slice['close'].rolling(window=20).mean().iloc[-1]
        return (latest['volume'] > data_slice['volume'].rolling(window=20).mean().iloc[-1] * 1.1 or
                latest['adx'] > 15 or
                (latest['close'] > sma20 if direction == 'long' else latest['close'] < sma20))

    def calculate_historical_iv(self, data_slice):
        if len(data_slice) < 20:
            logger.debug("Data slice too short (<20), returning base IV: 0.7")
            return 0.7
        returns = data_slice['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) if not returns.empty else 0.1
        atr_ratio = data_slice['atr'].iloc[-1] / data_slice['atr_sma20'].iloc[-1] if data_slice['atr_sma20'].iloc[-1] != 0 else 1.0
        price_range_20 = data_slice['high'].rolling(window=20).max() - data_slice['low'].rolling(window=20).min()
        price_range_50 = data_slice['high'].rolling(window=50).max() - data_slice['low'].rolling(window=50).min()
        iv_adjust = 1.1 if price_range_20.iloc[-1] > price_range_50.iloc[-1] else 1.0
        iv_base = 0.7 if price_range_20.iloc[-1] <= price_range_50.iloc[-1] else 0.5
        iv = max(volatility * atr_ratio * 1.5 * iv_adjust, iv_base)
        iv = min(iv, 1.3)
        logger.debug("Calculated IV: volatility=%.4f, atr_ratio=%.2f, iv_adjust=%.2f, iv_base=%.2f, final_iv=%.2f",
                     volatility, atr_ratio, iv_adjust, iv_base, iv)
        return iv

    def calculate_option_metrics(self, stock_price, buy_strike, sell_strike, days_to_expiry, iv):
        logger.debug("Calculating option metrics: stock_price=%.2f, buy_strike=%.2f, sell_strike=%.2f, days=%.2f, iv=%.2f",
                     stock_price, buy_strike, sell_strike, days_to_expiry, iv)
        risk_free_rate = 0.04
        t = days_to_expiry / 365
        if t <= 0 or np.isnan(stock_price):
            logger.warning("Invalid time to expiry or stock price, returning default metrics")
            return 0.05, (sell_strike - buy_strike) * 100 * 0.1, 0.5
        try:
            d1 = (math.log(stock_price / buy_strike) + (risk_free_rate + iv**2 / 2) * t) / (iv * math.sqrt(t))
            d2 = d1 - iv * math.sqrt(t)
            call_price = stock_price * norm.cdf(d1) - buy_strike * math.exp(-risk_free_rate * t) * norm.cdf(d2)
            d1_sell = (math.log(stock_price / sell_strike) + (risk_free_rate + iv**2 / 2) * t) / (iv * math.sqrt(t))
            d2_sell = d1_sell - iv * math.sqrt(t)
            call_price_sell = stock_price * norm.cdf(d1_sell) - sell_strike * math.exp(-risk_free_rate * t) * norm.cdf(d2_sell)
            spread_delta = norm.cdf(d1) - norm.cdf(d1_sell)
            net_cost = (call_price - call_price_sell) * 100 * (1 + self.slippage_pct)
            delta = norm.cdf(d1)
            return spread_delta, net_cost, delta
        except Exception as e:
            logger.error("Error in option metrics calculation: %s, returning default metrics", str(e))
            return 0.05, (sell_strike - buy_strike) * 100 * 0.1, 0.5

    def close_position(self, timestamp, current_price, reason, signal_type, entry_time, entry_stock_price, size, direction, data_slice, trade, equity):
        logger.debug("Closing position at %s: %s, signal_type=%s, direction=%s", timestamp, reason, signal_type, direction)
        days_to_expiry = max(0, (trade['expiry_date'] - timestamp).total_seconds() / (24 * 3600))
        iv = self.calculate_historical_iv(data_slice)
        _, current_value, _ = self.calculate_option_metrics(
            current_price, trade['buy_strike'], trade['sell_strike'], days_to_expiry, iv
        )
        current_value *= size
        commission = size * self.commission_per_contract * 2
        trade['exit_time'] = timestamp
        trade['exit_value'] = current_value
        trade['pnl'] = current_value - trade['entry_value'] - commission
        trade['result'] = 'profit' if trade['pnl'] > 0 else 'loss'
        self.trade_features.append({
            'rsi_entry': trade['rsi_entry'],
            'atr_entry': trade['atr_entry'],
            'adx_entry': trade['adx_entry'],
            'pnl': trade['pnl'],
            'direction': trade['direction'],
            'signal_type': signal_type
        })
        logger.info("Closed trade: signal_type=%s, direction=%s, entry_price=%.2f, exit_price=%.2f, pnl=%.2f, reason=%s",
                    signal_type, direction, entry_stock_price, current_price, trade['pnl'], reason)
        return trade['pnl']

    def precompute_prophet_predictions(self, data):
        logger.debug("Precomputing Prophet predictions")
        self.prophet_model = Prophet()
        df_prophet = data[['close']].reset_index().rename(columns={'timestamp': 'ds', 'close': 'y'})
        df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)
        train_size = int(len(df_prophet) * 0.7)
        train_prophet = df_prophet.iloc[:train_size]
        try:
            self.prophet_model.fit(train_prophet)
            forecast = self.prophet_model.predict(df_prophet[['ds']])
            predictions = forecast['yhat'].values
        except Exception as e:
            logger.error("Error in Prophet prediction: %s, returning close prices as fallback", str(e))
            predictions = df_prophet['y'].values
        return predictions

    def calculate_performance_metrics(self):
        trades_df = pd.DataFrame(self.trades)
        if trades_df.empty:
            logger.warning("No trades to calculate metrics")
            return {'total_pnl': 0, 'win_rate': 0, 'profit_factor': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'expectancy': 0}

        total_pnl = trades_df['pnl'].sum()
        win_rate = len(trades_df[trades_df['pnl'] > 0]) / len(trades_df)
        profits = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        losses = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        profit_factor = profits / losses if losses != 0 else np.inf
        returns = trades_df['pnl'] / self.initial_capital
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
        equity = self.initial_capital + trades_df['pnl'].cumsum()
        max_drawdown = ((equity.cummax() - equity) / equity.cummax()).max() if not equity.empty else 0
        expectancy = (win_rate * trades_df[trades_df['pnl'] > 0]['pnl'].mean() -
                      (1 - win_rate) * abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean())) / self.initial_capital if not trades_df.empty else 0

        signal_summary = trades_df.groupby('signal_type').agg({
            'pnl': ['sum', 'count', 'mean'],
            'result': lambda x: (x == 'profit').mean()
        }).to_dict() if not trades_df.empty else {}
        logger.info("Signal Type Summary: %s", signal_summary)

        return {
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'expectancy': expectancy
        }

    def run(self, start_bar, end_bar, data_subset='all'):
        data = self.tsla_5min_data if data_subset == 'all' else self.train_data if data_subset == 'train' else self.test_data
        start_bar = max(200, start_bar)
        end_bar = min(end_bar + 1, len(data))

        self.tsla_daily = self.get_mtf_trend(data, 'D')
        self.tsla_4h = self.get_mtf_trend(data, '4h')
        self.tsla_1h = self.get_mtf_trend(data, '1h')
        self.qqq_daily = self.get_mtf_trend(self.qqq_5min_data, 'D')
        self.qqq_4h = self.get_mtf_trend(self.qqq_5min_data, '4h')
        self.qqq_1h = self.get_mtf_trend(self.qqq_5min_data, '1h')

        self.tsla_liquidity_zones = self.identify_liquidity_zones(data, 'D')
        self.key_levels, self.daily_levels = self.calculate_key_levels(data)

        data = data.copy()
        data['time'] = data.index.time
        data['date'] = data.index.date
        data['is_trading_window'] = (data['time'] >= self.trading_window_start) & (data['time'] <= self.trading_window_end)
        data['rsi'] = ta.momentum.RSIIndicator(data['close'], window=14).rsi()
        data['atr'] = ta.volatility.AverageTrueRange(data['high'], data['low'], data['close'], window=14).average_true_range()
        data['sma20'] = data['close'].rolling(window=20).mean()
        data['ema50'] = ta.trend.EMAIndicator(data['close'], window=50).ema_indicator()
        data['ema200'] = ta.trend.EMAIndicator(data['close'], window=200).ema_indicator()
        data['trend_bullish'] = data['ema50'] > data['ema200']
        data['trend_bearish'] = data['ema50'] < data['ema200']
        data['volume_sma'] = data['volume'].rolling(window=20).mean()
        data['high_volume'] = data['volume'] > data['volume_sma']
        data['adx'] = ta.trend.ADXIndicator(data['high'], data['low'], data['close'], window=14).adx()
        data['atr_sma20'] = data['atr'].rolling(window=20).mean()
        data['rsi_slope'] = data['rsi'].diff(5) / 5
        vwap = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
        data['vwap'] = vwap
        data['vwap_dev'] = (data['close'] - vwap) / vwap

        prophet_movements = self.precompute_prophet_predictions(data)
        data['prophet_movement'] = prophet_movements

        signals = pd.DataFrame(index=data.index)
        bos_signals = self.detect_ug_signals(data)
        break_retest_signals = self.detect_break_and_retest(data)
        orb_signals_5min = self.detect_opening_range_signals(data, '5min')
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

        for i in range(start_bar, end_bar):
            current = data.iloc[i]
            timestamp = data.index[i]
            bar = i
            current_date = current['date']

            if not current['is_trading_window']:
                continue

            logger.debug("Processing bar %d, timestamp=%s", bar, timestamp)
            self.equity_curve.append({'timestamp': timestamp, 'equity': equity})

            for trade in open_positions[:]:
                days_to_expiry = max(0, (trade['expiry_date'] - timestamp).total_seconds() / (24 * 3600))
                iv = self.calculate_historical_iv(data.iloc[i-20:i+1])
                _, current_value, _ = self.calculate_option_metrics(
                    current['close'], trade['buy_strike'], trade['sell_strike'], days_to_expiry, iv
                )
                current_value *= trade['size']
                max_loss = trade['max_loss']
                atr_ratio = current['atr'] / current['atr_sma20'] if current['atr_sma20'] != 0 else 1.0
                dynamic_rr = 1.8 if atr_ratio < 1.0 else self.min_rr_ratio
                target_profit = trade['entry_value'] + max_loss * dynamic_rr
                partial_profit = trade['entry_value'] + max_loss * 1.6
                trade['highest_value'] = max(trade.get('highest_value', current_value), current_value)
                trailing_stop = trade['highest_value'] * 0.85

                if atr_ratio < 0.8 and current_value > trade['entry_value'] * 0.99:
                    pnl = self.close_position(
                        timestamp, current['close'], 'Low Volatility Exit', trade['signal_type'],
                        trade['entry_time'], trade['entry_stock_price'], trade['size'],
                        trade['direction'], data.iloc[i-20:i+1], trade, equity
                    )
                    self.trades.append(trade)
                    open_positions.remove(trade)
                    equity += current_value - trade['size'] * self.commission_per_contract
                    logger.info("Equity updated after low volatility exit: %.2f", equity)
                    continue

                if current_value >= partial_profit and trade['size'] > 1:
                    partial_size = trade['size'] // 2
                    partial_value = current_value * partial_size / trade['size']
                    trade['size'] -= partial_size
                    current_value = current_value * trade['size'] / (trade['size'] + partial_size)
                    partial_pnl = self.close_position(
                        timestamp, current['close'], 'Partial Profit Taken', trade['signal_type'],
                        trade['entry_time'], trade['entry_stock_price'], partial_size,
                        trade['direction'], data.iloc[i-20:i+1], trade.copy(), equity
                    )
                    equity += partial_value - partial_size * self.commission_per_contract
                    logger.info("Partial position closed: size=%d, equity=%.2f", partial_size, equity)

                if current_value < trade['entry_value'] - max_loss:
                    pnl = self.close_position(
                        timestamp, current['close'], 'Loss Limit Exceeded', trade['signal_type'],
                        trade['entry_time'], trade['entry_stock_price'], trade['size'],
                        trade['direction'], data.iloc[i-20:i+1], trade, equity
                    )
                    self.trades.append(trade)
                    open_positions.remove(trade)
                    equity += current_value - trade['size'] * self.commission_per_contract
                    logger.info("Equity updated after loss limit: %.2f", equity)
                elif current_value < trailing_stop:
                    pnl = self.close_position(
                        timestamp, current['close'], 'Trailing Stop Hit', trade['signal_type'],
                        trade['entry_time'], trade['entry_stock_price'], trade['size'],
                        trade['direction'], data.iloc[i-20:i+1], trade, equity
                    )
                    self.trades.append(trade)
                    open_positions.remove(trade)
                    equity += current_value - trade['size'] * self.commission_per_contract
                    logger.info("Equity updated after trailing stop: %.2f", equity)
                elif current_value >= target_profit:
                    pnl = self.close_position(
                        timestamp, current['close'], 'Profit Target Reached', trade['signal_type'],
                        trade['entry_time'], trade['entry_stock_price'], trade['size'],
                        trade['direction'], data.iloc[i-20:i+1], trade, equity
                    )
                    self.trades.append(trade)
                    open_positions.remove(trade)
                    equity += current_value - trade['size'] * self.commission_per_contract
                    logger.info("Equity updated after profit target: %.2f", equity)
                else:
                    time_since_entry = (timestamp - trade['entry_time']).total_seconds()
                    if time_since_entry >= 259200 and current_value >= trade['entry_value'] * 0.997:
                        pnl = self.close_position(
                            timestamp, current['close'], 'Time-based Profit Exit', trade['signal_type'],
                            trade['entry_time'], trade['entry_stock_price'], trade['size'],
                            trade['direction'], data.iloc[i-20:i+1], trade, equity
                        )
                        self.trades.append(trade)
                        open_positions.remove(trade)
                        equity += current_value - trade['size'] * self.commission_per_contract
                        logger.info("Equity updated after time-based profit exit: %.2f", equity)

            if self.check_news_event(current, bar):
                self.log_rejection(timestamp, "News event detected, skipping trade", bar=bar)
                continue

            matching_signals = []
            signal_score = 0
            if signals['bos_long'].iloc[i]:
                matching_signals.append(('bos (Long)', 'long'))
                signal_score += 2.0 if current['high_volume'] else 1.0
            if signals['bos_short'].iloc[i]:
                matching_signals.append(('bos (Short)', 'short'))
                signal_score += 2.0 if current['high_volume'] else 1.0
            if signals['break_and_retest_long'].iloc[i]:
                matching_signals.append(('break_and_retest (Long)', 'long'))
                signal_score += 2.0 if current['high_volume'] else 1.5
            if signals['break_and_retest_short'].iloc[i]:
                matching_signals.append(('break_and_retest (Short)', 'short'))
                signal_score += 2.0 if current['high_volume'] else 1.5

            logger.debug("Found %d matching signals for bar %d, timestamp=%s, score=%.2f",
                         len(matching_signals), bar, timestamp, signal_score)

            for signal_type, direction in matching_signals:
                open_position_types = [trade['signal_type'] for trade in open_positions]
                if signal_type in open_position_types:
                    continue
                confluences = {
                    'Uptrend': current['trend_bullish'],
                    'Downtrend': current['trend_bearish'],
                    'QQQ Aligned': True,
                    'High Volume': current['high_volume'],
                    'Prophet Bullish': current['prophet_movement'] > current['close'] if direction == 'long' else current['prophet_movement'] < current['close'],
                    'Above SMA20': current['close'] > current['sma20'] if signal_type in ['bos (Long)', 'break_and_retest (Long)'] else True,
                    'Below SMA20': current['close'] < current['sma20'] if signal_type == 'break_and_retest (Short)' else True
                }
                if signal_type in ['bos (Long)', 'break_and_retest (Long)'] and not confluences['Above SMA20']:
                    self.log_rejection(timestamp, "Price below SMA20 for %s" % signal_type, signal_type=signal_type, score=signal_score)
                    continue
                if signal_type == 'break_and_retest (Short)' and not (current['close'] < current['sma20'] or current['rsi'] < 60):
                    self.log_rejection(timestamp, "Price above SMA20 or RSI >= 60 for break_and_retest (Short)", signal_type=signal_type, score=signal_score)
                    continue
                self.pending_signals.append({
                    'signal_type': signal_type,
                    'direction': direction,
                    'confluences': confluences,
                    'bar': bar,
                    'timestamp': timestamp,
                    'wait_until': timestamp + timedelta(minutes=15),
                    'score': signal_score
                })

            for pending in self.pending_signals[:]:
                if timestamp < pending['wait_until']:
                    continue
                signal_type = pending['signal_type']
                direction = pending['direction']
                confluences = pending['confluences']
                signal_bar = pending['bar']
                signal_timestamp = pending['timestamp']

                current_idx = i
                if current_idx - signal_bar < 3:
                    continue

                current_data = data.iloc[i]
                qqq_data = self.qqq_5min_data.loc[self.qqq_5min_data.index <= timestamp].iloc[-1] if timestamp in self.qqq_5min_data.index else self.qqq_5min_data.iloc[i]
                confluences['Uptrend'] = current_data['trend_bullish']
                confluences['Downtrend'] = current_data['trend_bearish']
                confluences['QQQ Aligned'] = True

                if direction == 'short' and confluences['Uptrend']:
                    if not (current_data['rsi_slope'] < -0.2 or current_data['rsi'] < 65):
                        self.log_rejection(timestamp, "No RSI divergence or low RSI for short in uptrend", signal_type=signal_type, score=pending['score'])
                        self.pending_signals.remove(pending)
                        continue

                if not self.check_displacement_candle(data.iloc[i-1:i+1], direction):
                    self.log_rejection(timestamp, "No displacement candle", signal_type=signal_type, score=pending['score'])
                    self.pending_signals.remove(pending)
                    continue

                if not self.check_momentum(data.iloc[i-20:i+1], direction):
                    self.log_rejection(timestamp, "Insufficient momentum", signal_type=signal_type, score=pending['score'])
                    self.pending_signals.remove(pending)
                    continue

                iv = self.calculate_historical_iv(data.iloc[i-20:i+1])
                buy_strike = current_data['close'] + current_data['atr'] * 0.5
                sell_strike = buy_strike + current_data['atr'] * 6
                days_to_expiry = 7
                spread_delta, net_cost, delta = self.calculate_option_metrics(
                    current_data['close'], buy_strike, sell_strike, days_to_expiry, iv
                )
                risk_amount = equity * self.risk_per_trade
                size = max(1, min(int(risk_amount / net_cost), 5))
                entry_value = net_cost * size

                signal_candle = data.iloc[signal_bar]
                atr = current_data['atr']
                atr_scale = min(max(1.0, current_data['atr_sma20'] / atr), 1.5) if atr > 0 else 1.0
                atr_ratio = current_data['atr'] / current_data['atr_sma20'] if current_data['atr_sma20'] != 0 else 1.0
                dynamic_rr = 1.8 if atr_ratio < 1.0 else self.min_rr_ratio
                if direction == 'long':
                    sl_price = signal_candle['low'] - max(atr * 3.0 * atr_scale, atr * 0.5)
                    tp_price = current_data['close'] + atr * dynamic_rr
                else:
                    sl_price = signal_candle['high'] + max(atr * 3.0 * atr_scale, atr * 0.5)
                    tp_price = current_data['close'] - atr * dynamic_rr

                sl_value = self.calculate_option_metrics(
                    sl_price, buy_strike, sell_strike, days_to_expiry, iv
                )[1] * size
                max_loss = entry_value - sl_value if direction == 'long' else sl_value - entry_value
                if max_loss <= 0:
                    self.log_rejection(timestamp, "Invalid stop-loss", signal_type=signal_type, score=pending['score'])
                    self.pending_signals.remove(pending)
                    continue

                tp1_value = self.calculate_option_metrics(
                    tp_price, buy_strike, sell_strike, days_to_expiry, iv
                )[1] * size
                final_tp_value = self.calculate_option_metrics(
                    tp_price * 1.05, buy_strike, sell_strike, days_to_expiry, iv
                )[1] * size

                current_date_str = current_date.strftime('%Y-%m-%d')
                pmh = self.key_levels.get(current_date, {}).get('PMH', current_data['close'])
                pml = self.key_levels.get(current_date, {}).get('PML', current_data['close'])
                pdh = self.daily_levels.loc[current_date_str, 'PDH'] if current_date_str in self.daily_levels.index else current_data['close']
                pdl = self.daily_levels.loc[current_date_str, 'PDL'] if current_date_str in self.daily_levels.index else current_data['close']

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
                equity -= entry_value + size * self.commission_per_contract
                self.last_trade_time = timestamp
                logger.info("Opened position: bar=%d, type=%s, size=%d, signal=%s, net_cost=%.2f, equity=%.2f",
                            bar, direction, size, signal_type, entry_value, equity)

                self.pending_signals.remove(pending)

        for trade in open_positions:
            days_to_expiry = max(0, (trade['expiry_date'] - data.index[-1]).total_seconds() / (24 * 3600))
            iv = self.calculate_historical_iv(data.iloc[-20:])
            _, current_value, _ = self.calculate_option_metrics(
                data['close'].iloc[-1], trade['buy_strike'], trade['sell_strike'], days_to_expiry, iv
            )
            current_value *= trade['size']
            pnl = self.close_position(
                data.index[-1], data['close'].iloc[-1], 'End of Backtest',
                trade['signal_type'], trade['entry_time'], trade['entry_stock_price'], trade['size'],
                trade['direction'], data.iloc[-20:], trade, equity
            )
            self.trades.append(trade)
            equity += current_value - trade['size'] * self.commission_per_contract
            logger.info("Equity updated after end of backtest: %.2f", equity)

        metrics = self.calculate_performance_metrics()
        logger.info("Performance Metrics: %s", metrics)

        rejection_summary = pd.Series([r['reason'] for r in self.rejections]).value_counts().to_dict()
        logger.info("Rejection Summary: %s", rejection_summary)

    def save_trades(self, filename):
        trades_df = pd.DataFrame(self.trades)
        trades_df.to_csv(filename, index=False)
        logger.info("Trades saved to %s", filename)
