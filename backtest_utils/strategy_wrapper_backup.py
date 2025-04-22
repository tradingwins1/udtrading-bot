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
    # ... [Previous methods unchanged: __init__, log_rejection, load_data, initialize_models, predict_trade_success, get_mtf_trend, detect_break_and_retest, detect_ug_signals, detect_gap_fill_reversal, detect_opening_range_signals, calculate_option_metrics, close_position, precompute_prophet_predictions, detect_pattern, detect_anomaly, detect_order_block]

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
        self.tsla_5min_data['vwap'] = vwap
        self.tsla_5min_data['vwap_distance'] = (self.tsla_5min_data['close'] - vwap) / vwap
        self.tsla_5min_data['support'] = self.tsla_5min_data['close'].rolling(window=5).min()
        self.tsla_5min_data['resistance'] = self.tsla_5min_data['close'].rolling(window=5).max()
        self.tsla_5min_data['rsi_slope'] = self.tsla_5min_data['RSI'].diff(5) / 5
        self.tsla_5min_data['atr_ratio'] = self.tsla_5min_data['ATR'] / self.tsla_5min_data['close']
        self.tsla_5min_data['price_vwap_slope'] = (self.tsla_5min_data['close'] - self.tsla_5min_data['vwap']) / 5
        self.tsla_5min_data['volatility_spread'] = self.tsla_5min_data['ATR'] / self.tsla_5min_data['ATR_SMA20']

        prophet_movements = self.precompute_prophet_predictions(self.tsla_5min_data)
        self.tsla_5min_data['prophet_movement'] = prophet_movements

        signals = pd.DataFrame(index=self.tsla_5min_data.index)
        bos_signals = self.detect_ug_signals(self.tsla_5min_data)
        gap_signals = self.detect_gap_fill_reversal(self.tsla_5min_data)
        orb_signals = self.detect_opening_range_signals(self.tsla_5min_data)
        break_retest_signals = self.detect_break_and_retest(self.tsla_5min_data)
        orb_signals_1min = self.detect_opening_range_signals(self.tsla_1min_data, '1min')
        ob_signals = self.detect_order_block(self.tsla_5min_data)

        signals['bos_long'] = bos_signals['bos_long']
        signals['bos_short'] = bos_signals['bos_short']
        signals['gap_fill_long'] = gap_signals['gap_fill_long']
        signals['gap_fill_short'] = gap_signals['gap_fill_short']
        signals['orb_long'] = orb_signals['orb_long']
        signals['orb_short'] = orb_signals['orb_short']
        signals['break_and_retest_long'] = break_retest_signals['break_and_retest_long']
        signals['break_and_retest_short'] = break_retest_signals['break_and_retest_short']
        signals['orb_long_1min'] = orb_signals_1min['orb_long'].reindex(self.tsla_5min_data.index, method='ffill')
        signals['orb_short_1min'] = orb_signals_1min['orb_short'].reindex(self.tsla_5min_data.index, method='ffill')
        signals['ob_long'] = ob_signals['ob_long']
        signals['ob_short'] = ob_signals['ob_short']

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

            if daily_trades >= 50:
                logger.debug("Skipping bar %d: Daily trade limit reached (%d trades)", i, daily_trades)
                self.log_rejection(timestamp, "Daily trade limit reached")
                continue
            if daily_pnl <= -self.initial_capital * 0.10:
                logger.debug("Skipping bar %d: Daily PNL limit reached (%.2f)", i, daily_pnl)
                self.log_rejection(timestamp, "Daily PNL limit reached")
                continue
            if daily_losers >= 50:
                logger.debug("Skipping bar %d: Daily loser limit reached (%d losers)", i, daily_losers)
                self.log_rejection(timestamp, "Daily loser limit reached")
                continue

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
                _, current_value, current_delta = self.calculate_option_metrics(current['close'], buy_strike, sell_strike, days_to_expiry, iv=iv_entry)
                current_iv = iv_entry

                profit_potential = target_value - entry_value
                if current_value >= entry_value + 0.02 * profit_potential:
                    trailing_stop = current_value - (trade['atr_entry'] * 0.03)
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

                if current_iv >= iv_entry * 1.2:
                    self.close_position(timestamp, current['close'], "IV Spike Exit", signal_type, entry_time, entry_stock_price, size, direction, data_slice, trade)
                    trade['closed'] = True
                    daily_pnl += trade['pnl']
                    equity += trade['exit_value']
                    logger.info("Equity updated after IV spike exit: %.2f", equity)
                    if trade['result'] == 'loss':
                        daily_losers += 1
                    continue

                if current_delta < 0.2 or current_delta > 0.8:
                    self.close_position(timestamp, current['close'], "Delta Exit", signal_type, entry_time, entry_stock_price, size, direction, data_slice, trade)
                    trade['closed'] = True
                    daily_pnl += trade['pnl']
                    equity += trade['exit_value']
                    logger.info("Equity updated after delta exit: %.2f", equity)
                    if trade['result'] == 'loss':
                        daily_losers += 1
                    continue

                holding_period = (timestamp - entry_time).total_seconds() / 3600
                if holding_period >= 6 and (current_value - entry_value) / entry_value < -0.005:
                    self.close_position(timestamp, current['close'], "Loss Limit Exceeded", signal_type, entry_time, entry_stock_price, size, direction, data_slice, trade)
                    trade['closed'] = True
                    daily_pnl += trade['pnl']
                    equity += trade['exit_value']
                    logger.info("Equity updated after loss limit: %.2f", equity)
                    if trade['result'] == 'loss':
                        daily_losers += 1
                    continue

                if holding_period >= 24:
                    self.close_position(timestamp, current['close'], "Timeout", signal_type, entry_time, entry_stock_price, size, direction, data_slice, trade)
                    trade['closed'] = True
                    daily_pnl += trade['pnl']
                    equity += trade['exit_value']
                    logger.info("Equity updated after timeout: %.2f", equity)
                    if trade['result'] == 'loss':
                        daily_losers += 1
                    continue

            self.trades = [trade for trade in self.trades if not trade.get('closed')]

            time_window = timedelta(minutes=10)
            matching_signals = signals[(signals.index >= (timestamp - time_window)) & (signals.index <= (timestamp + time_window))]
            if matching_signals.empty:
                logger.debug("No matching signals for bar %d, timestamp=%s", i, timestamp)
                self.log_rejection(timestamp, "No matching signals")
                continue
            logger.debug("Found %d matching signals for bar %d, timestamp=%s", len(matching_signals), i, timestamp)

            signal_types = matching_signals[['bos_long', 'bos_short', 'gap_fill_long', 'gap_fill_short', 'orb_long', 'orb_short', 'break_and_retest_long', 'break_and_retest_short', 'orb_long_1min', 'orb_short_1min', 'ob_long', 'ob_short']].any()
            for signal_type in signal_types.index:
                if not signal_types[signal_type]:
                    continue
                long_signal = signal_type.endswith('_long') or signal_type == 'orb_long_1min' or signal_type == 'ob_long'
                short_signal = signal_type.endswith('_short') or signal_type == 'orb_short_1min' or signal_type == 'ob_short'
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
                    trend_bearish = current['trend_bearish']
                    if trend_bearish or current['vwap_distance'] < 0:
                        direction = 'short' if short_prob > 0.6 else 'long'
                        lgb_prob = short_prob if direction == 'short' else long_prob
                    elif abs(long_prob - short_prob) < 0.05:
                        direction = 'long' if current['trend_bullish'] else 'short'
                        lgb_prob = long_prob if direction == 'long' else short_prob
                    else:
                        direction = 'long' if long_prob > short_prob else 'short'
                        lgb_prob = long_prob if direction == 'long' else short_prob
                    signal_type_full = f"{signal_name} ({'Long' if direction == 'long' else 'Short'})"

                    for trade in self.trades:
                        if trade['signal_type'] == signal_type_full and trade['direction'] == direction and (timestamp - trade['entry_time']).total_seconds() < 600:
                            logger.debug("Skipping trade at bar %d: Duplicate signal %s already open", i, signal_type_full)
                            self.log_rejection(timestamp, "Duplicate signal already open")
                            break
                    else:
                        rsi = current['RSI']
                        adx = current['ADX']
                        strong_trend = adx > 25
                        if strong_trend:
                            momentum_up = rsi > 35
                            momentum_down = rsi < 65
                        else:
                            momentum_up = True
                            momentum_down = True
                        logger.debug("Momentum check: RSI=%.2f, ADX=%.2f, strong_trend=%s, momentum_up=%s, momentum_down=%s",
                                     rsi, adx, strong_trend, momentum_up, momentum_down)

                        if direction == 'long' and not momentum_up:
                            logger.debug("Skipping trade at bar %d: RSI (%.2f) not above threshold", i, rsi)
                            self.log_rejection(timestamp, "RSI not above threshold")
                            continue
                        if direction == 'short' and not momentum_down:
                            logger.debug("Skipping trade at bar %d: RSI (%.2f) not below threshold", i, rsi)
                            self.log_rejection(timestamp, "RSI not below threshold")
                            continue

                        qqq_aligned = True
                        is_trend_day = current['trend_bullish'] or current['trend_bearish']
                        high_volatility = current['ATR'] > current['ATR_SMA20']
                        if not is_trend_day:
                            qqq_slice = self.qqq_5min_data.loc[timestamp - timedelta(minutes=1000):timestamp]
                            logger.debug("QQQ slice size for bar %d: %d bars", i, len(qqq_slice))
                            if len(qqq_slice) < 200:
                                logger.debug("Skipping trade at bar %d: Insufficient QQQ data for EMAs (%d bars < 200)", i, len(qqq_slice))
                                self.log_rejection(timestamp, "Insufficient QQQ data for EMAs")
                                continue
                            qqq_ema50 = ta.trend.EMAIndicator(qqq_slice['Close'], window=50).ema_indicator().iloc[-1]
                            qqq_ema200 = ta.trend.EMAIndicator(qqq_slice['Close'], window=200).ema_indicator().iloc[-1]
                            if not pd.isna(qqq_ema50) and not pd.isna(qqq_ema200):
                                if high_volatility:
                                    qqq_aligned = (direction == 'long' and qqq_ema50 > qqq_ema200 * 0.99) or (direction == 'short' and qqq_ema50 < qqq_ema200 * 1.01)
                                else:
                                    qqq_aligned = (direction == 'long' and qqq_ema50 > qqq_ema200) or (direction == 'short' and qqq_ema50 < qqq_ema200)
                            else:
                                logger.debug("QQQ EMAs are NaN at bar %d (EMA50=%s, EMA200=%s), skipping trade", i,
                                             'nan' if pd.isna(qqq_ema50) else f"{qqq_ema50:.2f}",
                                             'nan' if pd.isna(qqq_ema200) else f"{qqq_ema200:.2f}")
                                self.log_rejection(timestamp, "QQQ EMAs are NaN")
                                continue
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

                        wave_confirmation = True  # Manual check from MotiveWave (wave 3 or C)
                        if not wave_confirmation:
                            logger.debug("Skipping trade at bar %d: No Elliott Wave confirmation", i)
                            self.log_rejection(timestamp, "No Elliott Wave confirmation")
                            continue

                        if lgb_prob >= 0.2: threshold_counts[0.2] += 1
                        if lgb_prob >= 0.4: threshold_counts[0.4] += 1
                        if lgb_prob >= 0.6: threshold_counts[0.6] += 1
                        if lgb_prob >= 0.65: threshold_counts[0.65] += 1

                        # Preliminary strike and risk calculations
                        stock_price = current['close']
                        buy_strike = stock_price + 1
                        sell_strike = stock_price + 5
                        days_to_expiry = 7
                        iv = 0.4
                        spread_delta, net_cost, delta = self.calculate_option_metrics(stock_price, buy_strike, sell_strike, days_to_expiry, iv)
                        size = 3  # Fixed 3 contracts
                        net_cost_total = net_cost * size
                        bar_atr = current['ATR']
                        volatility_factor = min(2.0, max(0.5, bar_atr / current['ATR_SMA20']))
                        sl_atr_mult = 3.0 * volatility_factor
                        tp_atr_mult = 4.0 * volatility_factor
                        max_loss = bar_atr * sl_atr_mult * 100 * size
                        max_profit = ((sell_strike - buy_strike) * 100 - net_cost) * size
                        rr_ratio = max_profit / max_loss if max_loss != 0 else 1.0

                        lgbm_threshold = 0.60 if rr_ratio >= 3 else 0.65
                        if lgb_prob < lgbm_threshold:
                            logger.debug("Skipping trade at bar %d: LightGBM probability %.2f below %.2f", i, lgb_prob, lgbm_threshold)
                            self.log_rejection(timestamp, f"LightGBM probability below {lgbm_threshold}", lgbm_score=lgb_prob)
                            continue

                        confidence_score = lgb_prob * 10
                        if confidence_score < 3.0:
                            logger.debug("Skipping trade at bar %d: Confidence score %.2f below 3.0 (lgb_prob=%.2f)", i, confidence_score, lgb_prob)
                            self.log_rejection(timestamp, "Confidence score below 3.0", lgbm_score=lgb_prob)
                            continue

                        # Refine strike selection
                        if net_cost_total * 0.2 > 100:
                            logger.debug("Adjusting strikes to meet risk limit")
                            buy_strike = stock_price + 0.3 * bar_atr
                            sell_strike = buy_strike + 1.5
                            spread_delta, net_cost, delta = self.calculate_option_metrics(stock_price, buy_strike, sell_strike, days_to_expiry, iv)
                            net_cost_total = net_cost * size
                        max_loss = min(net_cost_total * 0.2, 100)

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

                        entry_value = net_cost * size
                        target_value = entry_value + (bar_atr * tp_atr_mult * 100 * size)

                        logger.debug("Profit calculation: bar=%d, net_cost=%.2f, size=%d, entry_value=%.2f, bar_atr=%.2f, tp_atr_mult=%.2f, target_value=%.2f",
                                     i, net_cost, size, entry_value, bar_atr, tp_atr_mult, target_value)

                        if net_cost_total > equity:
                            logger.debug("Skipping trade at bar %d: Insufficient equity (%.2f required, %.2f available)", i, net_cost_total, equity)
                            self.log_rejection(timestamp, "Insufficient equity")
                            continue

                        if net_cost_total > 0.15 * equity:
                            logger.debug("Skipping trade at bar %d: Capital exposure exceeds 15%% (%.2f required, %.2f allowed)", i, net_cost_total, 0.15 * equity)
                            self.log_rejection(timestamp, "Capital exposure exceeds 15%")
                            continue

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
                        rr_ratio = max_profit / max_loss if max_loss != 0 else 1.0

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
                                'rsi_slope': current['rsi_slope'],
                                'atr_ratio': current['atr_ratio'],
                                'price_vwap_slope': current['price_vwap_slope'],
                                'volatility_spread': current['volatility_spread'],
                                'implied_volatility': iv,
                                'option_delta': delta,
                                'option_theta': -0.05,
                                'option_vega': 0.1,
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

        logger.info("Threshold analysis - trades passing thresholds: 0.2: %d, 0.4: %d, 0.6: %d, 0.65: %d",
                    threshold_counts[0.2], threshold_counts[0.4], threshold_counts[0.6], threshold_counts[0.65])

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