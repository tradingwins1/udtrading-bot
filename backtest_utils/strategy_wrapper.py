import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
from bos_logic import detect_ug_signals
from gap_logic import detect_gap_fill_reversal
from learn import update_trade_log
from scorer import score_setup
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange
from ta.trend import SMAIndicator
import logging
import time
from datetime import timedelta, datetime

logger = logging.getLogger(__name__)

class UGBacktestStrategy:
    def __init__(self, tsla_5min_data, qqq_5min_data, tsla_1min_data, qqq_1min_data, initial_capital=10000, commission=0.001, slippage=0.0005, max_holding_bars=120):
        self.tsla_5min_data = tsla_5min_data.copy()
        self.qqq_5min_data = qqq_5min_data.copy()
        self.tsla_1min_data = tsla_1min_data.copy()
        self.qqq_1min_data = qqq_1min_data.copy()
        self.initial_capital = float(initial_capital)
        self.equity = self.initial_capital
        self.positions = []
        self.trades = []
        self.commission = commission
        self.slippage = slippage
        self.max_holding_bars = max_holding_bars
        self.daily_trades = {}
        self.daily_pnl = {}
        self.daily_losing_trades = {}
        self.results = pd.DataFrame(index=self.tsla_5min_data.index, columns=['Equity'], dtype='float64')
        self.results['Equity'] = self.initial_capital
        logger.debug("Strategy initialized with initial_capital=%s, 5min_data_shape=%s, 1min_data_shape=%s", 
                     initial_capital, self.tsla_5min_data.shape, self.tsla_1min_data.shape)

        try:
            self.tsla_5min_data['RSI'] = RSIIndicator(self.tsla_5min_data['Close'], window=14).rsi()
            self.tsla_5min_data['ATR'] = AverageTrueRange(self.tsla_5min_data['High'], self.tsla_5min_data['Low'], self.tsla_5min_data['Close'], window=14).average_true_range()
            self.tsla_5min_data['SMA200'] = SMAIndicator(self.tsla_5min_data['Close'], window=200).sma_indicator()
            self.tsla_5min_data['time'] = self.tsla_5min_data.index.time
            self.tsla_5min_data['is_trading_window'] = (self.tsla_5min_data['time'] >= pd.Timestamp('09:45').time()) & (self.tsla_5min_data['time'] <= pd.Timestamp('15:45').time())
            logger.debug("Indicators and trading window calculated successfully for TSLA 5min data")
        except Exception as e:
            logger.error("Error calculating indicators for TSLA 5min data: %s", e)
            raise

        # HTF Analysis for TSLA and QQQ
        self.htf_bias = self.calculate_htf_bias()
        self.qqq_trend = self.calculate_qqq_trend()
        self.max_daily_points = self.calculate_max_daily_points()
        logger.debug("HTF bias: %s, QQQ trend: %s, Max daily points for TSLA: %.2f", self.htf_bias, self.qqq_trend, self.max_daily_points)

    def calculate_htf_bias(self):
        """Calculate HTF bias using Daily, 4h, and 1h SMA200 trends for TSLA."""
        daily_data = self.tsla_5min_data.resample('D').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
        h4_data = self.tsla_5min_data.resample('4h').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
        h1_data = self.tsla_5min_data.resample('1h').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()

        daily_data['SMA200'] = SMAIndicator(daily_data['Close'], window=200).sma_indicator()
        h4_data['SMA200'] = SMAIndicator(h4_data['Close'], window=200).sma_indicator()
        h1_data['SMA200'] = SMAIndicator(h1_data['Close'], window=200).sma_indicator()

        daily_trend = 'up' if daily_data['Close'].iloc[-1] > daily_data['SMA200'].iloc[-1] else 'down'
        h4_trend = 'up' if h4_data['Close'].iloc[-1] > h4_data['SMA200'].iloc[-1] else 'down'
        h1_trend = 'up' if h1_data['Close'].iloc[-1] > h1_data['SMA200'].iloc[-1] else 'down'

        if daily_trend == h4_trend == h1_trend == 'up':
            return 'uptrend'
        elif daily_trend == h4_trend == h1_trend == 'down':
            return 'downtrend'
        else:
            return 'neutral'

    def calculate_qqq_trend(self):
        """Calculate QQQ trend using 1h SMA200 for confirmation."""
        h1_data = self.qqq_5min_data.resample('1h').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
        h1_data['SMA200'] = SMAIndicator(h1_data['Close'], window=200).sma_indicator()
        qqq_trend = 'up' if h1_data['Close'].iloc[-1] > h1_data['SMA200'].iloc[-1] else 'down'
        return qqq_trend

    def calculate_max_daily_points(self):
        """Calculate max daily points for TSLA (average daily range over 30 days)."""
        daily_data = self.tsla_5min_data.resample('D').agg({'High': 'max', 'Low': 'min'}).dropna()
        daily_data['Range'] = daily_data['High'] - daily_data['Low']
        max_points = daily_data['Range'].tail(30).mean()
        return max_points if not pd.isna(max_points) else 25.0  # Default to 25 points

    def calculate_position_size(self, entry_price, stop_loss, atr, confidence_score):
        """Calculate position size based on risk (1% or 2% of capital)."""
        try:
            risk_percent = 0.02 if confidence_score > 7 and self.htf_bias in ['uptrend', 'downtrend'] and self.htf_bias == self.qqq_trend else 0.01
            risk_amount = self.equity * risk_percent
            risk_per_share = max(abs(entry_price - stop_loss), atr)
            if risk_per_share < 0.01:  # Avoid division by near-zero
                logger.debug("Risk per share too small: %.4f, using default size", risk_per_share)
                return 10
            size = int(risk_amount / risk_per_share)
            size = min(size, 50)
            size = max(size, 10)
            logger.debug("Calculated position size: %s for entry_price=%.2f, stop_loss=%.2f, atr=%.2f, risk=%.2f%%", 
                         size, entry_price, stop_loss, atr, risk_percent * 100)
            return size
        except Exception as e:
            logger.error("Error calculating position size: %s", e)
            return 10

    def calculate_key_levels(self):
        """Calculate PML, PMH, PDL, PDH, and KPL for each trading day."""
        key_levels = {}
        daily_groups = self.tsla_5min_data.groupby(self.tsla_5min_data.index.date)
        for day, day_data in daily_groups:
            # Pre-market session (4:00 AM to 9:30 AM EST)
            pre_market = day_data.between_time('04:00', '09:30')
            pml = pre_market['Low'].min() if not pre_market.empty else np.nan
            pmh = pre_market['High'].max() if not pre_market.empty else np.nan

            # Previous day's levels
            prev_day = self.tsla_5min_data[self.tsla_5min_data.index.date < day].tail(78)  # Approx 1 trading day (6.5 hours)
            pdl = prev_day['Low'].min() if not prev_day.empty else np.nan
            pdh = prev_day['High'].max() if not prev_day.empty else np.nan

            key_levels[day] = {'PML': pml, 'PMH': pmh, 'PDL': pdl, 'PDH': pdh}
        return key_levels

    def run(self):
        logger.debug("Starting strategy run...")
        try:
            # Calculate key levels
            self.key_levels = self.calculate_key_levels()
            logger.debug("Key levels: %s", self.key_levels)

            # Generate signals
            bos_signals = detect_ug_signals(self.tsla_5min_data, self.tsla_1min_data, self.key_levels)
            gap_signals = detect_gap_fill_reversal(self.tsla_5min_data)

            if not bos_signals.empty:
                bos_duplicates = bos_signals.index[bos_signals.index.duplicated()].tolist()
                if bos_duplicates:
                    logger.warning("Duplicates in bos_signals: %s", bos_duplicates)
                bos_signals = bos_signals[~bos_signals.index.duplicated(keep='first')]
            if not gap_signals.empty:
                gap_duplicates = gap_signals.index[gap_signals.index.duplicated()].tolist()
                if gap_duplicates:
                    logger.warning("Duplicates in gap_signals: %s", gap_duplicates)
                gap_signals = gap_signals[~gap_signals.index.duplicated(keep='first')]

            logger.info("BOS signals generated: %d, timestamps: %s", len(bos_signals), bos_signals.index.tolist()[:5])
            logger.info("Gap signals generated: %d, timestamps: %s", len(gap_signals), gap_signals.index.tolist()[:5])

            if not bos_signals.empty and not gap_signals.empty:
                signals = pd.concat([bos_signals, gap_signals])
                signals = signals.reset_index().groupby('timestamp').first().reset_index()
                signals['Confluences'] = signals['timestamp'].apply(
                    lambda x: {k: v for signals_df in [bos_signals, gap_signals]
                               for k, v in (signals_df.loc[signals_df.index == x, 'Confluences'].iloc[0].items() if x in signals_df.index else {})}
                )
                signals = signals.set_index('timestamp')
            elif not bos_signals.empty:
                signals = bos_signals
            elif not gap_signals.empty:
                signals = gap_signals
            else:
                signals = pd.DataFrame()

            if not signals.empty:
                duplicates = signals.index[signals.index.duplicated()].tolist()
                if duplicates:
                    logger.error("Duplicates after merge: %s", duplicates)
                    signals = signals[~signals.index.duplicated(keep='first')]
            logger.info("Total signals after merging: %d, timestamps: %s", len(signals), signals.index.tolist()[:5])

            if signals.empty:
                logger.warning("No signals generated. Skipping signal processing.")
                return self.results, pd.DataFrame(self.trades)

            signal_timestamps = signals.index.tolist()
            signal_types = signals[['Type', 'Direction', 'Confluences', 'PDH', 'PDL', 'PMH', 'PML', 'BOS_Level']].to_dict('index')
            logger.debug("Signal timestamps: %s", signal_timestamps[:5])
            logger.debug("Data timestamps: %s", self.tsla_5min_data.index[:5].tolist())

            total_bars = len(self.tsla_5min_data)
            start_time = time.time()
            current_day = None

            for i in range(1, total_bars):
                if i % 1000 == 0:
                    progress = (i / total_bars) * 100
                    elapsed_time = time.time() - start_time
                    eta = (elapsed_time / i) * (total_bars - i) if i > 0 else 0
                    logger.info("Processing bar %d/%d (%.2f%%), ETA: %.2f seconds", i, total_bars, progress, eta)

                current = self.tsla_5min_data.iloc[i]
                timestamp = current.name
                day = timestamp.date()

                if not current['is_trading_window']:
                    logger.debug("Skipping bar %d: Outside trading window (9:45 AM - 3:45 PM EST), timestamp=%s", i, timestamp)
                    continue

                if current_day != day:
                    current_day = day
                    self.daily_trades[day] = 0
                    self.daily_pnl[day] = 0
                    self.daily_losing_trades[day] = 0
                    logger.debug("New day %s: Resetting trade counter and P&L", day)

                if self.daily_trades.get(day, 0) >= 3 or self.daily_pnl.get(day, 0) <= -500 or self.daily_losing_trades.get(day, 0) >= 3:
                    logger.debug("Daily limits reached for %s at bar %d: trades=%d, P&L=%.2f, losing trades=%d", 
                                 day, i, self.daily_trades.get(day, 0), self.daily_pnl.get(day, 0), self.daily_losing_trades.get(day, 0))
                    continue

                atr = current['ATR']
                avg_atr = self.tsla_5min_data['ATR'].iloc[max(0, i-50):i].mean()
                if atr > 7 * avg_atr:
                    logger.debug("Skipping bar %d due to high volatility: atr=%.2f, avg_atr=%.2f", i, atr, avg_atr)
                    continue

                time_window = timedelta(minutes=10)
                matching_signals = [ts for ts in signal_timestamps if abs((ts - timestamp).total_seconds()) <= time_window.total_seconds()]
                if matching_signals:
                    closest_ts = min(matching_signals, key=lambda ts: abs((ts - timestamp).total_seconds()))
                    signal_info = signal_types[closest_ts]
                    signal_type = signal_info['Type']
                    direction = signal_info['Direction']
                    confluences = signal_info['Confluences']
                    pdh = signal_info['PDH']
                    pdl = signal_info['PDL']
                    pmh = signal_info['PMH']
                    pml = signal_info['PML']
                    bos_level = signal_info['BOS_Level']
                    atr = current['ATR']
                    size = 0
                    pos_type = None
                    stop_loss = 0
                    take_profit1 = 0
                    take_profit2 = 0

                    confidence_score = score_setup(self.tsla_5min_data.iloc[max(0, i-50):i+1], direction=direction)
                    logger.debug("Signal at bar %d, timestamp=%s: type=%s, direction=%s, confidence=%.2f, htf_bias=%s, qqq_trend=%s", 
                                 i, closest_ts, signal_type, direction, confidence_score, self.htf_bias, self.qqq_trend)

                    try:
                        if direction == 'long' and (self.htf_bias == 'uptrend' or self.htf_bias == 'neutral') and (self.qqq_trend == 'up' or self.htf_bias == 'neutral'):
                            stop_loss = min([pml, pdl, bos_level], default=current['Close'] - 1.0 * atr)
                            stop_loss = max(stop_loss, current['Close'] * 0.95)
                            size = self.calculate_position_size(current['Close'], stop_loss, atr, confidence_score)
                            pos_type = 'long'
                            risk = current['Close'] - stop_loss
                            take_profit1 = current['Close'] + risk  # 1:1 R:R for TP1
                            kpl = round(current['Close'] / 0.5) * 0.5
                            take_profit2 = min(kpl - 0.5, current['Close'] + self.max_daily_points)
                            if pmh and pmh < take_profit2:
                                take_profit2 = pmh - 0.5
                            if pdh and pdh < take_profit2:
                                take_profit2 = pdh - 0.5
                            take_profit2 = max(take_profit2, take_profit1)  # Ensure TP2 > TP1
                        elif direction == 'short' and self.htf_bias == 'downtrend' and self.qqq_trend == 'down':
                            stop_loss = max([pmh, pdh, bos_level], default=current['Close'] + 1.0 * atr)
                            stop_loss = min(stop_loss, current['Close'] * 1.05)
                            size = self.calculate_position_size(current['Close'], stop_loss, atr, confidence_score)
                            pos_type = 'short'
                            risk = stop_loss - current['Close']
                            take_profit1 = current['Close'] - risk  # 1:1 R:R for TP1
                            kpl = round(current['Close'] / 0.5) * 0.5
                            take_profit2 = max(kpl + 0.5, current['Close'] - self.max_daily_points)
                            if pml and pml > take_profit2:
                                take_profit2 = pml + 0.5
                            if pdl and pdl > take_profit2:
                                take_profit2 = pdl + 0.5
                            take_profit2 = min(take_profit2, take_profit1)  # Ensure TP2 < TP1
                        else:
                            logger.debug("Signal rejected at bar %d: direction=%s does not match htf_bias=%s or qqq_trend=%s", 
                                         i, direction, self.htf_bias, self.qqq_trend)

                        if size > 0 and confidence_score >= 3.0:
                            entry_price = current['Close'] * (1 + self.slippage if pos_type == 'long' else 1 - self.slippage)
                            self.positions.append({
                                'entry_price': entry_price,
                                'entry_time': timestamp,
                                'type': pos_type,
                                'size': size,
                                'stop_loss': stop_loss,
                                'take_profit1': take_profit1,
                                'take_profit2': take_profit2,
                                'breakeven': False,
                                'rsi': current['RSI'],
                                'atr': atr,
                                'signal_type': signal_type,
                                'confidence_score': confidence_score,
                                'entry_bar': i,
                                'confluences': confluences,
                                'tp1_hit': False
                            })
                            self.daily_trades[day] = self.daily_trades.get(day, 0) + 1
                            logger.info("Opened position: bar=%d, type=%s, size=%d, signal=%s, confidence=%.2f, TP1=%.2f, TP2=%.2f", 
                                        i, pos_type, size, signal_type, confidence_score, take_profit1, take_profit2)
                        else:
                            logger.debug("No position opened at bar %d: size=%d, confidence_score=%.2f", i, size, confidence_score)
                    except Exception as e:
                        logger.error("Error opening position at bar %d: %s", i, e)
                        continue

                for pos in self.positions[:]:
                    try:
                        entry_price = pos['entry_price']
                        size = pos['size']
                        stop_loss = pos['stop_loss']
                        take_profit1 = pos['take_profit1']
                        take_profit2 = pos['take_profit2']
                        commission_cost = entry_price * size * self.commission

                        bars_held = i - pos['entry_bar']
                        if bars_held >= self.max_holding_bars:
                            exit_price = current['Close']
                            if pos['type'] == 'long':
                                exit_price *= (1 - self.slippage)
                            else:
                                exit_price *= (1 + self.slippage)
                            pnl = (exit_price - entry_price) * size - commission_cost if pos['type'] == 'long' else (entry_price - exit_price) * size - commission_cost
                            self.equity += pnl
                            self.daily_pnl[day] = self.daily_pnl.get(day, 0) + pnl
                            if pnl < 0:
                                self.daily_losing_trades[day] = self.daily_losing_trades.get(day, 0) + 1
                            trade_data = {
                                'entry_time': pos['entry_time'],
                                'exit_time': timestamp,
                                'entry_price': entry_price,
                                'exit_price': exit_price,
                                'pnl': pnl,
                                'type': pos['type'],
                                'size': size,
                                'rsi_entry': pos['rsi'],
                                'atr_entry': pos['atr'],
                                'signal_type': pos['signal_type'],
                                'confidence_score': pos['confidence_score'],
                                'holding_period': (timestamp - pos['entry_time']).total_seconds() / 3600,
                                'reason': 'Timeout',
                                'result': 'win' if pnl > 0 else 'loss',
                                'setup_type': 'scalp',
                                'rr_ratio': (exit_price - entry_price) / (entry_price - stop_loss) if pos['type'] == 'long' else (entry_price - exit_price) / (stop_loss - entry_price),
                                'confluences': pos['confluences'],
                                'entry_bar': pos['entry_bar'],
                                'exit_bar': i
                            }
                            self.trades.append(trade_data)
                            update_trade_log(trade_data, self.tsla_5min_data.iloc[max(0, i - 50):i + 1].copy())
                            logger.info("Trade closed: signal_type=%s, pnl=%.2f, holding_period=%.2f hours", trade_data['signal_type'], trade_data['pnl'], trade_data['holding_period'])
                            self.positions.remove(pos)
                            logger.debug("Closed position at bar %d due to timeout: pnl=%.2f", i, pnl)
                            continue

                        if pos['type'] == 'long':
                            if current['Close'] <= stop_loss:
                                exit_price = stop_loss * (1 - self.slippage)
                                pnl = (exit_price - entry_price) * size - commission_cost
                                self.equity += pnl
                                self.daily_pnl[day] = self.daily_pnl.get(day, 0) + pnl
                                if pnl < 0:
                                    self.daily_losing_trades[day] = self.daily_losing_trades.get(day, 0) + 1
                                trade_data = {
                                    'entry_time': pos['entry_time'],
                                    'exit_time': timestamp,
                                    'entry_price': entry_price,
                                    'exit_price': exit_price,
                                    'pnl': pnl,
                                    'type': pos['type'],
                                    'size': size,
                                    'rsi_entry': pos['rsi'],
                                    'atr_entry': pos['atr'],
                                    'signal_type': pos['signal_type'],
                                    'confidence_score': pos['confidence_score'],
                                    'holding_period': (timestamp - pos['entry_time']).total_seconds() / 3600,
                                    'reason': 'SL Hit',
                                    'result': 'win' if pnl > 0 else 'loss',
                                    'setup_type': 'scalp',
                                    'rr_ratio': (exit_price - entry_price) / (entry_price - stop_loss),
                                    'confluences': pos['confluences'],
                                    'entry_bar': pos['entry_bar'],
                                    'exit_bar': i
                                }
                                self.trades.append(trade_data)
                                update_trade_log(trade_data, self.tsla_5min_data.iloc[max(0, i - 50):i + 1].copy())
                                logger.info("Trade closed: signal_type=%s, pnl=%.2f, holding_period=%.2f hours", trade_data['signal_type'], trade_data['pnl'], trade_data['holding_period'])
                                self.positions.remove(pos)
                                logger.debug("Closed position at bar %d: pnl=%.2f, reason=%s", i, pnl, trade_data['reason'])
                            elif current['Close'] >= take_profit1 and not pos['tp1_hit']:
                                pos['tp1_hit'] = True
                                pos['stop_loss'] = entry_price  # Trailing stop to breakeven
                                logger.debug("TP1 hit at bar %d: Adjusted SL to breakeven %.2f", i, entry_price)
                            elif current['Close'] >= take_profit2 and pos['tp1_hit']:
                                exit_price = take_profit2 * (1 - self.slippage)
                                pnl = (exit_price - entry_price) * size - commission_cost
                                self.equity += pnl
                                self.daily_pnl[day] = self.daily_pnl.get(day, 0) + pnl
                                if pnl < 0:
                                    self.daily_losing_trades[day] = self.daily_losing_trades.get(day, 0) + 1
                                trade_data = {
                                    'entry_time': pos['entry_time'],
                                    'exit_time': timestamp,
                                    'entry_price': entry_price,
                                    'exit_price': exit_price,
                                    'pnl': pnl,
                                    'type': pos['type'],
                                    'size': size,
                                    'rsi_entry': pos['rsi'],
                                    'atr_entry': pos['atr'],
                                    'signal_type': pos['signal_type'],
                                    'confidence_score': pos['confidence_score'],
                                    'holding_period': (timestamp - pos['entry_time']).total_seconds() / 3600,
                                    'reason': 'TP2 Hit',
                                    'result': 'win' if pnl > 0 else 'loss',
                                    'setup_type': 'scalp',
                                    'rr_ratio': (exit_price - entry_price) / (entry_price - stop_loss),
                                    'confluences': pos['confluences'],
                                    'entry_bar': pos['entry_bar'],
                                    'exit_bar': i
                                }
                                self.trades.append(trade_data)
                                update_trade_log(trade_data, self.tsla_5min_data.iloc[max(0, i - 50):i + 1].copy())
                                logger.info("Trade closed: signal_type=%s, pnl=%.2f, holding_period=%.2f hours", trade_data['signal_type'], trade_data['pnl'], trade_data['holding_period'])
                                self.positions.remove(pos)
                                logger.debug("Closed position at bar %d: pnl=%.2f, reason=%s", i, pnl, trade_data['reason'])

                        elif pos['type'] == 'short':
                            if current['Close'] >= stop_loss:
                                exit_price = stop_loss * (1 + self.slippage)
                                pnl = (entry_price - exit_price) * size - commission_cost
                                self.equity += pnl
                                self.daily_pnl[day] = self.daily_pnl.get(day, 0) + pnl
                                if pnl < 0:
                                    self.daily_losing_trades[day] = self.daily_losing_trades.get(day, 0) + 1
                                trade_data = {
                                    'entry_time': pos['entry_time'],
                                    'exit_time': timestamp,
                                    'entry_price': entry_price,
                                    'exit_price': exit_price,
                                    'pnl': pnl,
                                    'type': pos['type'],
                                    'size': size,
                                    'rsi_entry': pos['rsi'],
                                    'atr_entry': pos['atr'],
                                    'signal_type': pos['signal_type'],
                                    'confidence_score': pos['confidence_score'],
                                    'holding_period': (timestamp - pos['entry_time']).total_seconds() / 3600,
                                    'reason': 'SL Hit',
                                    'result': 'win' if pnl > 0 else 'loss',
                                    'setup_type': 'scalp',
                                    'rr_ratio': (entry_price - exit_price) / (stop_loss - entry_price),
                                    'confluences': pos['confluences'],
                                    'entry_bar': pos['entry_bar'],
                                    'exit_bar': i
                                }
                                self.trades.append(trade_data)
                                update_trade_log(trade_data, self.tsla_5min_data.iloc[max(0, i - 50):i + 1].copy())
                                logger.info("Trade closed: signal_type=%s, pnl=%.2f, holding_period=%.2f hours", trade_data['signal_type'], trade_data['pnl'], trade_data['holding_period'])
                                self.positions.remove(pos)
                                logger.debug("Closed position at bar %d: pnl=%.2f, reason=%s", i, pnl, trade_data['reason'])
                            elif current['Close'] <= take_profit1 and not pos['tp1_hit']:
                                pos['tp1_hit'] = True
                                pos['stop_loss'] = entry_price  # Trailing stop to breakeven
                                logger.debug("TP1 hit at bar %d: Adjusted SL to breakeven %.2f", i, entry_price)
                            elif current['Close'] <= take_profit2 and pos['tp1_hit']:
                                exit_price = take_profit2 * (1 + self.slippage)
                                pnl = (entry_price - exit_price) * size - commission_cost
                                self.equity += pnl
                                self.daily_pnl[day] = self.daily_pnl.get(day, 0) + pnl
                                if pnl < 0:
                                    self.daily_losing_trades[day] = self.daily_losing_trades.get(day, 0) + 1
                                trade_data = {
                                    'entry_time': pos['entry_time'],
                                    'exit_time': timestamp,
                                    'entry_price': entry_price,
                                    'exit_price': exit_price,
                                    'pnl': pnl,
                                    'type': pos['type'],
                                    'size': size,
                                    'rsi_entry': pos['rsi'],
                                    'atr_entry': pos['atr'],
                                    'signal_type': pos['signal_type'],
                                    'confidence_score': pos['confidence_score'],
                                    'holding_period': (timestamp - pos['entry_time']).total_seconds() / 3600,
                                    'reason': 'TP2 Hit',
                                    'result': 'win' if pnl > 0 else 'loss',
                                    'setup_type': 'scalp',
                                    'rr_ratio': (entry_price - exit_price) / (stop_loss - entry_price),
                                    'confluences': pos['confluences'],
                                    'entry_bar': pos['entry_bar'],
                                    'exit_bar': i
                                }
                                self.trades.append(trade_data)
                                update_trade_log(trade_data, self.tsla_5min_data.iloc[max(0, i - 50):i + 1].copy())
                                logger.info("Trade closed: signal_type=%s, pnl=%.2f, holding_period=%.2f hours", trade_data['signal_type'], trade_data['pnl'], trade_data['holding_period'])
                                self.positions.remove(pos)
                                logger.debug("Closed position at bar %d: pnl=%.2f, reason=%s", i, pnl, trade_data['reason'])

                    except Exception as e:
                        logger.error("Error processing position at bar %d: %s", i, e)
                        continue

                self.results.loc[timestamp, 'Equity'] = self.equity
                if i % 1000 == 0:
                    logger.debug("Updated equity at bar %d: %.2f", i, self.equity)

            logger.info("Strategy run completed: %d trades executed", len(self.trades))
            return self.results, pd.DataFrame(self.trades)

        except Exception as e:
            logger.error("Error in strategy run: %s", e)
            raise