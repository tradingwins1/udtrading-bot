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
from ta.volatility import AverageTrueRange
from ta.trend import SMAIndicator
import logging
import time
from datetime import timedelta

logger = logging.getLogger(__name__)

class UGBacktestStrategy:
    def __init__(self, data, initial_capital=10000, commission=0.001, slippage=0.0005, max_holding_bars=6):
        self.data = data.copy()
        self.initial_capital = float(initial_capital)
        self.equity = self.initial_capital
        self.positions = []
        self.trades = []
        self.commission = commission
        self.slippage = slippage
        self.max_holding_bars = max_holding_bars
        self.daily_trades = {}
        self.results = pd.DataFrame(index=self.data.index, columns=['Equity'], dtype='float64')
        self.results['Equity'] = self.initial_capital
        logger.debug("Strategy initialized with initial_capital=%s, data_shape=%s", initial_capital, data.shape)

        try:
            self.data['RSI'] = RSIIndicator(self.data['Close'], window=14).rsi()
            self.data['ATR'] = AverageTrueRange(self.data['High'], self.data['Low'], self.data['Close'], window=14).average_true_range()
            self.data['SMA200'] = SMAIndicator(self.data['Close'], window=200).sma_indicator()
            logger.debug("Indicators calculated successfully")
        except Exception as e:
            logger.error("Error calculating indicators: %s", e)
            raise

    def calculate_position_size(self, entry_price, stop_loss, atr):
        try:
            risk_amount = 100  # 1% of $10,000
            risk_per_share = max(abs(entry_price - stop_loss), atr)  # Use ATR to adjust risk
            size = int(risk_amount / risk_per_share) if risk_per_share != 0 else 10  # Minimum size of 10
            size = max(size, 10)  # Ensure minimum position size
            logger.debug("Calculated position size: %s for entry_price=%s, stop_loss=%s, atr=%s", size, entry_price, stop_loss, atr)
            return size
        except Exception as e:
            logger.error("Error calculating position size: %s", e)
            return 10  # Fallback to minimum size

    def run(self):
        logger.debug("Starting strategy run...")
        try:
            # Generate signals
            bos_signals = detect_ug_signals(self.data)
            gap_signals = detect_gap_fill_reversal(self.data)

            # Check for duplicates within each signal set
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

            logger.info("BOS signals generated: %s, timestamps: %s", len(bos_signals), bos_signals.index.tolist()[:5])
            logger.info("Gap signals generated: %s, timestamps: %s", len(gap_signals), gap_signals.index.tolist()[:5])

            # Merge signals
            if not bos_signals.empty and not gap_signals.empty:
                signals = pd.concat([bos_signals, gap_signals])
                # Deduplicate by keeping first signal (BOS priority) and merging confluences
                signals = signals.reset_index()
                signals = signals.groupby('timestamp').first().reset_index()
                signals['Confluences'] = signals['timestamp'].apply(
                    lambda x: {k: v for signals_df in [bos_signals, gap_signals]
                               for k, v in (signals_df.loc[signals_df.index == x, 'Confluences'].iloc[0].items()
                                            if x in signals_df.index else {})}
                )
                signals = signals.set_index('timestamp')
            elif not bos_signals.empty:
                signals = bos_signals
            elif not gap_signals.empty:
                signals = gap_signals
            else:
                signals = pd.DataFrame()

            # Final duplicate check
            if not signals.empty:
                duplicates = signals.index[signals.index.duplicated()].tolist()
                if duplicates:
                    logger.error("Duplicates after merge: %s", duplicates)
                    signals = signals[~signals.index.duplicated(keep='first')]
            logger.info("Total signals after merging: %s, timestamps: %s", len(signals), signals.index.tolist()[:5])

            if signals.empty:
                logger.warning("No signals generated. Skipping signal processing.")
                return self.results, pd.DataFrame(self.trades)

            signal_timestamps = signals.index.tolist()
            signal_types = signals[['Type', 'Direction', 'Confluences', 'PDH', 'PDL', 'PMH', 'PML', 'BOS_Level']].to_dict('index')
            logger.debug("Signal timestamps: %s", signal_timestamps[:5])
            logger.debug("Data timestamps: %s", self.data.index[:5].tolist())

            total_bars = len(self.data)
            start_time = time.time()
            current_day = None
            daily_risk = 0

            for i in range(1, total_bars):
                if i % 1000 == 0:
                    progress = (i / total_bars) * 100
                    elapsed_time = time.time() - start_time
                    eta = (elapsed_time / i) * (total_bars - i) if i > 0 else 0
                    logger.info("Processing bar %s/%s (%.2f%%), ETA: %.2f seconds", i, total_bars, progress, eta)
                
                current = self.data.iloc[i]
                timestamp = current.name
                day = timestamp.date()

                # Reset daily trade counter and risk at the start of a new day
                if current_day != day:
                    current_day = day
                    self.daily_trades[day] = 0
                    daily_risk = 0
                    logger.debug("New day %s: Resetting trade counter and risk", day)

                # Daily trade limit (max 3 trades or $500 risk)
                if self.daily_trades.get(day, 0) >= 3 or daily_risk >= 500:
                    logger.debug("Daily trade/risk limit reached for %s at bar %s: trades=%s, risk=%s", 
                                 day, i, self.daily_trades.get(day, 0), daily_risk)
                    continue

                # Fuzzy matching: check for signals within ±10 minutes
                time_window = timedelta(minutes=10)
                matching_signals = [
                    ts for ts in signal_timestamps
                    if abs((ts - timestamp).total_seconds()) <= time_window.total_seconds()
                ]
                if matching_signals:
                    # Use the closest signal timestamp
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
                    tp1 = tp2 = 0

                    # Calculate confidence score dynamically
                    confidence_score = score_setup(self.data.iloc[max(0, i-50):i+1])
                    logger.debug("Calculated confidence score at bar %s: %s", i, confidence_score)

                    try:
                        if direction == 'long':
                            stop_loss = current['Close'] - 2 * atr
                            stop_loss = max(stop_loss, current['Close'] * 0.95)
                            stop_loss = min(stop_loss, bos_level, pdl) if not pd.isna(pdl) else min(stop_loss, bos_level)
                            size = self.calculate_position_size(current['Close'], stop_loss, atr)
                            pos_type = 'long'
                            risk = current['Close'] - stop_loss
                            tp1 = current['Close'] + 1.5 * risk  # 1.5:1 R:R
                            tp2 = current['Close'] + 2 * risk    # 2:1 R:R
                            if not pd.isna(pdh) and pdh < tp2:
                                tp2 = pdh
                            if not pd.isna(pmh) and pmh < tp2:
                                tp2 = pmh
                            if not pd.isna(pdh) and pdh < tp1:
                                tp1 = pdh
                        elif direction == 'short':
                            stop_loss = current['Close'] + 2 * atr
                            stop_loss = min(stop_loss, current['Close'] * 1.05)
                            stop_loss = max(stop_loss, bos_level, pdh) if not pd.isna(pdh) else max(stop_loss, bos_level)
                            size = self.calculate_position_size(current['Close'], stop_loss, atr)
                            pos_type = 'short'
                            risk = stop_loss - current['Close']
                            tp1 = current['Close'] - 1.5 * risk  # 1.5:1 R:R
                            tp2 = current['Close'] - 2 * risk    # 2:1 R:R
                            if not pd.isna(pdl) and pdl > tp2:
                                tp2 = pdl
                            if not pd.isna(pml) and pml > tp2:
                                tp2 = pml
                            if not pd.isna(pdl) and pdl > tp1:
                                tp1 = pdl

                        if size > 0 and confidence_score >= 3.0:  # Adjusted threshold
                            entry_price = current['Close'] * (1 + self.slippage if pos_type == 'long' else 1 - self.slippage)
                            self.positions.append({
                                'entry_price': entry_price,
                                'entry_time': timestamp,
                                'type': pos_type,
                                'size': size,
                                'stop_loss': stop_loss,
                                'tp1': tp1,
                                'tp2': tp2,
                                'breakeven': False,
                                'rsi': current['RSI'],
                                'atr': atr,
                                'signal_type': signal_type,
                                'confidence_score': confidence_score,
                                'entry_bar': i,
                                'confluences': confluences
                            })
                            self.daily_trades[day] = self.daily_trades.get(day, 0) + 1
                            daily_risk += 100  # 1% risk
                            logger.debug("Opened position at bar %s: type=%s, size=%s, signal=%s, confidence=%s, confluences=%s, daily_trades=%s, daily_risk=%s", 
                                         i, pos_type, size, signal_type, confidence_score, confluences, self.daily_trades[day], daily_risk)
                        else:
                            logger.debug("No position opened at bar %s: size=%s, confidence_score=%s", i, size, confidence_score)
                    except Exception as e:
                        logger.error("Error opening position at bar %s: %s", i, e)
                        continue

                for pos in self.positions[:]:
                    try:
                        entry_price = pos['entry_price']
                        size = pos['size']
                        stop_loss = pos['stop_loss']
                        tp1 = pos['tp1']
                        tp2 = pos['tp2']
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
                            update_trade_log(trade_data, self.data.iloc[max(0, i - 50):i + 1].copy())
                            logger.info("Trade closed: signal_type=%s, pnl=%s, holding_period=%s hours", trade_data['signal_type'], trade_data['pnl'], trade_data['holding_period'])
                            self.positions.remove(pos)
                            logger.debug("Closed position at bar %s due to timeout: pnl=%s", i, pnl)
                            continue

                        if pos['type'] == 'long':
                            # Update trailing stop-loss if breakeven
                            if pos['breakeven']:
                                pos['stop_loss'] = max(pos['stop_loss'], current['Close'] * 0.99)
                            if not pos['breakeven'] and current['Close'] >= tp1:
                                pos['size'] = size // 2
                                pos['breakeven'] = True
                                pos['stop_loss'] = entry_price
                                pnl = (tp1 - entry_price) * (size // 2) - commission_cost
                                self.equity += pnl
                                trade_data = {
                                    'entry_time': pos['entry_time'],
                                    'exit_time': timestamp,
                                    'entry_price': entry_price,
                                    'exit_price': tp1,
                                    'pnl': pnl,
                                    'type': pos['type'],
                                    'size': size // 2,
                                    'rsi_entry': pos['rsi'],
                                    'atr_entry': pos['atr'],
                                    'signal_type': pos['signal_type'],
                                    'confidence_score': pos['confidence_score'],
                                    'holding_period': (timestamp - pos['entry_time']).total_seconds() / 3600,
                                    'reason': 'TP1 Hit',
                                    'result': 'win' if pnl > 0 else 'loss',
                                    'setup_type': 'scalp',
                                    'rr_ratio': (tp1 - entry_price) / (entry_price - stop_loss) if entry_price != stop_loss else 1.5,
                                    'confluences': pos['confluences'],
                                    'entry_bar': pos['entry_bar'],
                                    'exit_bar': i
                                }
                                self.trades.append(trade_data)
                                update_trade_log(trade_data, self.data.iloc[max(0, i - 50):i + 1].copy())
                                logger.info("Trade closed: signal_type=%s, pnl=%s, holding_period=%s hours", trade_data['signal_type'], trade_data['pnl'], trade_data['holding_period'])
                                logger.debug("Closed TP1 at bar %s: pnl=%s", i, pnl)

                            if pos['breakeven'] and (current['Close'] >= tp2 or current['Close'] <= stop_loss):
                                exit_price = tp2 if current['Close'] >= tp2 else stop_loss
                                exit_price *= (1 - self.slippage)
                                pnl = (exit_price - entry_price) * pos['size'] - commission_cost
                                self.equity += pnl
                                trade_data = {
                                    'entry_time': pos['entry_time'],
                                    'exit_time': timestamp,
                                    'entry_price': entry_price,
                                    'exit_price': exit_price,
                                    'pnl': pnl,
                                    'type': pos['type'],
                                    'size': pos['size'],
                                    'rsi_entry': pos['rsi'],
                                    'atr_entry': pos['atr'],
                                    'signal_type': pos['signal_type'],
                                    'confidence_score': pos['confidence_score'],
                                    'holding_period': (timestamp - pos['entry_time']).total_seconds() / 3600,
                                    'reason': 'TP2 Hit' if current['Close'] >= tp2 else 'SL Hit',
                                    'result': 'win' if pnl > 0 else 'loss',
                                    'setup_type': 'scalp',
                                    'rr_ratio': (tp2 - entry_price) / (entry_price - stop_loss) if entry_price != stop_loss else 2.0,
                                    'confluences': pos['confluences'],
                                    'entry_bar': pos['entry_bar'],
                                    'exit_bar': i
                                }
                                self.trades.append(trade_data)
                                update_trade_log(trade_data, self.data.iloc[max(0, i - 50):i + 1].copy())
                                logger.info("Trade closed: signal_type=%s, pnl=%s, holding_period=%s hours", trade_data['signal_type'], trade_data['pnl'], trade_data['holding_period'])
                                self.positions.remove(pos)
                                logger.debug("Closed position at bar %s: pnl=%s, reason=%s", i, pnl, trade_data['reason'])

                        elif pos['type'] == 'short':
                            # Update trailing stop-loss if breakeven
                            if pos['breakeven']:
                                pos['stop_loss'] = min(pos['stop_loss'], current['Close'] * 1.01)
                            if not pos['breakeven'] and current['Close'] <= tp1:
                                pos['size'] = size // 2
                                pos['breakeven'] = True
                                pos['stop_loss'] = entry_price
                                pnl = (entry_price - tp1) * (size // 2) - commission_cost
                                self.equity += pnl
                                trade_data = {
                                    'entry_time': pos['entry_time'],
                                    'exit_time': timestamp,
                                    'entry_price': entry_price,
                                    'exit_price': tp1,
                                    'pnl': pnl,
                                    'type': pos['type'],
                                    'size': size // 2,
                                    'rsi_entry': pos['rsi'],
                                    'atr_entry': pos['atr'],
                                    'signal_type': pos['signal_type'],
                                    'confidence_score': pos['confidence_score'],
                                    'holding_period': (timestamp - pos['entry_time']).total_seconds() / 3600,
                                    'reason': 'TP1 Hit',
                                    'result': 'win' if pnl > 0 else 'loss',
                                    'setup_type': 'scalp',
                                    'rr_ratio': (entry_price - tp1) / (stop_loss - entry_price) if stop_loss != entry_price else 1.5,
                                    'confluences': pos['confluences'],
                                    'entry_bar': pos['entry_bar'],
                                    'exit_bar': i
                                }
                                self.trades.append(trade_data)
                                update_trade_log(trade_data, self.data.iloc[max(0, i - 50):i + 1].copy())
                                logger.info("Trade closed: signal_type=%s, pnl=%s, holding_period=%s hours", trade_data['signal_type'], trade_data['pnl'], trade_data['holding_period'])
                                logger.debug("Closed TP1 at bar %s: pnl=%s", i, pnl)

                            if pos['breakeven'] and (current['Close'] <= tp2 or current['Close'] >= stop_loss):
                                exit_price = tp2 if current['Close'] <= tp2 else stop_loss
                                exit_price *= (1 + self.slippage)
                                pnl = (entry_price - exit_price) * pos['size'] - commission_cost
                                self.equity += pnl
                                trade_data = {
                                    'entry_time': pos['entry_time'],
                                    'exit_time': timestamp,
                                    'entry_price': entry_price,
                                    'exit_price': exit_price,
                                    'pnl': pnl,
                                    'type': pos['type'],
                                    'size': pos['size'],
                                    'rsi_entry': pos['rsi'],
                                    'atr_entry': pos['atr'],
                                    'signal_type': pos['signal_type'],
                                    'confidence_score': pos['confidence_score'],
                                    'holding_period': (timestamp - pos['entry_time']).total_seconds() / 3600,
                                    'reason': 'TP2 Hit' if current['Close'] <= tp2 else 'SL Hit',
                                    'result': 'win' if pnl > 0 else 'loss',
                                    'setup_type': 'scalp',
                                    'rr_ratio': (entry_price - tp2) / (stop_loss - entry_price) if stop_loss != entry_price else 2.0,
                                    'confluences': pos['confluences'],
                                    'entry_bar': pos['entry_bar'],
                                    'exit_bar': i
                                }
                                self.trades.append(trade_data)
                                update_trade_log(trade_data, self.data.iloc[max(0, i - 50):i + 1].copy())
                                logger.info("Trade closed: signal_type=%s, pnl=%s, holding_period=%s hours", trade_data['signal_type'], trade_data['pnl'], trade_data['holding_period'])
                                self.positions.remove(pos)
                                logger.debug("Closed position at bar %s: pnl=%s, reason=%s", i, pnl, trade_data['reason'])

                    except Exception as e:
                        logger.error("Error processing position at bar %s: %s", i, e)
                        continue

                self.results.loc[timestamp, 'Equity'] = self.equity
                if i % 1000 == 0:
                    logger.debug("Updated equity at bar %s: %s", i, self.equity)

            logger.info("Strategy run completed: %s trades executed", len(self.trades))
            return self.results, pd.DataFrame(self.trades)

        except Exception as e:
            logger.error("Error in strategy run: %s", e)
            raise