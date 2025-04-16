# BacktestEngine (Enhanced)
# --------------------------------------------------
# - Integrates with UGBacktestStrategy 75% win logic
# - Supports CLI run with confidence score filter
# - Outputs full equity curve and drawdown
# - Logs and exports trades with results

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from strategy_wrapper import UGBacktestStrategy
from data.data_fetcher import DataFetcher
import logging
import sys as sys_module

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest.log', encoding='utf-8'),
        logging.StreamHandler(sys_module.stdout)
    ]
)
logger = logging.getLogger(__name__)

class BacktestEngine:
    def __init__(self, initial_capital=10000, use_cached=True, cache_file='tsla_5min.csv', mock_data_file='TSLA_1M_5min_mock.csv'):
        self.initial_capital = initial_capital
        self.use_cached = use_cached
        self.cache_file = cache_file
        self.mock_data_file = mock_data_file
        self.results = None
        self.trades = None
        self.data = None
        logger.debug("Initialized BacktestEngine with initial_capital=%s, use_cached=%s", initial_capital, use_cached)

    def load_data(self):
        logger.debug("Starting data loading...")
        mock_path = os.path.join('data', self.mock_data_file)
        if os.path.exists(mock_path):
            logger.info("Loading mock data from %s...", mock_path)
            try:
                df = pd.read_csv(mock_path, parse_dates=['timestamp'], index_col='timestamp')
                # Adjust prices for TSLA split (divide by 30)
                df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']] / 30
                # Localize timestamps to US/Eastern
                if df.index.tz is None:
                    df.index = df.index.tz_localize('US/Eastern')
                    logger.debug("Localized mock data timestamps to US/Eastern")
                # Standardize column names
                column_mapping = {col: col.capitalize() for col in df.columns}
                for col in df.columns:
                    for std_col in ['open', 'high', 'low', 'close', 'volume']:
                        if col.lower() == std_col:
                            column_mapping[col] = std_col.capitalize()
                df = df.rename(columns=column_mapping)
                expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in df.columns for col in expected_cols):
                    logger.error("CSV missing required columns: %s", expected_cols)
                    raise ValueError(f"CSV missing required columns: {expected_cols}")
                logger.info("Loaded mock data with shape: %s, tz: %s", df.shape, df.index.tz)
                return df.sort_index()
            except Exception as e:
                logger.error("Failed to load mock data: %s", e)
                raise

        logger.debug("No mock data found, attempting Alpha Vantage...")
        fetcher = DataFetcher()
        try:
            if self.use_cached:
                cache_path = os.path.join('data', self.cache_file)
                if os.path.exists(cache_path):
                    df = pd.read_csv(cache_path, parse_dates=['timestamp'], index_col='timestamp')
                    # Adjust prices for TSLA split (divide by 30)
                    df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']] / 30
                    # Localize timestamps to US/Eastern
                    if df.index.tz is None:
                        df.index = df.index.tz_localize('US/Eastern')
                        logger.debug("Localized cached data timestamps to US/Eastern")
                    logger.info("Loaded cached data with shape: %s, tz: %s", df.shape, df.index.tz)
                    return df
            df = fetcher.fetch_5min_tsla(cache_file=self.cache_file)
            df = fetcher.clean_data(df)
            # Adjust prices for TSLA split (divide by 30)
            df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']] / 30
            # Localize timestamps to US/Eastern
            if df.index.tz is None:
                df.index = df.index.tz_localize('US/Eastern')
                logger.debug("Localized Alpha Vantage data timestamps to US/Eastern")
            logger.info("Loaded Alpha Vantage data with shape: %s, tz: %s", df.shape, df.index.tz)
            return df
        except Exception as e:
            logger.error("Failed to load Alpha Vantage data: %s", e)
            raise

    def run(self):
        logger.debug("Starting backtest run...")
        try:
            self.data = self.load_data()
            strategy = UGBacktestStrategy(self.data, self.initial_capital)
            self.results, self.trades = strategy.run()
            self.trades.to_pickle('backtest_trades.pkl')
            logger.info("Backtest completed successfully, trades saved to backtest_trades.pkl")
            return self.results, self.trades
        except Exception as e:
            logger.error("Backtest failed: %s", e)
            raise

    def calculate_metrics(self):
        logger.debug("Calculating metrics...")
        try:
            if self.results is None or self.trades is None or self.trades.empty:
                logger.warning("No trades executed. Skipping metrics calculation.")
                return {
                    'Total Return': 0.0,
                    'Annualized Return': 0.0,
                    'Max Drawdown': 0.0,
                    'Sharpe Ratio': 0.0,
                    'Sortino Ratio': 0.0,
                    'Alpha': 0.0,
                    'Number of Trades': 0,
                    'Win Rate': 0.0,
                    'Avg Trade PNL': 0.0,
                    'Avg Holding Period (Hours)': 0.0,
                    'Avg Holding Period Bars': 0.0
                }
            returns = self.results['Equity'].pct_change().dropna()
            benchmark_returns = self.data['Close'].pct_change().reindex(returns.index).dropna()
            aligned_returns = returns.loc[benchmark_returns.index]
            
            risk_free_rate = 0.02 / (252 * 78)
            excess_returns = aligned_returns - risk_free_rate
            excess_benchmark = benchmark_returns - risk_free_rate
            
            metrics = {
                'Total Return': (self.results['Equity'].iloc[-1] / self.initial_capital - 1) * 100,
                'Annualized Return': ((1 + returns.mean()) ** (252 * 78) - 1) * 100,
                'Max Drawdown': ((self.results['Equity'].cummax() - self.results['Equity']) / self.results['Equity'].cummax()).max() * 100,
                'Sharpe Ratio': excess_returns.mean() / excess_returns.std() * np.sqrt(252 * 78) if excess_returns.std() != 0 else np.nan,
                'Sortino Ratio': excess_returns.mean() / excess_returns[excess_returns < 0].std() * np.sqrt(252 * 78) if excess_returns[excess_returns < 0].std() != 0 else np.nan,
                'Alpha': (aligned_returns.mean() - benchmark_returns.mean()) * (252 * 78) * 100 if len(benchmark_returns) > 0 else np.nan,
                'Number of Trades': len(self.trades),
                'Win Rate': len(self.trades[self.trades['pnl'] > 0]) / len(self.trades) * 100 if len(self.trades) > 0 else 0,
                'Avg Trade PNL': self.trades['pnl'].mean() if len(self.trades) > 0 else 0,
                'Avg Holding Period (Hours)': self.trades['holding_period'].mean() if len(self.trades) > 0 else 0,
                'Avg Holding Period Bars': (self.trades['exit_bar'] - self.trades['entry_bar']).mean() if 'entry_bar' in self.trades.columns and len(self.trades) > 0 else 0
            }
            logger.debug("Metrics calculated: %s", metrics)
            return metrics
        except Exception as e:
            logger.error("Error calculating metrics: %s", e)
            return None

    def analyze(self):
        logger.debug("Analyzing backtest results...")
        try:
            if self.results is not None:
                metrics = self.calculate_metrics()
                if metrics:
                    for key, value in metrics.items():
                        print(f"{key}: {value:.2f}")
                    logger.info("Analysis completed")
                    return metrics
                else:
                    logger.warning("No metrics to analyze")
            else:
                logger.warning("No results to analyze")
            print("No results to analyze.")
            return None
        except Exception as e:
            logger.error("Error during analysis: %s", e)
            return None

    def plot(self):
        logger.debug("Generating plots...")
        try:
            if self.results is not None:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                ax1.plot(self.results.index, self.results['Equity'], label='Equity')
                ax1.set_title('Equity Curve')
                ax1.set_ylabel('Equity')
                ax1.legend()
                drawdown = (self.results['Equity'].cummax() - self.results['Equity']) / self.results['Equity'].cummax() * 100
                ax2.plot(self.results.index, drawdown, label='Drawdown', color='red')
                ax2.set_title('Drawdown')
                ax2.set_ylabel('Drawdown %')
                ax2.legend()
                plt.tight_layout()
                plt.show()
                if not self.trades.empty:
                    plt.figure(figsize=(8, 6))
                    sns.histplot(self.trades['pnl'], bins=50)
                    plt.title('Trade PNL Distribution')
                    plt.xlabel('PNL')
                    plt.show()
                logger.info("Plots generated successfully")
            else:
                logger.warning("No results to plot")
                print("No results to plot.")
        except Exception as e:
            logger.error("Error generating plots: %s", e)

if __name__ == "__main__":
    from learn import init_db
    logger.info("Starting main execution...")
    try:
        init_db()
        logger.debug("Database initialized")
        engine = BacktestEngine(use_cached=True)
        results, trades = engine.run()
        engine.analyze()
        engine.plot()
    except Exception as e:
        logger.error("Main execution failed: %s", e)
        raise