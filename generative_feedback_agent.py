import pandas as pd
import re
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenerativeFeedbackAgent:
    def __init__(self, trade_log_path, trades_csv_path, openai_api_key):
        self.trade_log_path = trade_log_path
        self.trades_csv_path = trades_csv_path
        self.llm = ChatOpenAI(model="gpt-4o", api_key=openai_api_key)
        self.trade_log_data = None
        self.trades_df = None
        self.rejection_stats = {}
        self.executed_trade_stats = {}

    def load_data(self):
        """Load trade log and executed trades data."""
        logger.info("Loading trade log from %s", self.trade_log_path)
        with open(self.trade_log_path, 'r') as f:
            self.trade_log_data = f.readlines()

        logger.info("Loading executed trades from %s", self.trades_csv_path)
        self.trades_df = pd.read_csv(self.trades_csv_path)
        self.trades_df['entry_time'] = pd.to_datetime(self.trades_df['entry_time'], utc=True)

    def parse_trade_log(self):
        """Parse trade log to extract rejection statistics."""
        rejection_reasons = {
            'QQQ trend not aligned': 0,
            'MTF trend not bearish': 0,
            'MTF trend not bullish': 0,
            'RSI not below 60': 0,
            'RSI not above 40': 0,
        }

        for line in self.trade_log_data:
            if "Skipping trade at bar" in line:
                for reason in rejection_reasons.keys():
                    if reason in line:
                        rejection_reasons[reason] += 1

        total_rejections = sum(rejection_reasons.values())
        self.rejection_stats = {
            reason: (count / total_rejections * 100) if total_rejections > 0 else 0
            for reason, count in rejection_reasons.items()
        }
        self.rejection_stats['total_rejections'] = total_rejections

    def analyze_executed_trades(self):
        """Analyze executed trades for R:R and trend day statistics."""
        self.trades_df['is_trend_day'] = self.trades_df['confluences'].apply(
            lambda x: True if "'Uptrend': True" in x or "'Downtrend': True" in x else False
        )
        self.trades_df['high_rr'] = self.trades_df['rr_ratio'] > 3.0
        self.executed_trade_stats = {
            'total_trades': len(self.trades_df),
            'trend_day_trades': len(self.trades_df[self.trades_df['is_trend_day']]),
            'high_rr_trades': len(self.trades_df[self.trades_df['high_rr']]),
            'trend_day_high_rr': len(self.trades_df[self.trades_df['is_trend_day'] & self.trades_df['high_rr']]),
        }

    def generate_feedback(self):
        """Use LangChain to generate feedback based on trade analysis."""
        # Flatten the dictionaries for the prompt
        flattened_inputs = {
            'rejection_stats_total_rejections': self.rejection_stats['total_rejections'],
            'rejection_stats_qqq_trend_not_aligned': self.rejection_stats['QQQ trend not aligned'],
            'rejection_stats_mtf_trend_not_bearish': self.rejection_stats['MTF trend not bearish'],
            'rejection_stats_mtf_trend_not_bullish': self.rejection_stats['MTF trend not bullish'],
            'rejection_stats_rsi_not_below_60': self.rejection_stats['RSI not below 60'],
            'rejection_stats_rsi_not_above_40': self.rejection_stats['RSI not above 40'],
            'executed_trade_stats_total_trades': self.executed_trade_stats['total_trades'],
            'executed_trade_stats_trend_day_trades': self.executed_trade_stats['trend_day_trades'],
            'executed_trade_stats_high_rr_trades': self.executed_trade_stats['high_rr_trades'],
            'executed_trade_stats_trend_day_high_rr': self.executed_trade_stats['trend_day_high_rr'],
        }

        prompt_template = PromptTemplate(
            input_variables=[
                "rejection_stats_total_rejections",
                "rejection_stats_qqq_trend_not_aligned",
                "rejection_stats_mtf_trend_not_bearish",
                "rejection_stats_mtf_trend_not_bullish",
                "rejection_stats_rsi_not_below_60",
                "rejection_stats_rsi_not_above_40",
                "executed_trade_stats_total_trades",
                "executed_trade_stats_trend_day_trades",
                "executed_trade_stats_high_rr_trades",
                "executed_trade_stats_trend_day_high_rr"
            ],
            template="""
You are a trading strategy optimization agent. Your task is to analyze the trade rejection statistics and executed trade statistics to suggest specific rule tweaks that could improve the strategy's performance. Focus on identifying filters that are blocking potentially profitable trades, especially on trend days or for high reward-to-risk (R:R) trades (R:R > 3.0).

**Rejection Statistics:**
- Total trades rejected: {rejection_stats_total_rejections}
- QQQ trend not aligned: {rejection_stats_qqq_trend_not_aligned:.2f}%
- MTF trend not bearish: {rejection_stats_mtf_trend_not_bearish:.2f}%
- MTF trend not bullish: {rejection_stats_mtf_trend_not_bullish:.2f}%
- RSI not below 60: {rejection_stats_rsi_not_below_60:.2f}%
- RSI not above 40: {rejection_stats_rsi_not_above_40:.2f}%

**Executed Trade Statistics:**
- Total trades executed: {executed_trade_stats_total_trades}
- Trades on trend days: {executed_trade_stats_trend_day_trades}
- High R:R trades (R:R > 3.0): {executed_trade_stats_high_rr_trades}
- High R:R trades on trend days: {executed_trade_stats_trend_day_high_rr}

**Instructions:**
1. Analyze the rejection statistics to identify which filters are blocking the most trades.
2. Consider the executed trade statistics to determine if high R:R trades or trend day trades are underrepresented.
3. Suggest specific rule tweaks (e.g., "relax the QQQ alignment filter on trend days", "lower the RSI threshold for longs to 35") to allow more high R:R trades or trend day trades to be executed.
4. Provide concise, actionable suggestions (2-3 suggestions max).

**Output Format:**
- Suggestion 1: [Specific rule tweak with reasoning]
- Suggestion 2: [Specific rule tweak with reasoning]
"""
        )

        # Create a RunnableSequence using the prompt and LLM
        chain = prompt_template | self.llm

        # Use invoke instead of run
        response = chain.invoke(flattened_inputs)
        feedback = response.content  # Extract the content from the response
        return feedback

    def run(self):
        """Run the feedback agent."""
        self.load_data()
        self.parse_trade_log()
        self.analyze_executed_trades()
        feedback = self.generate_feedback()
        logger.info("Feedback Generated:\n%s", feedback)
        return feedback

if __name__ == "__main__":
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")

    agent = GenerativeFeedbackAgent(
        trade_log_path="PM Strategy log1.txt",
        trades_csv_path="trades_output.csv",
        openai_api_key=openai_api_key
    )
    feedback = agent.run()
    print(feedback)