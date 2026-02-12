
"""
Backtesting Engine
Simulates trading strategies against historical data.
"""
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from analysis.indicators import indicators, IndicatorValues

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Backtester:
    """
    Backtesting engine handling data loading, simulation loop, and performance metrics.
    """
    
    def __init__(self, symbol: str, initial_balance: float = None):
        self.symbol = symbol
        self.config = getattr(config, 'BACKTEST_CONFIG', {
            "initial_balance": 10000.0,
            "commission_rate": 0.0006,
            "slippage_pct": 0.0002
        })
        
        self.balance = initial_balance or self.config.get('initial_balance', 10000.0)
        self.initial_balance = self.balance
        self.position = 0.0  # Current position size
        self.entry_price = 0.0
        self.equity_curve = []
        self.trades = []
        
        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.max_drawdown = 0.0
        
    def load_data(self, days: int = 30, interval: str = "1h") -> pd.DataFrame:
        """Load historical data from yfinance."""
        logger.info(f"Loading {days} days of {interval} data for {self.symbol}...")
        
        # Map symbol to yfinance format (e.g. BTCUSD -> BTC-USD)
        yf_symbol = self.symbol
        if "USD" in self.symbol and "-" not in self.symbol:
             yf_symbol = self.symbol.replace("USD", "-USD")
             
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            df = yf.download(yf_symbol, start=start_date, end=end_date, interval=interval, progress=False)
            
            if df.empty:
                logger.error(f"No data found for {yf_symbol}")
                return None
                
            # Flatten multi-index columns if present (yfinance update)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Standardize columns
            df.columns = [c.lower() for c in df.columns]
            df.rename(columns={'date': 'datetime'}, inplace=True)
            
            logger.info(f"Loaded {len(df)} candles.")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None

    def run(self, df: pd.DataFrame):
        """Run the backtest simulation."""
        logger.info("Starting backtest simulation...")
        
        # Pre-calculate indicators for speed
        logger.info("Calculating technical indicators...")
        
        # We need to simulate the rolling window for signals
        # But for efficiency, we can pre-calc indicators on the whole DF
        # WARNING: Some indicators might "peek" if not careful. 
        # For this v1, we assume indicators library is safe (it usually is standard TA-Lib style).
        
        full_indicators = indicators.calculate_all(df)
        if not full_indicators:
            logger.error("Failed to calculate indicators.")
            return

        # Prepare strategy selector
        try:
            from strategy.selector import get_strategy_selector
            strategy_selector = get_strategy_selector()
            SELECTOR_AVAILABLE = True
        except ImportError:
            strategy_selector = None
            SELECTOR_AVAILABLE = False
            logger.warning("Adaptive Strategy Selector not available, using default signals.")
                
        # Lazy import strategy components
        from analysis.signals import signal_generator, SignalType

        # Simulation Loop
        # Start from index 50 to allow indicators to warm up
        params = self.config
        comm = params.get('commission_rate', 0.0006)
        slippage = params.get('slippage_pct', 0.0002)
        
        for i in range(50, len(df)):
            # Slice concept: At index i, we only know data up to i
            # But calculating indicators on every slice is slow. 
            # We use pre-calculated values at index i.
            
            current_candle = df.iloc[i]
            timestamp = df.index[i]
            price = current_candle['close']
            
            # Create a localized indicator object for this step
            # This is a bit of a hack to reuse existing logic which expects an IndicatorValues object
            # We need to populate it with scalar values from the current step
            
            # Since `indicators.calculate_all` returns an object with Series, 
            # we can't easily extract a scalar object without modifying the library.
            # Workaround: The system uses `ind.rsi` (scalar) during live trading.
            # We need to mock that interface.
            
            class MockIndicators:
                pass
            
            step_ind = MockIndicators()
            step_ind.price = price
            
            # Map standard attributes from the pre-calculated series
            attrs = ['rsi', 'adx', 'plus_di', 'minus_di', 'macd_line', 'macd_signal', 'macd_histogram',
                     'ema_9', 'ema_21', 'ema_50', 'sma_200', 'atr', 'bb_upper', 'bb_lower', 'bb_width',
                     'volume_ratio', 'bb_percent']
            
            for attr in attrs:
                if hasattr(full_indicators, attr):
                    val = getattr(full_indicators, attr)
                    if isinstance(val, (pd.Series, np.ndarray)):
                         # Get value at 'i'. If using Series with datetime index, use iloc[i]
                         # But wait, full_indicators stores raw values.
                         # Let's check `indicators.py`. It returns `IndicatorValues` with scalars?
                         # correct. `calculate_all` returns an instance with scalar values of the LAST candle.
                         # Ah, so we SHOULD calculate inside the loop for exact parity, but that's O(N^2).
                         pass
            
            # RE-EVALUATION:
            # `indicators.calculate_all(df)` takes the ENTIRE df and returns values for the LAST row.
            # To backtest correctly without modifying `calculate_all`, we must pass `df.iloc[:i+1]`.
            # Yes, meaningful backtesting requires O(N^2) indicator calculation or a refactor.
            # Given N is usually small (< 1000 for 1H candles in 30 days), we can do the loop.
            
            # Optimization: pass a window
            window_df = df.iloc[max(0, i-200):i+1] # Pass enough history for 200 SMA
            
            step_ind = indicators.calculate_all(window_df)
            if not step_ind:
                continue
                
            # === GENERATE SIGNAL ===
            signal = None
            strategy_name = "None"
            
            # Use Selector if available
            if SELECTOR_AVAILABLE and strategy_selector:
                # Need market condition
                # Mocking market analyzer context... simpler to just instantiate one
                from analysis.market_analyzer import MarketAnalyzer
                ma = MarketAnalyzer()
                
                # Analyze market (expensive but necessary for correct regime)
                market_condition = ma.analyze_market(self.symbol, window_df, step_ind)
                
                res = strategy_selector.select_and_generate(self.symbol, step_ind, market_condition, window_df)
                signal = res.signal
                strategy_name = res.strategy_name
            else:
                # Fallback to simple signal generator
                signal = signal_generator.generate_signal(self.symbol, step_ind, window_df)
                strategy_name = "Momentum"

            # === EXECUTE TRADES ===
            
            # 1. Check Exit if in position
            if self.position != 0:
                # Simple Exit Logic for Backtest (SL/TP from signal)
                # In real system, TradeManager handles this. We simulate basic logic here.
                
                # Check SL
                if self.position > 0 and price <= self.sl_price:
                     self._close_position(price, timestamp, "Stop Loss")
                elif self.position < 0 and price >= self.sl_price:
                     self._close_position(price, timestamp, "Stop Loss")
                     
                # Check TP
                elif self.position > 0 and price >= self.tp_price:
                     self._close_position(price, timestamp, "Take Profit")
                elif self.position < 0 and price <= self.tp_price:
                     self._close_position(price, timestamp, "Take Profit")
                     
                # Check Reversal (Opposite Signal)
                elif self.position > 0 and signal.signal_type == SignalType.SHORT and signal.confidence > 60:
                     self._close_position(price, timestamp, "Reversal Short")
                elif self.position < 0 and signal.signal_type == SignalType.LONG and signal.confidence > 60:
                     self._close_position(price, timestamp, "Reversal Long")

            # 2. Check Entry if flat
            if self.position == 0 and signal.is_actionable():
                # Simulate Entry
                size = 1.0 # fixed 1 unit for simplicity or use config
                if self.config.get('use_fixed_size'):
                    size = self.config.get('fixed_position_size', 1.0)
                else:
                    # Dynamic sizing simulation
                    risk = self.balance * 0.02 # 2% risk
                    stop_dist = abs(price - signal.stop_loss)
                    if stop_dist > 0:
                        size = risk / stop_dist
                
                # Apply slippage
                # Long: buy HIGHER, Short: sell LOWER (wait, sell price is lower for bad slippage)
                exec_price = price * (1 + slippage) if signal.signal_type == SignalType.LONG else price * (1 - slippage)
                
                self._open_position(signal.signal_type, exec_price, size, signal.stop_loss, signal.take_profit, timestamp, strategy_name)

            # Record Equity
            unrealized_pnl = 0
            if self.position > 0:
                unrealized_pnl = (price - self.entry_price) * abs(self.position)
            elif self.position < 0:
                unrealized_pnl = (self.entry_price - price) * abs(self.position)
                
            current_equity = self.balance + unrealized_pnl
            self.equity_curve.append({'timestamp': timestamp, 'equity': current_equity})
            
        logger.info("Backtest complete.")
        self._generate_report()

    def _open_position(self, type_, price, size, sl, tp, timestamp, strategy):
        from analysis.signals import SignalType
        self.position = size if type_ == SignalType.LONG else -size
        self.entry_price = price
        self.sl_price = sl
        self.tp_price = tp
        self.entry_time = timestamp
        self.current_strategy = strategy
        
        # Deduct Commission
        cost = price * size * self.config['commission_rate']
        self.balance -= cost
        
        # logger.info(f"OPEN {type_.name} @ {price:.2f} ({timestamp})")

    def _close_position(self, price, timestamp, reason):
        # Calculate PnL
        size = abs(self.position)
        if self.position > 0: # Long
            revenue = (price - self.entry_price) * size
        else: # Short
            revenue = (self.entry_price - price) * size
            
        # Deduct Commission
        cost = price * size * self.config['commission_rate']
        final_pnl = revenue - cost
        
        self.balance += final_pnl
        self.total_trades += 1
        if final_pnl > 0: self.winning_trades += 1
        else: self.losing_trades += 1
        
        self.trades.append({
            'entry_time': self.entry_time,
            'exit_time': timestamp,
            'type': 'LONG' if self.position > 0 else 'SHORT',
            'entry_price': self.entry_price,
            'exit_price': price,
            'pnl': final_pnl,
            'pnl_pct': (final_pnl / (self.entry_price * size)) * 100,
            'strategy': self.current_strategy,
            'reason': reason
        })
        
        # logger.info(f"CLOSE {reason} @ {price:.2f} | PnL: {final_pnl:.2f}")
        self.position = 0

    def _generate_report(self):
        if not self.trades:
            print("No trades generated.")
            return

        df_trades = pd.DataFrame(self.trades)
        equity_series = pd.DataFrame(self.equity_curve).set_index('timestamp')
        
        total_pnl = self.balance - self.initial_balance
        pnl_pct = (total_pnl / self.initial_balance) * 100
        win_rate = (self.winning_trades / self.total_trades) * 100
        
        # Drawdown
        equity_series['peak'] = equity_series['equity'].cummax()
        equity_series['drawdown'] = (equity_series['equity'] - equity_series['peak']) / equity_series['peak'] * 100
        max_dd = equity_series['drawdown'].min()

        print("\n" + "="*30)
        print(f" BACKTEST REPORT: {self.symbol}")
        print("="*30)
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Final Balance:   ${self.balance:,.2f}")
        print(f"Total PnL:       ${total_pnl:,.2f} ({pnl_pct:+.2f}%)")
        print(f"Total Trades:    {self.total_trades}")
        print(f"Win Rate:        {win_rate:.1f}%")
        print(f"Max Drawdown:    {max_dd:.2f}%")
        print("-" * 30)
        print("Trades by Strategy:")
        print(df_trades.groupby('strategy')['pnl'].sum())
        print("="*30 + "\n")
        
        # Visualization
        try:
            plt.figure(figsize=(12, 8))
            
            # Plot 1: Equity Curve
            plt.subplot(2, 1, 1)
            plt.plot(equity_series.index, equity_series['equity'], label='Equity', color='green')
            plt.title(f'Backtest: {self.symbol} Equity Curve')
            plt.ylabel('Balance ($)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Plot 2: Drawdown
            plt.subplot(2, 1, 2)
            plt.fill_between(equity_series.index, equity_series['drawdown'], 0, color='red', alpha=0.3)
            plt.title('Drawdown (%)')
            plt.ylabel('% Drawdown')
            plt.grid(True, alpha=0.3)
            
            # Save plot
            filename = f"backtest_{self.symbol}_{datetime.now().strftime('%Y%m%d')}.png"
            plt.savefig(filename)
            print(f"Chart saved to {filename}")
            
        except Exception as e:
            logger.error(f"Plotting failed: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSD", help="Symbol to backtest")
    parser.add_argument("--days", type=int, default=7, help="Days of history")
    parser.add_argument("--no-ml", action="store_true", help="Disable ML models for speed")
    parser.add_argument("--mock-signals", action="store_true", help="Use random signals for engine testing")
    args = parser.parse_args()
    
    # Disable ML if requested
    if args.no_ml:
        # Patch config before imports
        config.ML_AVAILABLE = False
        config.LAG_LLAMA_AVAILABLE = False
        config.SENTIMENT_AVAILABLE = False
        
        # Aggressively mock ML modules to prevent heavy imports
        import sys
        from unittest.mock import MagicMock
        
        # Mock specific ML modules
        sys.modules['ml'] = MagicMock()
        sys.modules['ml.models'] = MagicMock()
        sys.modules['ml.models.lstm_predictor'] = MagicMock()
        sys.modules['ml.models.lag_llama_predictor'] = MagicMock()
        sys.modules['ml.sentiment'] = MagicMock()
        sys.modules['ml.sentiment.market_sentiment'] = MagicMock()
        
        # Now import signals - it will use the mocks
        if 'analysis.signals' in sys.modules:
            del sys.modules['analysis.signals']
            
        import analysis.signals as signals_module
        signals_module.ML_AVAILABLE = False
        signals_module.LAG_LLAMA_AVAILABLE = False
        signals_module.SENTIMENT_AVAILABLE = False
        signals_module.RANGE_STRATEGY_AVAILABLE = True 
        
        logger.info("ML models disabled and mocked for backtest.")
    
    bt = Backtester(args.symbol)
    
    # Mock signals if requested
    if args.mock_signals:
        def mock_generate(*args, **kwargs):
            import random
            from analysis.signals import SignalType, TradeSignal
            
            # 10% chance of signal
            if random.random() < 0.1:
                direction = SignalType.LONG if random.random() > 0.5 else SignalType.SHORT
                price = args[1].price # args[1] is step_ind
                sl = price * 0.98 if direction == SignalType.LONG else price * 1.02
                tp = price * 1.04 if direction == SignalType.LONG else price * 0.96
                return TradeSignal(direction, 0.8, price, sl, tp, "Mock", "MockStrategy")
            return TradeSignal(SignalType.NEUTRAL, 0, 0, 0, 0, "Mock", "None")
            
        # Monkey patch
        from analysis import signals
        signals.signal_generator.generate_signal = mock_generate
        logger.info("Using MOCKED signals for engine verification.")

    df = bt.load_data(days=args.days)
    if df is not None:
        bt.run(df)
