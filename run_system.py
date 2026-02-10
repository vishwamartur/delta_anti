"""
Delta Anti Trading System - Unified Launcher
Runs API Server + Trading System + Trade Manager as ONE integrated system
"""
import threading
import logging
import sys
import os
import time
import signal as sig

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

import config
from data.market_data import market_data
from analysis.indicators import indicators
from analysis.signals import signal_generator, TradeSignal, SignalType
from strategy.advanced_trade_manager import (
    initialize_trade_manager, get_trade_manager, ExitReason
)
from api.delta_rest import rest_client
from api.delta_websocket import ws_client
from ui.dashboard import dashboard

# DQN Agent (optional)
try:
    from ml.agents.dqn_trader import dqn_agent
    DQN_AVAILABLE = True
except ImportError:
    DQN_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


class IntegratedTradingSystem:
    """
    Fully integrated trading system with:
    - Real-time WebSocket data
    - Technical analysis
    - ML predictions
    - Advanced trade management
    - Risk controls
    """
    
    def __init__(self):
        self.running = False
        self.symbols = config.TRADING_SYMBOLS
        self.timeframe = config.DEFAULT_TIMEFRAME
        
        # Initialize trade manager
        self.trade_manager = initialize_trade_manager(
            account_balance=config.INITIAL_ACCOUNT_BALANCE,
            trade_config=config.TRADE_MANAGER_CONFIG
        )
        
        # Reset daily stats on startup to clear any old loss limits
        if hasattr(self.trade_manager, 'risk_manager'):
            self.trade_manager.risk_manager.reset_daily_stats()
            logger.info("[SYSTEM] Daily loss limit reset - fresh start!")
        
        # Store indicators and signals
        self._indicators = {}
        self._signals = {}
        self._htf_data = {}  # Higher timeframe DataFrames per symbol
        
        # Multi-timeframe config
        self._mtf_config = getattr(config, 'MULTI_TIMEFRAME_CONFIG', {})
        self._htf_timeframe = self._mtf_config.get('higher_timeframe', '1h')
        self._htf_lookback = self._mtf_config.get('htf_lookback_hours', 24)
        
        # DQN tracking
        self._dqn_config = getattr(config, 'DQN_INTEGRATION_CONFIG', {})
        self._dqn_entry_states = {}  # trade_id -> (state, action)
        self._dqn_trades_completed = 0
        
        # Signal handlers
        sig.signal(sig.SIGINT, self._signal_handler)
        sig.signal(sig.SIGTERM, self._signal_handler)
        
        logger.info("Integrated Trading System initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print("\n[SYSTEM] Shutting down gracefully...")
        self.stop()
        sys.exit(0)
    
    def _on_ticker(self, data):
        """Handle ticker updates - REAL-TIME exit condition checking."""
        market_data.update_ticker(data)
        
        # Extract symbol and price from ticker
        symbol = data.get('symbol', '')
        if not symbol or symbol.startswith('MARK:'):
            return
        
        # Get current price from ticker (mark_price or last_price)
        current_price = None
        if 'mark_price' in data:
            current_price = float(data['mark_price'])
        elif 'close' in data:
            current_price = float(data['close'])
        elif 'last_price' in data:
            current_price = float(data['last_price'])
        
        if not current_price or current_price <= 0:
            return
        
        # === REAL-TIME EXIT CHECK ===
        # Check all open trades for this symbol on EVERY ticker update
        for trade_id, trade in list(self.trade_manager.open_trades.items()):
            if trade.symbol == symbol:
                # Update P&L with real-time price
                self.trade_manager.update_trade_pnl(trade_id, current_price)
                
                # Immediately check exit conditions (TP/SL/Trailing)
                exit_reason = self.trade_manager.check_exit_conditions(trade_id)
                if exit_reason:
                    logger.info(f"[REAL-TIME] {symbol} hit {exit_reason.value} @ ${current_price:,.2f}")
                    closed_trade = self.trade_manager.close_trade(
                        trade_id, exit_reason, current_price
                    )
                    if closed_trade:
                        self._feed_dqn_reward(closed_trade)
                        dashboard.update(
                            message=f"🎯 {exit_reason.value}: {symbol} @ ${current_price:,.2f} | "
                                   f"P&L: ${closed_trade.realized_pnl:+.2f} ({closed_trade.pnl_percent:+.2f}%)"
                        )
    
    def _on_candlestick(self, data):
        """Handle candlestick updates."""
        market_data.update_from_ws_candlestick(data)
        
        symbol = data.get('symbol', '')
        if symbol and not symbol.startswith('MARK:'):
            self._analyze_and_trade(symbol)
    
    def _analyze_and_trade(self, symbol):
        """Analyze symbol and execute trades via Advanced Trade Manager."""
        df = market_data.get_dataframe(symbol)
        if df is None or len(df) < 50:
            return
        
        # Calculate indicators
        ind = indicators.calculate_all(df)
        if ind is None:
            return
        
        self._indicators[symbol] = ind
        
        # Update open trades P&L and check exits
        for trade_id, trade in list(self.trade_manager.open_trades.items()):
            if trade.symbol == symbol:
                # Update P&L with current price
                self.trade_manager.update_trade_pnl(trade_id, ind.price)
                
                # Check exit conditions (SL/TP/Trailing)
                exit_reason = self.trade_manager.check_exit_conditions(trade_id)
                if exit_reason:
                    closed_trade = self.trade_manager.close_trade(
                        trade_id, exit_reason, ind.price
                    )
                    if closed_trade:
                        self._feed_dqn_reward(closed_trade)
                        dashboard.update(
                            message=f"EXIT: {symbol} - {exit_reason.value} "
                                   f"P&L: ${closed_trade.realized_pnl:+.2f} "
                                   f"({closed_trade.pnl_percent:+.2f}%)"
                        )
                    return
        
        # Generate entry signal if no open position for this symbol
        if not self.trade_manager.has_open_position(symbol):
            df_htf = self._htf_data.get(symbol)  # Higher timeframe data
            signal = signal_generator.generate_signal(symbol, ind, df, df_htf=df_htf)
            self._signals[symbol] = signal
            
            if signal.is_actionable():
                self._execute_signal(signal, ind)
    
    def _execute_signal(self, signal: TradeSignal, indicators):
        """Execute trade from signal via Advanced Trade Manager."""
        
        # Create trade via trade manager
        trade = self.trade_manager.create_trade_from_signal(signal, indicators)
        
        if trade:
            # Store DQN entry state for experience replay
            if DQN_AVAILABLE and self._dqn_config.get('train_on_outcomes', False):
                entry_state = getattr(signal_generator, '_last_dqn_state', None)
                entry_action = getattr(signal_generator, '_last_dqn_action', None)
                if entry_state is not None:
                    self._dqn_entry_states[trade.trade_id] = (entry_state, entry_action)
                    logger.info(f"[DQN] Stored entry state for trade {trade.trade_id}")
            
            dashboard.update(
                message=f"TRADE: {trade.direction.value} {trade.symbol} "
                       f"@ ${trade.entry_price:,.2f} | Size: {trade.entry_size:.4f} | "
                       f"SL: ${trade.stop_loss:,.2f} | TP: ${trade.take_profit:,.2f}"
            )
            dashboard.print_signal_alert(signal)
    
    def _feed_dqn_reward(self, closed_trade):
        """Feed trade outcome to DQN agent for online learning."""
        if not DQN_AVAILABLE or not self._dqn_config.get('train_on_outcomes', False):
            return
        
        trade_id = closed_trade.trade_id
        entry_data = self._dqn_entry_states.pop(trade_id, None)
        if entry_data is None:
            return
        
        entry_state, action = entry_data
        
        try:
            # Calculate reward from PnL
            pnl_pct = closed_trade.pnl_percent
            current_position = 1 if closed_trade.is_long else -1
            reward = dqn_agent.calculate_reward(action or 0, pnl_pct, current_position)
            
            # Build next state from latest indicators
            import numpy as np
            ind = self._indicators.get(closed_trade.symbol)
            if ind is not None:
                next_state = signal_generator._build_dqn_state(ind)
            else:
                next_state = np.zeros(50, dtype=np.float32)
            
            # Store experience and train
            dqn_agent.store_experience(entry_state, action or 0, reward, next_state, done=True)
            loss = dqn_agent.train_step()
            
            self._dqn_trades_completed += 1
            logger.info(f"[DQN] Trade {trade_id} feedback: reward={reward:.3f}, "
                       f"PnL={pnl_pct:+.2f}%, loss={loss:.4f if loss else 'N/A'}, "
                       f"epsilon={dqn_agent.epsilon:.3f}")
            
            # Periodic save
            save_freq = self._dqn_config.get('save_every_n_trades', 10)
            if self._dqn_trades_completed % save_freq == 0:
                dqn_agent.save()
                logger.info(f"[DQN] Model saved after {self._dqn_trades_completed} trades")
        
        except Exception as e:
            logger.warning(f"[DQN] Reward feedback error: {e}")
    
    def _load_historical_data(self):
        """Load historical candles for all symbols (primary + higher TF)."""
        print("[SYSTEM] Loading historical data...")
        for symbol in self.symbols:
            # Load primary timeframe (e.g., 5m)
            success = market_data.load_historical_candles(
                symbol=symbol,
                resolution=self.timeframe,
                lookback_hours=24
            )
            
            # Load higher timeframe (e.g., 1h) for multi-TF analysis
            if self._mtf_config.get('enabled', True):
                self._load_htf_data(symbol)
            
            if success:
                self._analyze_and_trade(symbol)
        print("[SYSTEM] Historical data loaded")
    
    def _load_htf_data(self, symbol: str):
        """Load higher timeframe candles for multi-timeframe analysis."""
        try:
            import pandas as pd
            end_time = int(time.time())
            start_time = end_time - (self._htf_lookback * 3600)
            
            response = rest_client.get_candles(
                symbol=symbol,
                resolution=self._htf_timeframe,
                start=start_time,
                end=end_time
            )
            
            if 'result' in response and response['result']:
                candles = response['result']
                data = []
                for c in candles:
                    data.append({
                        'timestamp': c.get('time', 0),
                        'open': float(c.get('open', 0)),
                        'high': float(c.get('high', 0)),
                        'low': float(c.get('low', 0)),
                        'close': float(c.get('close', 0)),
                        'volume': float(c.get('volume', 0))
                    })
                
                df_htf = pd.DataFrame(data)
                if not df_htf.empty:
                    df_htf['datetime'] = pd.to_datetime(df_htf['timestamp'], unit='us')
                    df_htf.set_index('datetime', inplace=True)
                    self._htf_data[symbol] = df_htf
                    logger.info(f"[HTF] Loaded {len(df_htf)} {self._htf_timeframe} candles for {symbol}")
        except Exception as e:
            logger.warning(f"[HTF] Failed to load data for {symbol}: {e}")
    
    def _setup_websocket(self):
        """Setup WebSocket connections."""
        ws_client.on_ticker = self._on_ticker
        ws_client.on_candlestick = self._on_candlestick
        
        ws_client.start()
        time.sleep(2)
        
        if ws_client.is_connected:
            ws_client.subscribe_ticker(self.symbols)
            ws_client.subscribe_candlestick(self.symbols, self.timeframe)
            print(f"[SYSTEM] WebSocket connected - Subscribed to {self.symbols}")
            print(f"[SYSTEM] ✅ Real-time TP/SL monitoring ACTIVE via ticker updates")
        else:
            print("[SYSTEM] WebSocket failed - Running in REST-only mode")
    
    def _update_dashboard(self):
        """Update dashboard with latest data and sync with exchange."""
        # Sync positions from Delta Exchange (real-time)
        try:
            if hasattr(self.trade_manager, 'sync_positions'):
                self.trade_manager.sync_positions()
                self.trade_manager.sync_balance()
        except Exception as e:
            logger.debug(f"Sync error: {e}")
        
        prices = {}
        for symbol in self.symbols:
            price = market_data.get_latest_price(symbol)
            if price:
                prices[symbol] = price
        
        # Get open trades from advanced trade manager
        open_trades = list(self.trade_manager.open_trades.values())
        
        dashboard.update(
            prices=prices,
            indicators=self._indicators,
            signals=self._signals,
            trades=open_trades,
            stats=self.trade_manager.get_statistics()
        )
    
    def start(self):
        """Start the integrated trading system."""
        self.running = True
        
        self._load_historical_data()
        self._setup_websocket()
        self._update_dashboard()
        
        print("\n[SYSTEM] Starting live trading dashboard... (Press Ctrl+C to exit)\n")
        time.sleep(1)
        
        try:
            if ws_client.is_connected:
                dashboard.run_live(
                    update_callback=self._update_dashboard,
                    refresh_rate=config.REFRESH_RATE
                )
            else:
                # Polling mode - fetch candles via REST and analyze
                while self.running:
                    try:
                        for symbol in self.symbols:
                            end_time = int(time.time())
                            start_time = end_time - 3600  # Last hour
                            
                            response = rest_client.get_candles(
                                symbol=symbol,
                                resolution=self.timeframe,
                                start=start_time,
                                end=end_time
                            )
                            
                            if 'result' in response:
                                from data.market_data import Candle
                                for c in response['result'][-5:]:
                                    candle = Candle(
                                        timestamp=c.get('time', 0),
                                        open=float(c.get('open', 0)),
                                        high=float(c.get('high', 0)),
                                        low=float(c.get('low', 0)),
                                        close=float(c.get('close', 0)),
                                        volume=float(c.get('volume', 0))
                                    )
                                    market_data.add_candle(symbol, candle)
                            
                            self._analyze_and_trade(symbol)
                        
                        self._update_dashboard()
                    except Exception as e:
                        logger.error(f"Polling error: {e}")
                    
                    time.sleep(10)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
    
    def stop(self):
        """Stop the trading system."""
        self.running = False
        ws_client.stop()
        print("\n[SYSTEM] Trading system stopped")
        
        # Print final stats
        stats = self.trade_manager.get_statistics()
        print(f"\n=== Final Statistics ===")
        print(f"Total Trades: {stats['total_trades']}")
        print(f"Win Rate: {stats['win_rate']:.1f}%")
        print(f"Daily P&L: ${stats['daily_pnl']:+.2f}")
        print(f"Account Balance: ${stats['account_balance']:,.2f}")


def run_api_server():
    """Run the API server in a thread."""
    try:
        import uvicorn
        from api.server.main import app
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
    except Exception as e:
        logger.error(f"API Server error: {e}")


def main():
    """Run the complete integrated trading system."""
    
    print("=" * 70)
    print("    DELTA ANTI TRADING SYSTEM - UNIFIED LAUNCHER")
    print("=" * 70)
    print(f"  Symbols: {', '.join(config.TRADING_SYMBOLS)}")
    print(f"  Timeframe: {config.DEFAULT_TIMEFRAME}")
    print(f"  Account Balance: ${config.INITIAL_ACCOUNT_BALANCE:,.2f}")
    print(f"  Risk Per Trade: {config.TRADE_MANAGER_CONFIG.get('max_risk_per_trade', 0.02)*100}%")
    print(f"  Max Positions: {config.TRADE_MANAGER_CONFIG.get('max_positions', 5)}")
    print(f"  Auto Execution: {config.TRADE_MANAGER_CONFIG.get('enable_auto_execution', False)}")
    print("=" * 70)
    
    # Start API server in background thread
    api_thread = threading.Thread(target=run_api_server, daemon=True)
    api_thread.start()
    
    print("[+] API Server started: http://localhost:8000")
    print("    - Docs: http://localhost:8000/docs")
    print("    - Trades: http://localhost:8000/api/trades/active")
    print("    - Stats: http://localhost:8000/api/trades/stats")
    
    time.sleep(2)  # Let API server start
    
    print("=" * 70)
    print("[+] Starting Integrated Trading System...")
    print("=" * 70)
    
    # Run the integrated trading system
    system = IntegratedTradingSystem()
    system.start()


if __name__ == "__main__":
    main()
