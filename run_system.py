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
        """Handle ticker updates."""
        market_data.update_ticker(data)
    
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
                        dashboard.update(
                            message=f"EXIT: {symbol} - {exit_reason.value} "
                                   f"P&L: ${closed_trade.realized_pnl:+.2f} "
                                   f"({closed_trade.pnl_percent:+.2f}%)"
                        )
                    return
        
        # Generate entry signal if no open position for this symbol
        if not self.trade_manager.has_open_position(symbol):
            signal = signal_generator.generate_signal(symbol, ind)
            self._signals[symbol] = signal
            
            if signal.is_actionable():
                self._execute_signal(signal, ind)
    
    def _execute_signal(self, signal: TradeSignal, indicators):
        """Execute trade from signal via Advanced Trade Manager."""
        
        # Create trade via trade manager
        trade = self.trade_manager.create_trade_from_signal(signal, indicators)
        
        if trade:
            dashboard.update(
                message=f"TRADE: {trade.direction.value} {trade.symbol} "
                       f"@ ${trade.entry_price:,.2f} | Size: {trade.entry_size:.4f} | "
                       f"SL: ${trade.stop_loss:,.2f} | TP: ${trade.take_profit:,.2f}"
            )
            dashboard.print_signal_alert(signal)
    
    def _load_historical_data(self):
        """Load historical candles for all symbols."""
        print("[SYSTEM] Loading historical data...")
        for symbol in self.symbols:
            success = market_data.load_historical_candles(
                symbol=symbol,
                resolution=self.timeframe,
                lookback_hours=24
            )
            if success:
                self._analyze_and_trade(symbol)
        print("[SYSTEM] Historical data loaded")
    
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
                # Polling mode
                while self.running:
                    self._update_dashboard()
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
