"""
Delta Exchange Real-Time Trading System
Main Application Entry Point
"""
import time
import signal
import sys
from datetime import datetime
from typing import Dict

import config
from api.delta_rest import rest_client
from api.delta_websocket import ws_client
from data.market_data import market_data
from analysis.indicators import indicators, IndicatorValues
from analysis.signals import signal_generator, TradeSignal, SignalType
from strategy.trade_manager import trade_manager
from ui.dashboard import dashboard


class TradingSystem:
    """
    Main trading system controller.
    Coordinates all components for real-time trading analysis.
    """
    
    def __init__(self):
        self.running = False
        self.symbols = config.TRADING_SYMBOLS
        self.timeframe = config.DEFAULT_TIMEFRAME
        
        # Latest data
        self._indicators: Dict[str, IndicatorValues] = {}
        self._signals: Dict[str, TradeSignal] = {}
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print("\n[SYSTEM] Shutting down...")
        self.stop()
        sys.exit(0)
    
    def _on_ticker(self, data: Dict):
        """Handle ticker updates from WebSocket."""
        market_data.update_ticker(data)
    
    def _on_candlestick(self, data: Dict):
        """Handle candlestick updates from WebSocket."""
        market_data.update_from_ws_candlestick(data)
        
        symbol = data.get('symbol', '')
        if symbol and not symbol.startswith('MARK:'):
            # Recalculate indicators on new candle
            self._analyze_symbol(symbol)
    
    def _analyze_symbol(self, symbol: str):
        """Analyze a symbol and generate signals."""
        df = market_data.get_dataframe(symbol)
        if df is None or len(df) < 50:
            return
        
        # Calculate indicators
        ind = indicators.calculate_all(df)
        if ind is None:
            return
        
        self._indicators[symbol] = ind
        
        # Check for exit signals on open positions
        trade = trade_manager.get_open_trade(symbol)
        if trade:
            # Check SL/TP first
            exit_reason = trade_manager.check_exit_conditions(symbol, ind.price)
            if exit_reason:
                self._handle_exit(symbol, ind.price, exit_reason)
                return
            
            # Check signal-based exit
            exit_signal = signal_generator.check_exit_signal(
                symbol, ind, trade.is_long, trade.entry_price
            )
            if exit_signal and exit_signal.confidence >= 65:
                self._handle_exit(symbol, ind.price, ", ".join(exit_signal.reasons[:2]))
                return
            
            # Update P&L
            trade.update_pnl(ind.price)
        
        # Generate entry signals if no position
        if not trade_manager.has_open_position(symbol):
            signal = signal_generator.generate_signal(symbol, ind, df)
            self._signals[symbol] = signal
            
            if signal.is_actionable():
                self._handle_signal(signal)
    
    def _handle_signal(self, signal: TradeSignal):
        """Handle an actionable trade signal."""
        dashboard.update(message=f"Signal: {signal.signal_type.value} {signal.symbol} @ ${signal.entry_price:,.2f}")
        dashboard.print_signal_alert(signal)
        
        # Create trade (not executed yet - for tracking)
        trade = trade_manager.create_trade_from_signal(signal)
        if trade:
            # For now, we track signals but don't auto-execute
            # Uncomment below to auto-place orders:
            # self._execute_trade(trade)
            pass
    
    def _handle_exit(self, symbol: str, price: float, reason: str):
        """Handle exit of a position."""
        trade = trade_manager.close_trade(symbol, price, reason)
        if trade:
            dashboard.update(message=f"Exit: {symbol} P&L {trade.pnl_percent:+.2f}%")
            dashboard.print_exit_alert(trade, reason)
    
    def _execute_trade(self, trade):
        """Execute a trade via API."""
        # This would place actual orders
        # For now, just mark as open for simulation
        trade_manager.open_trade(trade.id, trade.entry_price)
    
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
                self._analyze_symbol(symbol)
        print("[SYSTEM] Historical data loaded")
    
    def _setup_websocket(self):
        """Setup WebSocket connections and subscriptions."""
        # Set callback handlers
        ws_client.on_ticker = self._on_ticker
        ws_client.on_candlestick = self._on_candlestick
        
        # Start WebSocket
        ws_client.start()
        
        # Wait for connection
        time.sleep(2)
        
        if ws_client.is_connected:
            # Subscribe to channels
            ws_client.subscribe_ticker(self.symbols)
            ws_client.subscribe_candlestick(self.symbols, self.timeframe)
            print(f"[SYSTEM] Subscribed to {self.symbols}")
        else:
            print("[SYSTEM] WebSocket connection failed - running in REST-only mode")
    
    def _update_dashboard(self):
        """Update dashboard with latest data."""
        # Get prices
        prices = {}
        for symbol in self.symbols:
            price = market_data.get_latest_price(symbol)
            if price:
                prices[symbol] = price
        
        # Update trade P&L
        trade_manager.update_all_trades(prices)
        
        # Update dashboard
        dashboard.update(
            prices=prices,
            indicators=self._indicators,
            signals=self._signals,
            trades=trade_manager.open_trades,
            stats=trade_manager.get_stats()
        )
    
    def _polling_mode(self):
        """Run in polling mode (REST API only)."""
        print("[SYSTEM] Running in polling mode...")
        
        while self.running:
            try:
                for symbol in self.symbols:
                    # Fetch latest candles
                    end_time = int(time.time())
                    start_time = end_time - 3600  # Last hour
                    
                    response = rest_client.get_candles(
                        symbol=symbol,
                        resolution=self.timeframe,
                        start=start_time,
                        end=end_time
                    )
                    
                    if 'result' in response:
                        # Update market data with latest candles
                        for c in response['result'][-5:]:  # Last 5 candles
                            from data.market_data import Candle
                            candle = Candle(
                                timestamp=c.get('time', 0),
                                open=float(c.get('open', 0)),
                                high=float(c.get('high', 0)),
                                low=float(c.get('low', 0)),
                                close=float(c.get('close', 0)),
                                volume=float(c.get('volume', 0))
                            )
                            market_data.add_candle(symbol, candle)
                        
                        self._analyze_symbol(symbol)
                
                self._update_dashboard()
                time.sleep(10)  # Poll every 10 seconds
                
            except Exception as e:
                dashboard.update(message=f"Error: {str(e)[:50]}")
                time.sleep(5)
    
    def start(self):
        """Start the trading system."""
        print("=" * 60)
        print("DELTA EXCHANGE TRADING SYSTEM")
        print("=" * 60)
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Timeframe: {self.timeframe}")
        print(f"Risk per trade: {config.RISK_PER_TRADE * 100}%")
        print("=" * 60)
        
        self.running = True
        
        # Load historical data first
        self._load_historical_data()
        
        # Setup WebSocket
        self._setup_websocket()
        
        # Initial dashboard update
        self._update_dashboard()
        
        print("\n[SYSTEM] Starting dashboard... (Press Ctrl+C to exit)\n")
        time.sleep(1)
        
        # Run dashboard with update callback
        try:
            if ws_client.is_connected:
                # WebSocket mode - dashboard updates on data callbacks
                dashboard.run_live(
                    update_callback=self._update_dashboard,
                    refresh_rate=config.REFRESH_RATE
                )
            else:
                # Polling mode - fetch data periodically
                dashboard.run_live(
                    update_callback=None,
                    refresh_rate=config.REFRESH_RATE
                )
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
    
    def stop(self):
        """Stop the trading system."""
        self.running = False
        ws_client.stop()
        print("\n[SYSTEM] Trading system stopped")


def main():
    """Main entry point."""
    system = TradingSystem()
    system.start()


if __name__ == "__main__":
    main()
