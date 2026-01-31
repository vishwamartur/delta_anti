"""
Main HFT System Orchestrator
Coordinates all components for high-frequency trading
"""
import asyncio
import logging
import signal
import sys
import os
from typing import Dict, List, Optional
import time

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from hft.network.ultra_websocket import UltraWebSocketClient
from hft.data.fast_orderbook import FastOrderBook, OrderBookManager
from hft.strategies.market_maker import HFTMarketMaker, MarketMakerConfig
from hft.strategies.stat_arb import StatisticalArbitrage, StatArbConfig
from hft.monitoring.latency_tracker import LatencyTracker, latency_tracker
from hft.config.hft_config import HFT_CONFIG, HFT_RISK_CONTROLS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HFTSystem:
    """
    High-Frequency Trading System
    Target: <5ms tick-to-trade latency
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or HFT_CONFIG
        self.is_running = False
        
        # Core components
        self.ws_client = UltraWebSocketClient(self.config['exchange_url'])
        self.orderbook_manager = OrderBookManager()
        self.latency_tracker = latency_tracker
        
        # Strategies
        self.market_makers: List[HFTMarketMaker] = []
        self.stat_arb_strategies: List[StatisticalArbitrage] = []
        
        # Tasks
        self.tasks: List[asyncio.Task] = []
        
        # Metrics
        self.start_time = 0
        self.messages_processed = 0
        
        self._initialize_strategies()
        
        logger.info("[HFT] System initialized")
    
    def _initialize_strategies(self):
        """Initialize trading strategies based on config"""
        
        # Market Makers
        if self.config.get('enable_market_making'):
            for symbol in self.config['symbols']:
                mm_config = MarketMakerConfig(
                    symbol=symbol,
                    spread_target_bps=self.config.get('mm_spread_bps', 5.0),
                    quote_size=self.config.get('mm_quote_size', 1.0),
                    max_inventory=self.config.get('mm_max_inventory', 20.0),
                    update_interval_ms=self.config.get('mm_update_frequency_ms', 100),
                    skew_factor=self.config.get('mm_skew_factor', 0.3),
                    min_edge_bps=self.config.get('mm_min_edge_bps', 2.0)
                )
                self.market_makers.append(HFTMarketMaker(mm_config))
                logger.info(f"[HFT] Market maker initialized for {symbol}")
        
        # Statistical Arbitrage
        if self.config.get('enable_stat_arb'):
            for pair in self.config.get('stat_arb_pairs', []):
                arb_config = StatArbConfig(
                    symbol_a=pair[0],
                    symbol_b=pair[1],
                    lookback=self.config.get('stat_arb_lookback', 100),
                    z_threshold=self.config.get('stat_arb_z_threshold', 2.0),
                    exit_z_threshold=self.config.get('stat_arb_exit_z', 0.5)
                )
                self.stat_arb_strategies.append(StatisticalArbitrage(arb_config))
                logger.info(f"[HFT] Stat arb initialized for {pair[0]}/{pair[1]}")
    
    async def start(self):
        """Start the HFT system"""
        self.is_running = True
        self.start_time = time.time()
        
        print("=" * 70)
        print("    HFT SYSTEM - HIGH-FREQUENCY TRADING")
        print("=" * 70)
        print(f"  Symbols: {', '.join(self.config['symbols'])}")
        print(f"  Market Makers: {len(self.market_makers)}")
        print(f"  Stat Arb Pairs: {len(self.stat_arb_strategies)}")
        print(f"  Target Latency: <5ms tick-to-trade")
        print("=" * 70)
        
        # Setup signal handlers
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._signal_handler)
        
        # Connect to exchange
        connected = await self.ws_client.connect()
        if not connected:
            logger.error("[HFT] Failed to connect to exchange")
            return
        
        # Setup callbacks
        self.ws_client.on_orderbook = self._on_orderbook_update
        self.ws_client.on_trade = self._on_trade
        self.ws_client.on_ticker = self._on_ticker
        
        # Subscribe to data feeds
        await self.ws_client.subscribe_orderbook(self.config['symbols'])
        
        print("[HFT] Connected and subscribed to data feeds")
        print("=" * 70)
        
        # Start strategy tasks
        for mm in self.market_makers:
            ob = self.orderbook_manager.get_book(mm.config.symbol)
            task = asyncio.create_task(mm.run(ob))
            self.tasks.append(task)
        
        # Start monitoring
        self.tasks.append(asyncio.create_task(self._monitoring_loop()))
        
        # Start message receiver
        self.tasks.append(asyncio.create_task(self.ws_client.receive_messages()))
        
        # Wait for tasks
        try:
            await asyncio.gather(*self.tasks)
        except asyncio.CancelledError:
            logger.info("[HFT] Tasks cancelled")
    
    async def _on_orderbook_update(self, data: Dict):
        """Handle order book update with latency tracking"""
        self.latency_tracker.mark('market_data_arrival')
        self.messages_processed += 1
        
        # Extract symbol
        symbol = data.get('symbol', '')
        if not symbol:
            # Try Delta format
            if 'product_symbol' in data:
                symbol = data['product_symbol']
        
        if not symbol:
            return
        
        # Get order book
        orderbook = self.orderbook_manager.get_book(symbol)
        
        # Update order book
        start = time.perf_counter_ns()
        
        if 'buy' in data:  # Delta format
            bids = [[float(p), float(q)] for p, q in data.get('buy', [])]
            asks = [[float(p), float(q)] for p, q in data.get('sell', [])]
            orderbook.update_from_snapshot(bids, asks)
        elif 'bids' in data:  # Standard format
            orderbook.update_from_snapshot(data['bids'], data['asks'])
        
        update_time = time.perf_counter_ns() - start
        self.latency_tracker.record_orderbook_update(update_time)
        
        self.latency_tracker.mark('parse_complete')
        
        # Update stat arb strategies
        for arb in self.stat_arb_strategies:
            if symbol in (arb.config.symbol_a, arb.config.symbol_b):
                # Get prices for both symbols
                ob_a = self.orderbook_manager.get_book(arb.config.symbol_a)
                ob_b = self.orderbook_manager.get_book(arb.config.symbol_b)
                
                if ob_a.mid_price > 0 and ob_b.mid_price > 0:
                    arb.update_prices(ob_a.mid_price, ob_b.mid_price)
                    
                    # Check for signals
                    sig = arb.generate_signal()
                    if sig:
                        await self._handle_stat_arb_signal(arb, sig)
        
        self.latency_tracker.mark('strategy_complete')
    
    async def _on_trade(self, data: Dict):
        """Handle trade event"""
        # For fill notifications
        pass
    
    async def _on_ticker(self, data: Dict):
        """Handle ticker update"""
        pass
    
    async def _handle_stat_arb_signal(self, strategy: StatisticalArbitrage, signal):
        """Handle statistical arbitrage signal"""
        if signal.action == 'open_long_spread':
            logger.info(f"[STAT-ARB] LONG SPREAD: z={signal.z_score:.2f}")
            strategy.open_position('long_spread', signal.spread, signal.z_score)
            
        elif signal.action == 'open_short_spread':
            logger.info(f"[STAT-ARB] SHORT SPREAD: z={signal.z_score:.2f}")
            strategy.open_position('short_spread', signal.spread, signal.z_score)
            
        elif signal.action == 'close':
            pnl = strategy.close_position()
            logger.info(f"[STAT-ARB] CLOSE: P&L={pnl:.4f} points")
    
    async def _monitoring_loop(self):
        """Periodic monitoring and statistics"""
        interval = self.config.get('print_stats_interval_sec', 30)
        
        while self.is_running:
            await asyncio.sleep(interval)
            
            uptime = time.time() - self.start_time
            
            print("\n" + "=" * 70)
            print(f"HFT SYSTEM STATUS (uptime: {uptime:.0f}s)")
            print("=" * 70)
            
            # Connection stats
            ws_stats = self.ws_client.get_stats()
            print(f"\nWebSocket: {'Connected' if ws_stats['connected'] else 'Disconnected'}")
            print(f"  Messages: {ws_stats['messages_received']:,}")
            print(f"  Avg Latency: {ws_stats['avg_latency_ms']:.2f}ms")
            print(f"  P99 Latency: {ws_stats['p99_latency_ms']:.2f}ms")
            
            # Order book stats
            print("\nOrder Books:")
            for symbol, ob in self.orderbook_manager.books.items():
                stats = ob.get_stats()
                print(f"  {symbol}: spread={stats['spread_bps']:.1f}bps, "
                      f"updates={stats['update_count']}, "
                      f"avg_update={stats['avg_update_ns']:.0f}ns")
            
            # Market maker stats
            if self.market_makers:
                print("\nMarket Makers:")
                for mm in self.market_makers:
                    stats = mm.get_stats()
                    print(f"  {stats['symbol']}: inv={stats['inventory']:.2f}, "
                          f"quotes={stats['quotes_posted']}, fills={stats['fills']}")
            
            # Stat arb stats
            if self.stat_arb_strategies:
                print("\nStat Arb:")
                for arb in self.stat_arb_strategies:
                    stats = arb.get_stats()
                    print(f"  {stats['pair']}: z={stats['current_z']:.2f}, "
                          f"pos={stats['position'] or 'flat'}, "
                          f"pnl={stats['pnl_points']:.2f}")
            
            # Latency report
            self.latency_tracker.print_report()
            
            # Check alerts
            alerts = self.latency_tracker.check_alerts(
                threshold_us=self.config.get('latency_alert_threshold_ms', 10) * 1000
            )
            if alerts:
                print("\n!!! ALERTS !!!")
                for alert in alerts:
                    print(f"  - {alert}")
            
            print("=" * 70)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signal"""
        print("\n[HFT] Shutdown signal received...")
        asyncio.create_task(self.stop())
    
    async def stop(self):
        """Stop HFT system gracefully"""
        self.is_running = False
        
        # Stop strategies
        for mm in self.market_makers:
            mm.stop()
        
        # Cancel tasks
        for task in self.tasks:
            task.cancel()
        
        # Close connection
        await self.ws_client.close()
        
        print("\n[HFT] System stopped")
        
        # Final stats
        uptime = time.time() - self.start_time
        print(f"\n=== FINAL STATISTICS (uptime: {uptime:.0f}s) ===")
        print(f"Messages Processed: {self.messages_processed:,}")
        
        for mm in self.market_makers:
            stats = mm.get_stats()
            print(f"MM {stats['symbol']}: {stats['quotes_posted']} quotes, "
                  f"{stats['fills']} fills, P&L=${stats['gross_pnl']:.2f}")
        
        for arb in self.stat_arb_strategies:
            stats = arb.get_stats()
            print(f"Arb {stats['pair']}: {stats['trades_closed']} trades, "
                  f"P&L={stats['pnl_points']:.2f} points")


async def main():
    """Main entry point"""
    system = HFTSystem()
    await system.start()


if __name__ == "__main__":
    asyncio.run(main())
