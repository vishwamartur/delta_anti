"""
HFT Market Making Strategy
Continuously quotes bid and ask to capture spread
"""
import asyncio
from typing import Dict, Optional, List
import logging
from dataclasses import dataclass
import time

from hft.data.fast_orderbook import FastOrderBook
from hft.monitoring.latency_tracker import latency_tracker

logger = logging.getLogger(__name__)


@dataclass
class MarketMakerConfig:
    """Market maker configuration"""
    symbol: str
    spread_target_bps: float = 5.0       # 5 basis points target spread
    max_position_size: float = 10.0      # Maximum position in contracts
    quote_size: float = 1.0              # Size per quote
    skew_factor: float = 0.3             # Position skew adjustment
    min_edge_bps: float = 2.0            # Minimum edge required
    max_inventory: float = 20.0          # Max inventory before stopping
    inventory_penalty_bps: float = 1.0   # Penalty per unit inventory
    update_interval_ms: float = 100      # Quote update frequency


class HFTMarketMaker:
    """
    High-frequency market making strategy
    Target: Quote updates <1ms, profit from bid-ask spread
    """
    
    def __init__(self, config: MarketMakerConfig):
        self.config = config
        self.orderbook: Optional[FastOrderBook] = None
        
        # Position tracking
        self.inventory = 0.0    # Net position (positive = long, negative = short)
        self.active_quotes = {'bid': None, 'ask': None}
        self.active_order_ids = {'bid': None, 'ask': None}
        
        # Performance metrics
        self.quotes_posted = 0
        self.quotes_cancelled = 0
        self.fills_received = 0
        self.gross_pnl = 0.0
        self.fees_paid = 0.0
        
        # State
        self.is_active = True
        self.last_quote_time = 0
        self.pause_until = 0
        
        logger.info(f"[MM] Market maker initialized for {config.symbol}")
    
    async def run(self, orderbook: FastOrderBook):
        """Main market making loop"""
        self.orderbook = orderbook
        
        logger.info(f"[MM] Starting market maker for {self.config.symbol}")
        
        while self.is_active:
            try:
                loop_start = time.perf_counter_ns()
                
                # Check pause
                if time.time() < self.pause_until:
                    await asyncio.sleep(0.1)
                    continue
                
                # Calculate fair value
                fair_value = self.orderbook.mid_price
                if fair_value == 0:
                    await asyncio.sleep(0.001)  # 1ms
                    continue
                
                # Check risk limits
                if not self._check_risk_limits():
                    await asyncio.sleep(0.1)
                    continue
                
                # Calculate quote prices with inventory adjustment
                bid_price, ask_price = self._calculate_quote_prices(fair_value)
                
                # Update quotes if changed
                await self._update_quotes(bid_price, ask_price)
                
                # Track strategy latency
                strategy_time = time.perf_counter_ns() - loop_start
                latency_tracker.strategy_latency.append(strategy_time)
                
                # Wait for next update interval
                elapsed_ms = (time.perf_counter_ns() - loop_start) / 1_000_000
                wait_ms = max(0.1, self.config.update_interval_ms - elapsed_ms)
                await asyncio.sleep(wait_ms / 1000)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[MM] Market maker error: {e}")
                await asyncio.sleep(1)
        
        # Cleanup
        await self._cancel_quotes()
        logger.info(f"[MM] Market maker stopped for {self.config.symbol}")
    
    def _calculate_quote_prices(self, fair_value: float) -> tuple:
        """
        Calculate optimal bid/ask prices
        Includes: spread, inventory skew, market microstructure
        """
        # Base spread in dollars
        spread_bps = self.config.spread_target_bps
        half_spread = fair_value * (spread_bps / 10000) / 2
        
        # Inventory skew: shift quotes away from current position
        # If long, lower bid to reduce buying, raise ask to encourage selling
        inventory_ratio = self.inventory / self.config.max_inventory
        skew = inventory_ratio * self.config.skew_factor * half_spread
        
        # Order book imbalance adjustment
        # If more buy pressure, raise quotes slightly
        imbalance = self.orderbook.get_imbalance(levels=5)
        imbalance_adj = imbalance * 0.1 * half_spread
        
        # Inventory penalty: widen spread when inventory is high
        inv_penalty = abs(inventory_ratio) * self.config.inventory_penalty_bps * fair_value / 10000
        
        # Final quote prices
        bid_price = fair_value - half_spread - skew + imbalance_adj - inv_penalty
        ask_price = fair_value + half_spread - skew + imbalance_adj + inv_penalty
        
        # Ensure minimum edge
        min_edge = fair_value * (self.config.min_edge_bps / 10000)
        if ask_price - bid_price < min_edge:
            mid = (bid_price + ask_price) / 2
            bid_price = mid - min_edge / 2
            ask_price = mid + min_edge / 2
        
        # Round to tick size (assume 0.01 for now)
        bid_price = round(bid_price, 2)
        ask_price = round(ask_price, 2)
        
        return (bid_price, ask_price)
    
    async def _update_quotes(self, bid_price: float, ask_price: float):
        """
        Update quotes on exchange
        Fast path: only update if prices changed significantly
        """
        price_threshold = bid_price * 0.0001  # 1 bps threshold
        
        update_bid = True
        update_ask = True
        
        # Check if update needed (avoid unnecessary API calls)
        if self.active_quotes['bid']:
            if abs(self.active_quotes['bid'] - bid_price) < price_threshold:
                update_bid = False
        
        if self.active_quotes['ask']:
            if abs(self.active_quotes['ask'] - ask_price) < price_threshold:
                update_ask = False
        
        if not update_bid and not update_ask:
            return
        
        # Cancel old quotes first (in practice, use modify order)
        if update_bid or update_ask:
            await self._cancel_quotes()
        
        # Place new quotes
        if update_bid:
            # In production: place actual order via API
            self.active_quotes['bid'] = bid_price
            self.quotes_posted += 1
            
        if update_ask:
            self.active_quotes['ask'] = ask_price
            self.quotes_posted += 1
        
        self.last_quote_time = time.time()
    
    async def _cancel_quotes(self):
        """Cancel existing quotes"""
        # In production: cancel via API
        if self.active_quotes['bid']:
            self.quotes_cancelled += 1
        if self.active_quotes['ask']:
            self.quotes_cancelled += 1
        
        self.active_quotes = {'bid': None, 'ask': None}
        self.active_order_ids = {'bid': None, 'ask': None}
    
    def _check_risk_limits(self) -> bool:
        """Check if within risk limits"""
        # Inventory limit
        if abs(self.inventory) >= self.config.max_inventory:
            logger.warning(f"[MM] Max inventory reached: {self.inventory}")
            return False
        
        return True
    
    def on_fill(self, side: str, price: float, quantity: float):
        """Handle order fill notification"""
        if side == 'buy':
            self.inventory += quantity
            self.gross_pnl -= price * quantity  # Cash outflow
        else:
            self.inventory -= quantity
            self.gross_pnl += price * quantity  # Cash inflow
        
        self.fills_received += 1
        
        logger.info(f"[MM] Fill: {side} {quantity:.4f} @ ${price:.2f}, "
                   f"Inventory: {self.inventory:.4f}")
    
    def get_stats(self) -> Dict:
        """Get strategy statistics"""
        return {
            'symbol': self.config.symbol,
            'inventory': self.inventory,
            'quotes_posted': self.quotes_posted,
            'quotes_cancelled': self.quotes_cancelled,
            'fills': self.fills_received,
            'gross_pnl': self.gross_pnl,
            'active_bid': self.active_quotes['bid'],
            'active_ask': self.active_quotes['ask'],
            'is_active': self.is_active
        }
    
    def stop(self):
        """Stop the market maker"""
        self.is_active = False
