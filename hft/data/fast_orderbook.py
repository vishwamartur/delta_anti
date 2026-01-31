"""
Lock-free order book with microsecond update times
Optimized for HFT with sorted containers and minimal allocations
"""
from sortedcontainers import SortedDict
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class PriceLevel:
    """Single price level in order book"""
    price: float
    quantity: float
    order_count: int = 1
    timestamp: int = 0  # Nanosecond precision


class FastOrderBook:
    """
    Ultra-fast order book implementation
    Target: <100 nanoseconds per update
    """
    
    def __init__(self, symbol: str, max_depth: int = 100):
        self.symbol = symbol
        self.max_depth = max_depth
        
        # SortedDict for O(log n) inserts, maintains sorted order
        self.bids: SortedDict = SortedDict()  # price -> PriceLevel (descending)
        self.asks: SortedDict = SortedDict()  # price -> PriceLevel (ascending)
        
        # Best bid/ask cache for O(1) access
        self._best_bid: Optional[float] = None
        self._best_ask: Optional[float] = None
        self._spread: float = 0.0
        self._mid_price: float = 0.0
        
        # Metrics
        self.last_update_time = 0
        self.update_count = 0
        self.total_update_time_ns = 0
    
    def update_bid(self, price: float, quantity: float):
        """
        Update bid side (buy orders)
        Optimized path: inline operations, minimal branches
        """
        start = time.perf_counter_ns()
        
        if quantity > 0:
            # Add or update price level
            if price in self.bids:
                self.bids[price].quantity = quantity
                self.bids[price].timestamp = start
            else:
                self.bids[price] = PriceLevel(price, quantity, 1, start)
                
                # Prune depth if needed
                if len(self.bids) > self.max_depth:
                    self.bids.popitem(0)  # Remove worst bid (lowest)
        else:
            # Remove price level
            self.bids.pop(price, None)
        
        # Update best bid cache
        if self.bids:
            self._best_bid = self.bids.peekitem(-1)[0]  # Last key (highest)
        else:
            self._best_bid = None
        
        self._update_derived_metrics()
        
        # Track performance
        elapsed = time.perf_counter_ns() - start
        self.total_update_time_ns += elapsed
        self.update_count += 1
    
    def update_ask(self, price: float, quantity: float):
        """Update ask side (sell orders)"""
        start = time.perf_counter_ns()
        
        if quantity > 0:
            if price in self.asks:
                self.asks[price].quantity = quantity
                self.asks[price].timestamp = start
            else:
                self.asks[price] = PriceLevel(price, quantity, 1, start)
                
                if len(self.asks) > self.max_depth:
                    self.asks.popitem(-1)  # Remove worst ask (highest)
        else:
            self.asks.pop(price, None)
        
        # Update best ask cache
        if self.asks:
            self._best_ask = self.asks.peekitem(0)[0]  # First key (lowest)
        else:
            self._best_ask = None
        
        self._update_derived_metrics()
        
        elapsed = time.perf_counter_ns() - start
        self.total_update_time_ns += elapsed
        self.update_count += 1
    
    def update_from_snapshot(self, bids: List[List], asks: List[List]):
        """
        Update order book from full snapshot
        Format: [[price, quantity], ...]
        """
        start = time.perf_counter_ns()
        
        # Clear existing data
        self.bids.clear()
        self.asks.clear()
        
        # Load bids
        for bid in bids[:self.max_depth]:
            price, qty = float(bid[0]), float(bid[1])
            if qty > 0:
                self.bids[price] = PriceLevel(price, qty, 1, start)
        
        # Load asks
        for ask in asks[:self.max_depth]:
            price, qty = float(ask[0]), float(ask[1])
            if qty > 0:
                self.asks[price] = PriceLevel(price, qty, 1, start)
        
        # Update caches
        self._best_bid = self.bids.peekitem(-1)[0] if self.bids else None
        self._best_ask = self.asks.peekitem(0)[0] if self.asks else None
        self._update_derived_metrics()
        
        elapsed = time.perf_counter_ns() - start
        self.total_update_time_ns += elapsed
        self.update_count += 1
    
    def _update_derived_metrics(self):
        """Update spread and mid price (inline for speed)"""
        if self._best_bid and self._best_ask:
            self._spread = self._best_ask - self._best_bid
            self._mid_price = (self._best_bid + self._best_ask) / 2
        else:
            self._spread = 0.0
            self._mid_price = 0.0
        
        self.last_update_time = time.perf_counter_ns()
    
    @property
    def best_bid(self) -> Optional[Tuple[float, float]]:
        """Get best bid (price, quantity) in O(1)"""
        if self._best_bid and self._best_bid in self.bids:
            level = self.bids[self._best_bid]
            return (level.price, level.quantity)
        return None
    
    @property
    def best_ask(self) -> Optional[Tuple[float, float]]:
        """Get best ask (price, quantity) in O(1)"""
        if self._best_ask and self._best_ask in self.asks:
            level = self.asks[self._best_ask]
            return (level.price, level.quantity)
        return None
    
    @property
    def spread(self) -> float:
        """Get spread in O(1)"""
        return self._spread
    
    @property
    def spread_bps(self) -> float:
        """Get spread in basis points"""
        if self._mid_price > 0:
            return (self._spread / self._mid_price) * 10000
        return 0.0
    
    @property
    def mid_price(self) -> float:
        """Get mid price in O(1)"""
        return self._mid_price
    
    def get_depth(self, levels: int = 10) -> Dict:
        """Get order book depth (top N levels)"""
        bid_levels = []
        ask_levels = []
        
        # Get top bids (highest prices)
        for price in reversed(list(self.bids.keys())[-levels:]):
            level = self.bids[price]
            bid_levels.append([level.price, level.quantity])
        
        # Get top asks (lowest prices)
        for price in list(self.asks.keys())[:levels]:
            level = self.asks[price]
            ask_levels.append([level.price, level.quantity])
        
        return {
            'bids': bid_levels,
            'asks': ask_levels,
            'spread': self._spread,
            'spread_bps': self.spread_bps,
            'mid': self._mid_price
        }
    
    def get_average_update_time_ns(self) -> float:
        """Get average update time in nanoseconds"""
        if self.update_count == 0:
            return 0.0
        return self.total_update_time_ns / self.update_count
    
    def calculate_vwap(self, side: str, quantity: float) -> float:
        """
        Calculate Volume-Weighted Average Price for given quantity
        Used for market impact estimation
        """
        levels = self.bids if side == 'bid' else self.asks
        
        remaining = quantity
        total_cost = 0.0
        
        if side == 'bid':
            prices = reversed(list(levels.keys()))
        else:
            prices = list(levels.keys())
        
        for price in prices:
            level = levels[price]
            take_qty = min(remaining, level.quantity)
            total_cost += price * take_qty
            remaining -= take_qty
            
            if remaining <= 0:
                break
        
        if remaining > 0:
            # Not enough liquidity
            return 0.0
        
        return total_cost / quantity
    
    def get_imbalance(self, levels: int = 5) -> float:
        """
        Calculate order book imbalance
        Positive = more bid pressure, Negative = more ask pressure
        Range: -1.0 to +1.0
        """
        bid_keys = list(self.bids.keys())[-levels:]
        ask_keys = list(self.asks.keys())[:levels]
        
        bid_vol = sum(self.bids[p].quantity for p in bid_keys)
        ask_vol = sum(self.asks[p].quantity for p in ask_keys)
        
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        
        return (bid_vol - ask_vol) / total
    
    def get_weighted_mid(self, levels: int = 5) -> float:
        """Get volume-weighted mid price"""
        bid_keys = list(self.bids.keys())[-levels:]
        ask_keys = list(self.asks.keys())[:levels]
        
        bid_sum = sum(self.bids[p].price * self.bids[p].quantity for p in bid_keys)
        ask_sum = sum(self.asks[p].price * self.asks[p].quantity for p in ask_keys)
        
        bid_vol = sum(self.bids[p].quantity for p in bid_keys)
        ask_vol = sum(self.asks[p].quantity for p in ask_keys)
        
        total_vol = bid_vol + ask_vol
        if total_vol == 0:
            return self._mid_price
        
        return (bid_sum + ask_sum) / total_vol
    
    def get_stats(self) -> Dict:
        """Get order book statistics"""
        return {
            'symbol': self.symbol,
            'best_bid': self.best_bid,
            'best_ask': self.best_ask,
            'mid_price': self._mid_price,
            'spread': self._spread,
            'spread_bps': self.spread_bps,
            'imbalance': self.get_imbalance(),
            'bid_levels': len(self.bids),
            'ask_levels': len(self.asks),
            'update_count': self.update_count,
            'avg_update_ns': self.get_average_update_time_ns()
        }


# Order book manager
class OrderBookManager:
    """Manage multiple order books"""
    
    def __init__(self):
        self.books: Dict[str, FastOrderBook] = {}
    
    def get_book(self, symbol: str) -> FastOrderBook:
        """Get or create order book for symbol"""
        if symbol not in self.books:
            self.books[symbol] = FastOrderBook(symbol)
        return self.books[symbol]
    
    def update_orderbook(self, symbol: str, bids: List, asks: List):
        """Update order book from data"""
        book = self.get_book(symbol)
        book.update_from_snapshot(bids, asks)
    
    def get_all_stats(self) -> Dict:
        """Get stats for all order books"""
        return {sym: book.get_stats() for sym, book in self.books.items()}


# Singleton
orderbook_manager = OrderBookManager()
