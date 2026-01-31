"""
Statistical Arbitrage Strategy
Exploits temporary price discrepancies between correlated assets
"""
from collections import deque
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import time

# Try scipy/numpy for faster calculations
try:
    import numpy as np
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)


@dataclass
class StatArbConfig:
    """Statistical arbitrage configuration"""
    symbol_a: str
    symbol_b: str
    lookback: int = 100           # Window for calculating spread statistics
    z_threshold: float = 2.0      # Z-score threshold for entry
    exit_z_threshold: float = 0.5  # Z-score threshold for exit
    max_position_size: float = 5.0
    hedge_ratio_update_freq: int = 50  # Update beta every N observations


@dataclass
class StatArbSignal:
    """Statistical arbitrage signal"""
    action: str              # 'open_long_spread', 'open_short_spread', 'close'
    symbol_a: str
    symbol_b: str
    z_score: float
    spread: float
    hedge_ratio: float
    confidence: float
    timestamp: int


class StatisticalArbitrage:
    """
    HFT Statistical Arbitrage
    Trades mean-reversion opportunities in spread between correlated pairs
    
    Long spread = Buy A, Sell B
    Short spread = Sell A, Buy B
    """
    
    def __init__(self, config: StatArbConfig):
        self.config = config
        
        # Price history for spread calculation
        self.prices_a: deque = deque(maxlen=config.lookback)
        self.prices_b: deque = deque(maxlen=config.lookback)
        self.spreads: deque = deque(maxlen=config.lookback)
        self.timestamps: deque = deque(maxlen=config.lookback)
        
        # Spread statistics
        self.spread_mean = 0.0
        self.spread_std = 0.0
        self.current_z_score = 0.0
        self.hedge_ratio = 1.0  # beta
        
        # Position state
        self.position: Optional[str] = None  # 'long_spread' or 'short_spread'
        self.entry_spread = 0.0
        self.entry_z = 0.0
        self.entry_time = 0
        
        # Performance metrics
        self.signals_generated = 0
        self.trades_opened = 0
        self.trades_closed = 0
        self.pnl_points = 0.0
        
        self._update_counter = 0
        
        logger.info(f"[STAT-ARB] Initialized: {config.symbol_a}/{config.symbol_b}")
    
    def update_prices(self, price_a: float, price_b: float):
        """Update price history and recalculate spread statistics"""
        current_time = time.perf_counter_ns()
        
        self.prices_a.append(price_a)
        self.prices_b.append(price_b)
        self.timestamps.append(current_time)
        self._update_counter += 1
        
        # Need minimum data
        if len(self.prices_a) < 20:
            return
        
        # Update hedge ratio periodically
        if self._update_counter % self.config.hedge_ratio_update_freq == 0:
            self._update_hedge_ratio()
        
        # Calculate current spread
        spread = price_a - self.hedge_ratio * price_b
        self.spreads.append(spread)
        
        # Update spread statistics
        if len(self.spreads) >= 20:
            if HAS_SCIPY:
                self.spread_mean = float(np.mean(self.spreads))
                self.spread_std = float(np.std(self.spreads))
            else:
                spreads_list = list(self.spreads)
                self.spread_mean = sum(spreads_list) / len(spreads_list)
                self.spread_std = (sum((x - self.spread_mean)**2 for x in spreads_list) 
                                  / len(spreads_list)) ** 0.5
            
            if self.spread_std > 0:
                self.current_z_score = (spread - self.spread_mean) / self.spread_std
    
    def _update_hedge_ratio(self):
        """Calculate hedge ratio (beta) using linear regression"""
        if len(self.prices_a) < 20:
            return
        
        if HAS_SCIPY:
            x = np.array(self.prices_b)
            y = np.array(self.prices_a)
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            self.hedge_ratio = slope
        else:
            # Simple ratio for fallback
            self.hedge_ratio = sum(self.prices_a) / sum(self.prices_b)
    
    def generate_signal(self) -> Optional[StatArbSignal]:
        """
        Generate trading signal based on z-score
        Returns signal if trade should be executed
        """
        if len(self.spreads) < 20:
            return None
        
        current_spread = self.spreads[-1]
        z = self.current_z_score
        
        # Calculate confidence based on z-score magnitude
        confidence = min(abs(z) / 3.0, 1.0)  # Max confidence at z=3
        
        # No position: check for entry
        if self.position is None:
            if z > self.config.z_threshold:
                # Spread too high: short spread (sell A, buy B)
                self.signals_generated += 1
                return StatArbSignal(
                    action='open_short_spread',
                    symbol_a=self.config.symbol_a,
                    symbol_b=self.config.symbol_b,
                    z_score=z,
                    spread=current_spread,
                    hedge_ratio=self.hedge_ratio,
                    confidence=confidence,
                    timestamp=time.perf_counter_ns()
                )
                
            elif z < -self.config.z_threshold:
                # Spread too low: long spread (buy A, sell B)
                self.signals_generated += 1
                return StatArbSignal(
                    action='open_long_spread',
                    symbol_a=self.config.symbol_a,
                    symbol_b=self.config.symbol_b,
                    z_score=z,
                    spread=current_spread,
                    hedge_ratio=self.hedge_ratio,
                    confidence=confidence,
                    timestamp=time.perf_counter_ns()
                )
        
        # Has position: check for exit
        else:
            # Exit if spread reverts to mean
            if abs(z) < self.config.exit_z_threshold:
                pnl = self._calculate_pnl()
                self.signals_generated += 1
                return StatArbSignal(
                    action='close',
                    symbol_a=self.config.symbol_a,
                    symbol_b=self.config.symbol_b,
                    z_score=z,
                    spread=current_spread,
                    hedge_ratio=self.hedge_ratio,
                    confidence=1.0,
                    timestamp=time.perf_counter_ns()
                )
            
            # Stop loss: if z-score moves further against us
            if self.position == 'long_spread' and z < -3.0:
                return StatArbSignal(
                    action='close',
                    symbol_a=self.config.symbol_a,
                    symbol_b=self.config.symbol_b,
                    z_score=z,
                    spread=current_spread,
                    hedge_ratio=self.hedge_ratio,
                    confidence=1.0,
                    timestamp=time.perf_counter_ns()
                )
            elif self.position == 'short_spread' and z > 3.0:
                return StatArbSignal(
                    action='close',
                    symbol_a=self.config.symbol_a,
                    symbol_b=self.config.symbol_b,
                    z_score=z,
                    spread=current_spread,
                    hedge_ratio=self.hedge_ratio,
                    confidence=1.0,
                    timestamp=time.perf_counter_ns()
                )
        
        return None
    
    def _calculate_pnl(self) -> float:
        """Calculate P&L in spread points"""
        if not self.spreads:
            return 0.0
        
        current_spread = self.spreads[-1]
        
        if self.position == 'long_spread':
            return current_spread - self.entry_spread
        elif self.position == 'short_spread':
            return self.entry_spread - current_spread
        
        return 0.0
    
    def open_position(self, position_type: str, spread: float, z_score: float):
        """Record position opening"""
        self.position = position_type
        self.entry_spread = spread
        self.entry_z = z_score
        self.entry_time = time.perf_counter_ns()
        self.trades_opened += 1
        
        logger.info(f"[STAT-ARB] Opened {position_type}: spread={spread:.4f}, z={z_score:.2f}")
    
    def close_position(self) -> float:
        """Close position and return P&L"""
        pnl = self._calculate_pnl()
        self.pnl_points += pnl
        self.trades_closed += 1
        
        logger.info(f"[STAT-ARB] Closed {self.position}: P&L={pnl:.4f} points")
        
        self.position = None
        self.entry_spread = 0.0
        self.entry_z = 0.0
        self.entry_time = 0
        
        return pnl
    
    def get_stats(self) -> Dict:
        """Get strategy statistics"""
        return {
            'pair': f"{self.config.symbol_a}/{self.config.symbol_b}",
            'position': self.position,
            'current_z': self.current_z_score,
            'spread_mean': self.spread_mean,
            'spread_std': self.spread_std,
            'hedge_ratio': self.hedge_ratio,
            'signals': self.signals_generated,
            'trades_opened': self.trades_opened,
            'trades_closed': self.trades_closed,
            'pnl_points': self.pnl_points,
            'unrealized_pnl': self._calculate_pnl() if self.position else 0.0
        }
